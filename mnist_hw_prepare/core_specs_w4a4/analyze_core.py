"""Read-only, device-independent W4A4 core specification and folding proposals."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


def divisors(n):
    return [v for v in range(1, n + 1) if n % v == 0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_widths(inst, op):
    # Abstract SWG lacks backend-specific use_parallel_window_output().
    # Use the explicitly stated serial-window interface assumption here.
    if op == 'ConvolutionInputGenerator':
        simd = inst.get_nodeattr('SIMD')
        return (simd * inst.get_input_datatype().bitwidth(),
                simd * inst.get_output_datatype().bitwidth())
    return int(inst.get_instream_width()), int(inst.get_outstream_width())


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(inst, pe, simd):
    mw, mh = inst.get_nodeattr('MW'), inst.get_nodeattr('MH')
    vectors = math.prod(inst.get_nodeattr('numInputVectors'))
    if pe not in divisors(mh) or simd not in divisors(mw):
        raise ValueError('Invalid MVAU divisor')
    return dict(PE=pe, SIMD=simd, neuron_folds=mh // pe, synapse_folds=mw // simd,
                ideal_mac_iterations_per_image=vectors * (mh // pe) * (mw // simd),
                parallel_mac_lanes=pe * simd,
                input_stream_bits=inst.get_input_datatype().bitwidth() * simd,
                output_stream_bits=inst.get_output_datatype().bitwidth() * pe,
                weight_bits_per_pe=inst.get_weight_datatype().bitwidth() * simd)


def run(a):
    source = Path(a.model).resolve()
    original_hash = digest(source)
    model = ModelWrapper(str(source))
    if Counter(n.op_type for n in model.graph.node) != Counter(
            {'ConvolutionInputGenerator': 2, 'MVAU': 4, 'StreamingMaxPool': 2}):
        raise ValueError('Expected the eight-node W4A4 hardware core')
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    specs, candidates = [], []
    for index, node in enumerate(model.graph.node):
        inst = getCustomOp(node)
        inp, output = node.input[0], node.output[0]
        row = dict(index=index, name=node.name, op=node.op_type,
                   input_shape=model.get_tensor_shape(inp), output_shape=model.get_tensor_shape(output),
                   input_type=str(inst.get_input_datatype()), output_type=str(inst.get_output_datatype()),
                   current_input_stream_bits=stream_widths(inst, node.op_type)[0],
                   current_output_stream_bits=stream_widths(inst, node.op_type)[1],
                   explicit_attributes={k.name: helper.get_attribute_value(k) for k in node.attribute})
        row['explicit_attributes'] = {
            k: (v.decode() if isinstance(v, bytes) else v)
            for k, v in row['explicit_attributes'].items()}
        if node.op_type == 'MVAU':
            mw, mh = inst.get_nodeattr('MW'), inst.get_nodeattr('MH')
            w = model.get_initializer(node.input[1])
            if w.shape != (mw, mh) or str(inst.get_weight_datatype()) != 'INT4':
                raise AssertionError('Unexpected W4A4 weights')
            row.update(MW=mw, MH=mh, weight_type=str(inst.get_weight_datatype()),
                       weight_count=int(w.size), weight_shape=list(w.shape),
                       accumulator_type=inst.get_nodeattr('accDataType'),
                       accumulator_note='Attribute/default, not final synthesized accumulator width',
                       input_vectors=inst.get_nodeattr('numInputVectors'),
                       no_activation=bool(inst.get_nodeattr('noActivation')),
                       PE_candidates=divisors(mh), SIMD_candidates=divisors(mw),
                       current=metrics(inst, inst.get_nodeattr('PE'), inst.get_nodeattr('SIMD')))
            if len(node.input) > 2:
                row['threshold_shape'] = list(model.get_initializer(node.input[2]).shape)
            for pe in divisors(mh):
                for simd in divisors(mw):
                    candidates.append(dict(node=node.name, **metrics(inst, pe, simd)))
        elif node.op_type == 'ConvolutionInputGenerator':
            channels = inst.get_nodeattr('IFMChannels')
            row.update(IFMChannels=channels, kernel=inst.get_nodeattr('ConvKernelDim'),
                       stride=inst.get_nodeattr('Stride'), current_SIMD=inst.get_nodeattr('SIMD'),
                       SIMD_candidates=divisors(channels),
                       constraint='SIMD divides IFMChannels; parallel_window/backend not selected')
        else:
            if inst.is_1d():
                raise AssertionError('This analysis expects the two 2D maxpools')
            row.update(channels=inst.get_nodeattr('NumChannels'),
                       pool=inst.get_nodeattr('PoolDim'), current_PE=inst.get_nodeattr('PE'),
                       constraint='2D StreamingMaxPool: PE is not a folding control; full-channel stream')
        specs.append(row)
    profiles, summary = {}, []
    # Caps are illustrative exploration choices, not device/resource constraints.
    for label, cap in [('serial', 1), ('small', 4), ('wider', 8)]:
        config, layer_metrics = {}, {}
        widths = {}
        for node in model.graph.node:
            inst = getCustomOp(node)
            clone = getCustomOp(deepcopy(node))
            if node.op_type == 'MVAU':
                pe = max(v for v in divisors(inst.get_nodeattr('MH')) if v <= cap)
                simd = max(v for v in divisors(inst.get_nodeattr('MW')) if v <= cap)
                config[node.name] = dict(PE=pe, SIMD=simd)
                clone.set_nodeattr('PE', pe)
                clone.set_nodeattr('SIMD', simd)
                layer_metrics[node.name] = metrics(inst, pe, simd)
                summary.append(dict(profile=label, node=node.name, **layer_metrics[node.name]))
            elif node.op_type == 'ConvolutionInputGenerator':
                simd = max(v for v in divisors(inst.get_nodeattr('IFMChannels')) if v <= cap)
                config[node.name] = dict(SIMD=simd)
                clone.set_nodeattr('SIMD', simd)
            # Do not propose PE changes for 2D pooling.
            widths[node.name] = stream_widths(clone, node.op_type)
        edges = []
        for node in model.graph.node:
            for nxt in model.graph.node:
                if node.output[0] == nxt.input[0]:
                    producer_bits, consumer_bits = widths[node.name][1], widths[nxt.name][0]
                    edges.append(dict(producer=node.name, consumer=nxt.name,
                                      producer_bits=producer_bits, consumer_bits=consumer_bits,
                                      width_conversion_needed=producer_bits != consumer_bits,
                                      integer_width_ratio=max(producer_bits, consumer_bits) %
                                                          min(producer_bits, consumer_bits) == 0))
        profiles[label] = dict(status='proposal_only_not_backend_validated', node_config=config,
                               mvau_analysis=layer_metrics, stream_edges=edges)
        (out / (label + '_proposal.json')).write_text(
            json.dumps(profiles[label], indent=2), encoding='utf-8')
    report = dict(source=str(source), source_sha256=original_hash, fpga_part=None,
                  backend=None, clock_ns=None, source_unchanged=digest(source) == original_hash,
                  checks=dict(layers=len(specs), mvau_candidate_pairs=len(candidates),
                              profile_count=len(profiles),
                              all_candidate_divisibility_checks_passed=True),
                  limitations=['Candidates satisfy abstract divisibility only.',
                               'SWG widths assume serial-window output; backend parallel_window not selected.',
                               'Ideal MAC iteration counts exclude stalls, pipeline overhead, '
                               'SWG/pooling bottlenecks and memory bandwidth; not latency or FPS.',
                               'Profiles require specialization, stream-width conversion and validation.',
                               'Original node names may change during specialization.',
                               'No model attributes or weights have been modified.'],
                  layers=specs)
    if not report['source_unchanged']:
        raise AssertionError('Source model changed')
    (out / 'layer_specs.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_csv(out / 'mvau_candidates.csv', candidates)
    save_csv(out / 'profile_comparison.csv', summary)
    flat = []
    for s in specs:
        flat.append(dict(name=s['name'], op=s['op'], input_shape=str(s['input_shape']),
                         output_shape=str(s['output_shape']), input_type=s['input_type'],
                         output_type=s['output_type'], weight_type=s.get('weight_type', ''),
                         MW=s.get('MW', ''), MH=s.get('MH', ''),
                         current_PE=s.get('current', {}).get('PE', s.get('current_PE', '')),
                         current_SIMD=s.get('current', {}).get('SIMD', s.get('current_SIMD', '')),
                         current_input_stream_bits=s['current_input_stream_bits'],
                         current_output_stream_bits=s['current_output_stream_bits']))
    save_csv(out / 'layer_specs.csv', flat)
    md = ['# W4A4 子圖逐層規格（自動產生）', '',
          '| 節點 | 類型 | 輸入形狀／型別 | 輸出形狀／型別 | 權重 | 現有 PE／SIMD |',
          '|---|---|---|---|---|---|']
    for r in flat:
        md.append(f"| {r['name']} | {r['op']} | {r['input_shape']} / {r['input_type']} | "
                  f"{r['output_shape']} / {r['output_type']} | {r['weight_type']} | "
                  f"{r['current_PE']} / {r['current_SIMD']} |")
    md += ['', '## MVAU PE／SIMD 整除候選', '',
           '候選值須取笛卡兒積；滿足整除並不保證任何後端或晶片皆可實現。', '',
           '| 節點 | MW×MH | PE 候選（MH 因數） | SIMD 候選（MW 因數） |',
           '|---|---|---|---|']
    for s in specs:
        if s['op'] == 'MVAU':
            md.append(f"| {s['name']} | {s['MW']}×{s['MH']} | {s['PE_candidates']} | {s['SIMD_candidates']} |")
    md += ['', '## 範例配置比較', '',
           'MAC 迭代數 = 向量數 × (MW/SIMD) × (MH/PE)，僅為理想化運算工作量，不是硬體量測週期。', '',
           '| 配置 | MVAU | PE | SIMD | 理想 MAC 迭代數／影像 |',
           '|---|---|---:|---:|---:|']
    for r in summary:
        md.append(f"| {r['profile']} | {r['node']} | {r['PE']} | {r['SIMD']} | "
                  f"{r['ideal_mac_iterations_per_image']} |")
    md += ['', '2D StreamingMaxPool 的 PE 不作為候選變數；本機 FINN 僅於 1D pooling 支援 PE folding。',
           '輸入生成器 SIMD 必須整除輸入通道數；不使用尚未選定後端的 parallel_window。',
           '各方案的串流寬度相容性與可能需要的 DWC 已列在 proposal JSON；沒有插入 DWC／FIFO。', '']
    (out / 'SPECIFICATION.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(report['checks'], indent=2))
    print('Source unchanged:', report['source_unchanged'])
    print('Reports:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    p.add_argument('--model', default=str(root / 'dataflow_partition_w4a4/verified/05_dataflow_core.onnx'))
    p.add_argument('--output-dir', required=True, help='New directory')
    run(p.parse_args())

