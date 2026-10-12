"""Convert QONNX to FINN-ONNX; verify against QONNX and saved QAT outputs.
Run in the existing FINN environment. No training, HLS or RTL generation.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from time import perf_counter
import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN


def prepare_unsigned_input(model):
    """Unsigned zero-point-zero quantization already clamps negatives to zero.
    An explicit ReLU makes that input quantizer recognizable by FINN.
    Leave the original source graph untouched for numerical verification.
    """
    changed = 0
    for node in list(model.graph.node):
        if node.op_type != 'Quant' or node.input[0] != model.graph.input[0].name:
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get('signed', 1) != 0:
            continue
        zp = model.get_initializer(node.input[2])
        scale = model.get_initializer(node.input[1])
        if zp is None or np.any(zp != 0) or scale is None or np.any(scale <= 0):
            raise ValueError('Input adaptation requires zero point 0 and positive constant scale')
        tensor = model.make_new_valueinfo_name()
        relu = helper.make_node('Relu', [node.input[0]], [tensor])
        node.input[0] = tensor
        model.graph.node.insert(list(model.graph.node).index(node), relu)
        changed += 1
    return model.transform(InferShapes()), changed


def compare(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f'Output shape mismatch: {actual.shape} vs {reference.shape}')
    return dict(passed=bool(np.allclose(actual, reference, rtol=1e-4, atol=1e-5)),
                max_abs_error=float(np.max(np.abs(actual - reference))))


def run(folder):
    total_start = perf_counter()
    timings = dict(conversion=None, verification=None, total=None)
    active_stage = None
    stage_start = None
    folder = Path(folder).expanduser().resolve()
    paths = [folder / n for n in ('mnist_qonnx_clean.onnx', 'input.npy', 'expected_output.npy')]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f'Required file does not exist: {p}')
    x = np.load(paths[1], allow_pickle=False)
    expected = np.load(paths[2], allow_pickle=False)
    if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (1, 28, 28):
        raise ValueError('Expected float32 input shaped [N,1,28,28]')
    if not len(x) or expected.shape != (len(x), 10):
        raise ValueError('Expected saved QAT output shaped [N,10]')
    if not np.isfinite(x).all() or not np.isfinite(expected).all():
        raise ValueError('Non-finite input or reference output')
    out = folder / ('finn_check_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    print('Results:', out, flush=True)
    report = dict(source=str(paths[0]), rtol=1e-4, atol=1e-5, versions={},
                  timing_seconds=timings)
    for pkg in ('numpy', 'onnx', 'onnxruntime', 'qonnx', 'finn'):
        try:
            report['versions'][pkg] = version(pkg)
        except PackageNotFoundError:
            report['versions'][pkg] = 'not recorded by package metadata'
    try:
        source = ModelWrapper(str(paths[0])).transform(InferShapes())
        report['before_nodes'] = dict(Counter(n.op_type for n in source.graph.node))
        print('Before:', report['before_nodes'], flush=True)
        active_stage = 'conversion'
        stage_start = perf_counter()
        # Load separately so that source remains an independent reference.
        converted = ModelWrapper(str(paths[0])).transform(InferShapes())
        converted, count = prepare_unsigned_input(converted)
        report['unsigned_input_relu_insertions'] = count
        print('Unsigned input compatibility ReLU insertions:', count, flush=True)
        converted = converted.transform(ConvertQONNXtoFINN())
        converted = converted.transform(InferShapes()).transform(InferDataTypes())
        report['after_nodes'] = dict(Counter(n.op_type for n in converted.graph.node))
        print('After:', report['after_nodes'], flush=True)
        converted.save(str(out / 'mnist_finn_candidate.onnx'))
        remaining = set(report['after_nodes']) & {'Quant', 'BipolarQuant', 'BinaryQuant', 'Trunc'}
        if remaining:
            raise RuntimeError(f'Unconverted quantization nodes: {sorted(remaining)}')
        timings['conversion'] = perf_counter() - stage_start
        active_stage = 'verification'
        stage_start = perf_counter()
        rows, q_outputs, f_outputs = [], [], []
        for i in range(len(x)):
            sample = x[i:i+1]
            q = execute_onnx(source, {source.graph.input[0].name: sample.copy()})[source.graph.output[0].name]
            f = execute_onnx(converted, {converted.graph.input[0].name: sample.copy()})[converted.graph.output[0].name]
            row = dict(sample=i+1, qonnx_vs_qat=compare(q, expected[i:i+1]),
                       finn_vs_qonnx=compare(f, q), finn_vs_qat=compare(f, expected[i:i+1]),
                       same_argmax=bool(np.argmax(f) == np.argmax(q)))
            rows.append(row)
            q_outputs.append(q)
            f_outputs.append(f)
            ok = all(row[k]['passed'] for k in ('qonnx_vs_qat', 'finn_vs_qonnx', 'finn_vs_qat'))
            print(f"Sample {i+1:02d}: {'PASS' if ok else 'FAIL'}, "
                  f"FINN/QONNX max_error={row['finn_vs_qonnx']['max_abs_error']:.8g}", flush=True)
        timings['verification'] = perf_counter() - stage_start
        active_stage = None
        np.save(out / 'qonnx_output.npy', np.concatenate(q_outputs))
        np.save(out / 'finn_output.npy', np.concatenate(f_outputs))
        report['samples'] = rows
        report['passed'] = all(r[k]['passed'] for r in rows
                               for k in ('qonnx_vs_qat', 'finn_vs_qonnx', 'finn_vs_qat'))
        if report['passed']:
            converted.save(str(out / 'mnist_finn.onnx'))
    except Exception as exc:
        report.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if active_stage is not None:
            timings[active_stage] = perf_counter() - stage_start
        timings['total'] = perf_counter() - total_start
        (out / 'verification.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Verification FAILED. Inspect verification.json; do not proceed to HLS.')
    print('Timing (s):', timings)
    print(f'FINN verification: PASS ({len(x)} samples)')
    print('Verified FINN-ONNX:', out / 'mnist_finn.onnx')
    print('This checks sampled functional outputs, not full MNIST accuracy or hardware execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', required=True,
                        help='Directory containing the cleaned QONNX and NPY reference files')
    run(parser.parse_args().export_dir)
