"""Pinned FINN core preparation and HLS source generation; no claimed RTL validation.

Run under the existing WSL FINN Python environment. Each run gets a new directory.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'dataflow_partition_w4a4/verified'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise RuntimeError(f'{args}: {p.stderr}')
    return p.stdout.strip()


def specialize(model, part):
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    names = [n.name for n in model.graph.node]
    model = model.transform(SpecializeLayers(part))
    assert len(names) == len(model.graph.node)
    for n, name in zip(model.graph.node, names):
        n.name = name
    return model


def inventory(model):
    from qonnx.custom_op.registry import getCustomOp
    rows = []
    for n in model.graph.node:
        op = getCustomOp(n)
        attrs = op.get_nodeattr_types()
        row = dict(name=n.name, op_type=n.op_type,
                   input_shape=model.get_tensor_shape(n.input[0]),
                   output_shape=model.get_tensor_shape(n.output[0]),
                   input_datatype=str(op.get_input_datatype()),
                   output_datatype=str(op.get_output_datatype()))
        for key in ('PE', 'SIMD', 'MW', 'MH', 'mem_mode', 'ram_style',
                    'resType', 'weightDataType', 'accDataType', 'noActivation',
                    'inFIFODepths', 'outFIFODepths'):
            if key in attrs:
                row[key] = op.get_nodeattr(key)
        if n.op_type.endswith('_hls'):
            row.update(input_stream_bits=op.get_instream_width(),
                       output_stream_bits=op.get_outstream_width(),
                       estimated_cycles=op.get_exp_cycles())
        rows.append(row)
    return rows


def python_view(model):
    """Reference semantics only: strip backend and adapt the known FC shape.

    This is explicitly NOT execution of generated HLS C++ or RTL.
    """
    from onnx import helper, TensorProto
    from qonnx.custom_op.registry import getCustomOp
    model = deepcopy(model)
    for n in list(model.graph.node):
        if n.op_type.endswith(('_hls', '_rtl')):
            n.op_type = n.op_type.rsplit('_', 1)[0]
            n.domain = 'finn.custom_op.fpgadataflow'
        if n.op_type == 'MVAU' and getCustomOp(n).get_nodeattr('MW') == 512:
            incoming = n.input[0]
            assert model.get_tensor_shape(incoming) == [1, 4, 4, 32]
            tensor = model.make_new_valueinfo_name()
            pos = list(model.graph.node).index(n)
            model.graph.node.insert(pos, helper.make_node(
                'Flatten', [incoming], [tensor], axis=1, name='VerificationOnlyFlatten'))
            model.graph.value_info.append(helper.make_tensor_value_info(
                tensor, TensorProto.FLOAT, [1, 512]))
            model.set_tensor_datatype(tensor, model.get_tensor_datatype(incoming))
            n.input[0] = tensor
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    p.add_argument('--repeat', type=int, default=1, help='Independent preparation/codegen runs')
    args = p.parse_args()
    if args.repeat < 1 or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        p.error('Use a simple run ID and positive repeat count')
    for repetition in range(args.repeat):
        run(args.run_id if args.repeat == 1 else f'{args.run_id}_r{repetition+1}')
    if args.repeat > 1:
        results = [json.loads((ROOT / 'reports' / f'{args.run_id}_r{i+1}.json').read_text())
                   for i in range(args.repeat)]
        summary = {'scope': 'preparation, Python reference verification and source generation ONLY',
                   'repetitions': args.repeat, 'full_hardware_builds': 0, 'timings': {}}
        for name in {e['name'] for r in results for e in r['events']}:
            values = [e['seconds'] for r in results for e in r['events'] if e['name'] == name]
            summary['timings'][name] = dict(median_seconds=statistics.median(values),
                                           min_seconds=min(values), max_seconds=max(values))
        write_json(ROOT / 'reports' / f'{args.run_id}_summary.json', summary)


def run(run_id):
    import numpy as np
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.custom_op.registry import getCustomOp
    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

    cfg_path = ROOT / 'configs/aup_zu3_baseline.json'
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / 'build/aup_zu3' / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = dict(run_id=run_id, status='running', events=[],
                  hls_synthesis='not_run', cppsim='not_run', rtlsim='not_run',
                  stitch='not_run', fifo_validation='not_run', part_support='unconfirmed')
    @contextmanager
    def event(name):
        start = perf_counter()
        item = dict(name=name, status='running')
        report['events'].append(item)
        try:
            yield
            item['status'] = 'passed'
        except Exception as exc:
            item.update(status='failed', error=str(exc))
            raise
        finally:
            item['seconds'] = perf_counter() - start

    try:
        with event('T_prepare'):
            finn = Path(cfg['finn_root'])
            import finn as finn_package
            assert str(finn / 'src/finn') in list(finn_package.__path__)
            assert command(['git', '-C', str(finn), 'rev-parse', 'HEAD']) == cfg['finn_commit']
            assert not command(['git', '-C', str(finn), 'status', '--porcelain', '--untracked-files=no']), 'FINN tracked files modified'
            os.environ.update(FINN_ROOT=str(finn), FINN_BUILD_DIR=str(out / 'generated'),
                              NUM_DEFAULT_WORKERS=str(cfg['num_workers']))
            (out / 'generated').mkdir()
            shutil.copy2(cfg_path, out / cfg_path.name)
            runtime = ROOT / 'deps/finn_runtime'
            dependency_hashes = {}
            for folder in ('custom_hls', 'finn-rtllib'):
                for src in (finn / folder).rglob('*'):
                    if not src.is_file():
                        continue
                    relative = src.relative_to(finn)
                    dst = runtime / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        assert sha(dst) == sha(src), f'Dependency snapshot mismatch: {dst}'
                    else:
                        shutil.copy2(src, dst)
                    dependency_hashes[str(relative)] = sha(src)
            shutil.copy2(finn / 'LICENSE.txt', runtime / 'LICENSE.txt')
            hlslib = ROOT / 'deps/finn-hlslib'
            assert (hlslib / 'mvau.hpp').is_file(), 'Fetch the pinned finn-hlslib dependency first'
            assert command(['git', '-C', str(hlslib), 'rev-parse', 'HEAD']) == '16e5847a5e3ef76cffe84c8fad2f010d593457d3'
            assert not command(['git', '-C', str(hlslib), 'status', '--porcelain'])
            link = runtime / 'deps/finn-hlslib'
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(hlslib, target_is_directory=True)
            assert link.resolve() == hlslib.resolve()
            report['dependency_snapshot'] = dict(runtime=str(runtime), hashes=dependency_hashes)
            versions = {}
            for package in ('finn', 'qonnx', 'brevitas', 'onnx', 'numpy', 'pyverilator'):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            tools = {n: shutil.which(n) for n in ('vivado', 'vitis_hls', 'g++', 'verilator', 'docker')}
            report['environment'] = dict(python=sys.version, executable=sys.executable,
                platform=platform.platform(), cpu_count=os.cpu_count(),
                cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                meminfo=Path('/proc/meminfo').read_text(), versions=versions,
                tools=tools, finn_commit=cfg['finn_commit'], container_image=None,
                execution='WSL venv, not FINN Docker',
                toolchain_compatibility='unconfirmed until vendor tools are available')
            report['environment']['expected_vendor_version'] = cfg['expected_vendor_version']
            cgroup = {}
            for f in ('cpu.max', 'memory.max'):
                path = Path('/sys/fs/cgroup') / f
                cgroup[f] = path.read_text().strip() if path.exists() else None
            report['environment']['cgroup_limits'] = cgroup
            files = {
                'models/05_dataflow_core.onnx': SOURCE / '05_dataflow_core.onnx',
                'models/05_dataflow_parent_original.onnx': SOURCE / '05_dataflow_parent.onnx',
                'verification/core_input.npy': SOURCE / 'core_input.npy',
                'verification/core_expected_output.npy': SOURCE / 'core_expected_output.npy',
                'verification/input.npy': SOURCE / 'input.npy',
                'verification/reference_output.npy': SOURCE / 'reference_output.npy',
                'verification/source_report.json': SOURCE / 'report.json',
                'provenance/partition_w4a4.py': SOURCE.parent / 'partition_w4a4.py',
                'provenance/prepare_fc_boundary.py': ROOT.parent / 'fc_boundary_w4a4/prepare_fc_boundary.py',
                'provenance/prepare_mnist_hw.py': ROOT.parent / 'mnist_hw_prepare/prepare_mnist_hw.py',
            }
            hashes = {}
            for relative, src in files.items():
                dst = ROOT / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    assert sha(dst) == sha(src), f'Snapshot mismatch: {dst}'
                else:
                    shutil.copy2(src, dst)
                hashes[relative] = dict(source=str(src), sha256=sha(src))
            report['sources'] = hashes
            report['config'] = cfg
            report['config_sha256'] = sha(cfg_path)
            report['script_sha256'] = sha(__file__)
            model = ModelWrapper(str(ROOT / 'models/05_dataflow_core.onnx'))
            assert len(model.graph.node) == 8
            assert not any(n.op_type == 'StreamingDataflowPartition' for n in model.graph.node)
            for init in model.graph.initializer:
                assert model.get_initializer(init.name) is not None
                assert init.data_location == 0, 'External data needs an explicit snapshot'
            parent = ModelWrapper(str(ROOT / 'models/05_dataflow_parent_original.onnx'))
            partition = parent.get_nodes_by_op_type('StreamingDataflowPartition')
            assert len(partition) == 1
            getCustomOp(partition[0]).set_nodeattr('model', str(ROOT / 'models/05_dataflow_core.onnx'))
            parent.save(str(ROOT / 'models/05_dataflow_parent.onnx'))
            before = inventory(model)
            write_json(out / 'layers_before.json', before)
            for n in model.graph.node:
                op = getCustomOp(n)
                if n.op_type == 'MVAU':
                    assert op.get_nodeattr('PE') == op.get_nodeattr('SIMD') == 1
                    assert op.get_nodeattr('weightDataType') == 'INT4'
                op.set_nodeattr('preferred_impl_style', cfg['backend_policy'])
            model = specialize(model, cfg['part'])
            assert all(n.op_type.endswith('_hls') for n in model.graph.node)
            model.save(str(out / '06_specialized.onnx'))
            write_json(out / 'layers_specialized.json', inventory(model))
            write_json(out / 'folding_baseline.json', inventory(model))
            folding = ROOT / 'configs/folding_baseline.json'
            if folding.exists():
                assert json.loads(folding.read_text()) == inventory(model)
            else:
                write_json(folding, inventory(model))
        with event('T_stream_interfaces'):
            model = model.transform(InsertDWC())
            count = 0
            for n in model.graph.node:
                if n.op_type == 'StreamingDataWidthConverter':
                    n.name = f'StreamingDataWidthConverter_{count}'
                    count += 1
                    op = getCustomOp(n)
                    op.set_nodeattr('preferred_impl_style', 'hls')
                    model.set_tensor_datatype(n.output[0], op.get_output_datatype())
            model = specialize(model, cfg['part'])
            for n in model.graph.node:
                op = getCustomOp(n)
                for consumer in model.find_consumers(n.output[0]):
                    cop = getCustomOp(consumer)
                    assert op.get_outstream_width() == cop.get_instream_width()
                    assert np.prod(op.get_folded_output_shape()) == np.prod(cop.get_folded_input_shape())
            model.save(str(out / '07_widths.onnx'))
            report['dwc_count'] = count
            write_json(out / 'layers_with_dwc.json', inventory(model))
        with event('T_verify_python'):
            original = ModelWrapper(str(ROOT / 'models/05_dataflow_core.onnx'))
            for init in original.graph.initializer:
                np.testing.assert_array_equal(original.get_initializer(init.name), model.get_initializer(init.name))
            x = np.load(ROOT / 'verification/core_input.npy', allow_pickle=False)
            gold = np.load(ROOT / 'verification/core_expected_output.npy', allow_pickle=False)
            assert x.dtype == np.float32 and x.shape[1:] == (28, 28, 1)
            assert np.isfinite(x).all() and (x == np.floor(x)).all() and x.min() >= 0 and x.max() <= 255
            assert gold.shape == (len(x), 10)
            assert np.isfinite(gold).all() and (gold == np.floor(gold)).all()
            rows = []
            for label, candidate in (('before', original), ('after_interfaces', model)):
                reference = python_view(candidate)
                reference.save(str(out / f'verification_only_{label}.onnx'))
                actual = np.concatenate([execute_onnx(reference, {
                    reference.graph.input[0].name: sample[None].copy()})[
                    reference.graph.output[0].name] for sample in x])
                np.testing.assert_array_equal(actual, gold)
                np.save(out / f'{label}_output.npy', actual)
                rows.append(dict(stage=label, samples=len(x), exact_equal=True, max_abs_error=0))
            report['verification'] = dict(comparisons=rows, generated_code_executed=False,
                scope='Abstract Python semantics after stripping backend; FC reshape adapter only in verification copies',
                sample_source='18 synthetic samples inherited from partition verification, not MNIST accuracy',
                input_name=model.graph.input[0].name, input_shape=list(x.shape),
                input_layout='NHWC', input_datatype='UINT8', input_range=[float(x.min()), float(x.max())],
                output_name=model.graph.output[0].name, output_datatype='INT32', output_shape=list(gold.shape))
        with event('T_codegen'):
            model = model.transform(PrepareIP(cfg['part'], cfg['clock_ns']))
            model.save(str(out / '08_codegen.onnx'))
            generated = []
            for n in model.graph.node:
                op = getCustomOp(n)
                directory = Path(op.get_nodeattr('code_gen_dir_ipgen'))
                cpp = directory / f'top_{n.name}.cpp'
                tcl = directory / f'hls_syn_{n.name}.tcl'
                assert cpp.is_file() and tcl.is_file()
                assert cfg['part'] in tcl.read_text()
                generated.append(dict(node=n.name, backend='hls', directory=str(directory),
                    files={f.name: sha(f) for f in directory.iterdir() if f.is_file()}))
            report['generated'] = generated
        for relative, item in hashes.items():
            assert sha(item['source']) == item['sha256'], 'Source was modified'
        hlslib = ROOT / 'deps/finn-hlslib'
        report['hlslib'] = dict(path=str(hlslib), available=hlslib.is_dir(),
            required_commit='16e5847a5e3ef76cffe84c8fad2f010d593457d3')
        if hlslib.is_dir():
            report['hlslib']['actual_commit'] = command(['git', '-C', str(hlslib), 'rev-parse', 'HEAD'])
            assert report['hlslib']['actual_commit'] == report['hlslib']['required_commit']
        report['status'] = 'sources_generated_hardware_validation_pending'
        report['blockers'] = [f'{n} not on WSL PATH' for n in ('vitis_hls', 'vivado', 'g++', 'verilator') if not tools[n]]
        report['remaining'] = ['vendor part/version/license smoke check', 'HLS C++ simulation',
            'HLS synthesis and per-layer RTL/IP', 'FIFO insertion/sizing, codegen and IP rebuild',
            'stitched IP', 'multi-input RTL verification with timeout/backpressure',
            'parent postprocessing equivalence', 'three full hardware builds and timing aggregation']
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        write_json(out / 'report.json', report)
        write_json(ROOT / 'reports' / f'{run_id}.json', report)
        print(json.dumps(dict(run=str(out), status=report['status']), ensure_ascii=False))


if __name__ == '__main__':
    main()
