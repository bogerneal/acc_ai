"""Partition the W4A4 04 graph into host operations and one FPGA core."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.util.fpgadataflow import is_fpgadataflow_node


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def io_info(model, values):
    return [dict(name=v.name, shape=model.get_tensor_shape(v.name),
                 datatype=str(model.get_tensor_datatype(v.name))) for v in values]


def run(a):
    started = perf_counter()
    paths = {k: Path(getattr(a, k)).resolve() for k in ('model', 'reference', 'input')}
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, sources={k: str(v) for k, v in paths.items()},
                  rtol=1e-4, atol=1e-5,
                  validation='Partition structure/weights checked exactly; parent+child Python '
                             'execution uses an explicit NHWC Flatten adapter in a separate child. '
                             'Compared against original 03 graph, not RTL simulation.')
    try:
        report['sha256'] = {k: sha(v) for k, v in paths.items()}
        source = ModelWrapper(str(paths['model']))
        reference = ModelWrapper(str(paths['reference']))
        x = np.load(paths['input'], allow_pickle=False)
        if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (1, 28, 28) or not len(x):
            raise ValueError('Expected nonempty float32 [N,1,28,28]')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite input')
        expected_counts = {'ConvolutionInputGenerator': 2, 'MVAU': 4, 'StreamingMaxPool': 2}
        hw_nodes = [n for n in source.graph.node if is_fpgadataflow_node(n)]
        if dict(Counter(n.op_type for n in hw_nodes)) != expected_counts:
            raise ValueError('Expected W4A4 core with two convolution generators, four MVAUs and two pools')
        if source.get_nodes_by_op_type('Flatten'):
            raise ValueError('Use 04_fc_boundary.onnx after FC boundary conversion')
        if any(getCustomOp(n).get_nodeattr('weightDataType') != 'INT4'
               for n in hw_nodes if n.op_type == 'MVAU'):
            raise ValueError('Expected INT4 MVAU weights')

        parent = deepcopy(source).transform(CreateDataflowPartition(
            partition_model_dir=str(out / 'partitions')))
        parts = parent.get_nodes_by_op_type('StreamingDataflowPartition')
        if len(parts) != 1:
            raise AssertionError(f'Expected one partition, found {len(parts)}')
        part = parts[0]
        child = ModelWrapper(getCustomOp(part).get_nodeattr('model'))
        if len(child.graph.node) != len(hw_nodes) or any(
                not is_fpgadataflow_node(n) for n in child.graph.node):
            raise AssertionError('Child must contain only the eight hardware nodes')
        for old, new in zip(hw_nodes, child.graph.node):
            if old.SerializeToString() != new.SerializeToString():
                raise AssertionError(f'Partition modified hardware node {old.name}')
        for init in child.graph.initializer:
            np.testing.assert_array_equal(child.get_initializer(init.name),
                                          source.get_initializer(init.name))
        original_host = [n for n in source.graph.node if not is_fpgadataflow_node(n)]
        actual_host = [n for n in parent.graph.node if n.op_type != 'StreamingDataflowPartition']
        if [n.SerializeToString() for n in original_host] != [n.SerializeToString() for n in actual_host]:
            raise AssertionError('Host operations changed')
        for init in parent.graph.initializer:
            np.testing.assert_array_equal(parent.get_initializer(init.name),
                                          source.get_initializer(init.name))
        if len(child.graph.input) != 1 or len(child.graph.output) != 1:
            raise AssertionError('Expected single data input and output')
        if child.get_tensor_shape(child.graph.input[0].name) != [1, 28, 28, 1]:
            raise AssertionError('Unexpected core input shape')
        if str(child.get_tensor_datatype(child.graph.input[0].name)) != 'UINT8':
            raise AssertionError('Expected UINT8 core input')
        if child.get_tensor_shape(child.graph.output[0].name) != [1, 10]:
            raise AssertionError('Unexpected core output shape')
        if str(child.get_tensor_datatype(child.graph.output[0].name)) != 'INT32':
            raise AssertionError('Expected INT32 core output')
        report.update(partition_count=1, exact_nodes_and_weights=True,
                      parent_nodes=dict(Counter(n.op_type for n in parent.graph.node)),
                      core_nodes=dict(Counter(n.op_type for n in child.graph.node)),
                      core_inputs=io_info(child, child.graph.input),
                      core_outputs=io_info(child, child.graph.output))
        child.save(str(out / '05_dataflow_core_candidate.onnx'))
        getCustomOp(part).set_nodeattr('model', str(out / '05_dataflow_core_candidate.onnx'))
        parent.save(str(out / '05_dataflow_parent_candidate.onnx'))

        # Adapter for the abstract Python MVAU only; never added to the hardware core.
        verify_child = deepcopy(child)
        candidates = [n for n in verify_child.graph.node if n.op_type == 'MVAU'
                      and getCustomOp(n).get_nodeattr('MW') == 512
                      and getCustomOp(n).get_nodeattr('MH') == 128]
        if len(candidates) != 1:
            raise AssertionError('Expected one FC 512->128 adapter point')
        fc = candidates[0]
        incoming = fc.input[0]
        if verify_child.get_tensor_shape(incoming) != [1, 4, 4, 32]:
            raise AssertionError('Unexpected FC boundary')
        tensor = verify_child.make_new_valueinfo_name()
        pos = list(verify_child.graph.node).index(fc)
        verify_child.graph.node.insert(pos, helper.make_node(
            'Flatten', [incoming], [tensor], axis=1, name='VerificationOnlyFlatten'))
        fc.input[0] = tensor
        verify_child.graph.value_info.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, [1, 512]))
        verify_child.set_tensor_datatype(tensor, verify_child.get_tensor_datatype(incoming))
        verify_child.save(str(out / 'verification_core.onnx'))
        verify_parent = deepcopy(parent)
        vp = verify_parent.get_nodes_by_op_type('StreamingDataflowPartition')[0]
        getCustomOp(vp).set_nodeattr('model', str(out / 'verification_core.onnx'))
        verify_parent.save(str(out / 'verification_parent.onnx'))
        refs, actuals, core_inputs, core_outputs, rows = [], [], [], [], []
        for i, sample in enumerate(x):
            ref = execute_onnx(reference, {reference.graph.input[0].name: sample[None].copy()})[
                reference.graph.output[0].name]
            ctx = execute_onnx(verify_parent, {verify_parent.graph.input[0].name: sample[None].copy()},
                               return_full_exec_context=True)
            got = ctx[verify_parent.graph.output[0].name]
            if got.shape != ref.shape:
                raise AssertionError('Output shape mismatch')
            passed = bool(np.allclose(got, ref, rtol=1e-4, atol=1e-5))
            rows.append(dict(sample=i, passed=passed, max_abs_error=float(np.max(np.abs(got-ref))),
                             same_argmax=bool(got.argmax() == ref.argmax())))
            refs.append(ref)
            actuals.append(got)
            core_inputs.append(ctx[vp.input[0]])
            core_outputs.append(ctx[vp.output[0]])
        for name, array in [('input', x), ('reference_output', np.concatenate(refs)),
                            ('partition_output', np.concatenate(actuals)),
                            ('core_input', np.concatenate(core_inputs)),
                            ('core_expected_output', np.concatenate(core_outputs))]:
            np.save(out / (name + '.npy'), array)
        report.update(samples=len(x), comparisons=rows,
                      max_abs_error=max(r['max_abs_error'] for r in rows),
                      argmax_agreement=float(np.mean([r['same_argmax'] for r in rows])))
        if not all(r['passed'] for r in rows):
            raise AssertionError('Partition verification failed')
        if {k: sha(v) for k, v in paths.items()} != report['sha256']:
            raise AssertionError('Source files changed')
        child.save(str(out / '05_dataflow_core.onnx'))
        getCustomOp(part).set_nodeattr('model', str(out / '05_dataflow_core.onnx'))
        parent.save(str(out / '05_dataflow_parent.onnx'))
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['total_seconds'] = perf_counter() - started
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('passed', 'partition_count', 'samples',
                                            'max_abs_error', 'argmax_agreement')}, indent=2))
    print('Saved:', out)


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=str(root / 'fc_boundary_w4a4/verified_local/04_fc_boundary.onnx'))
    p.add_argument('--reference', default=str(root / 'mnist_hw_prepare/local_check_2/03_hw_layers.onnx'))
    p.add_argument('--input', default=str(root / 'fc_boundary_w4a4/verified_local/input.npy'))
    p.add_argument('--output-dir', required=True, help='New output directory')
    run(p.parse_args())


if __name__ == '__main__':
    main()

