"""Absorb the W4A4 CNN-to-FC transpose/flatten boundary; no RTL generation."""
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
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def consumers(model, tensor):
    return [n for n in model.graph.node if tensor in n.input]


def run(model_path, input_path, output_dir):
    start = perf_counter()
    model_path, input_path = Path(model_path).resolve(), Path(input_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, source=str(model_path), input_source=str(input_path),
                  rtol=1e-4, atol=1e-5,
                  validation='Weight permutation proof and Python verification with an '
                             'explicit NHWC Flatten adapter; not direct hardware-graph or RTL simulation')
    try:
        report['source_sha256'] = digest(model_path)
        report['input_sha256'] = digest(input_path)
        source = ModelWrapper(str(model_path))
        x = np.load(input_path, allow_pickle=False)
        if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (1, 28, 28) or len(x) == 0:
            raise ValueError('Expected nonempty float32 [N,1,28,28] inputs')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite inputs')
        matches = []
        for flat in source.graph.node:
            if flat.op_type != 'Flatten':
                continue
            trans = source.find_producer(flat.input[0])
            next_nodes = consumers(source, flat.output[0])
            if trans is None or trans.op_type != 'Transpose' or len(next_nodes) != 1:
                continue
            attrs = {a.name: helper.get_attribute_value(a) for a in trans.attribute}
            fc = next_nodes[0]
            if attrs.get('perm') == [0, 3, 1, 2] and fc.op_type == 'MVAU':
                matches.append((trans, flat, fc))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one NHWC-to-NCHW/Flatten/MVAU boundary, found {len(matches)}')
        trans, flat, fc = matches[0]
        if len(consumers(source, trans.output[0])) != 1:
            raise ValueError('Shared transpose output is unsupported')
        if any(t in {o.name for o in source.graph.output} for t in (trans.output[0], flat.output[0])):
            raise ValueError('Boundary tensors must not be graph outputs')
        weight_name, fc_name, incoming = fc.input[1], fc.name, trans.input[0]
        if len(consumers(source, weight_name)) != 1:
            raise ValueError('Shared FC weight initializer is unsupported')
        b, h, w, c = source.get_tensor_shape(incoming)
        weights = source.get_initializer(weight_name).copy()
        if b != 1 or weights.shape[0] != h * w * c:
            raise ValueError('Unexpected FC boundary shape')
        expected_weights = weights.reshape(c, h, w, -1).transpose(1, 2, 0, 3).reshape(weights.shape)
        report['boundary'] = dict(transpose=trans.name, flatten=flat.name, fc=fc_name,
                                  input_shape=[b, h, w, c], weight_shape=list(weights.shape))
        report['before_nodes'] = dict(Counter(n.op_type for n in source.graph.node))
        transformed = deepcopy(source).transform(RemoveCNVtoFCFlatten())
        candidate = out / '04_fc_boundary_candidate.onnx'
        transformed.save(str(candidate))
        report['after_nodes'] = dict(Counter(n.op_type for n in transformed.graph.node))
        remaining = {n.name for n in transformed.graph.node}
        if trans.name in remaining or flat.name in remaining:
            raise AssertionError('FC transpose/flatten was not removed')
        new_fc = next(n for n in transformed.graph.node if n.name == fc_name)
        if new_fc.input[0] != incoming:
            raise AssertionError('FC not connected directly to pooling output')
        np.testing.assert_array_equal(transformed.get_initializer(weight_name), expected_weights)
        # Only the FC input edge and its weight initializer may change.
        source_nodes = [n for n in source.graph.node if n.name not in (trans.name, flat.name)]
        if len(source_nodes) != len(transformed.graph.node):
            raise AssertionError('Unexpected node removal')
        for old, new in zip(source_nodes, transformed.graph.node):
            expected_node = deepcopy(old)
            if old.name == fc_name:
                expected_node.input[0] = incoming
            if expected_node.SerializeToString() != new.SerializeToString():
                raise AssertionError(f'Unexpected node modification: {old.name}')
        for init in source.graph.initializer:
            if init.name != weight_name:
                np.testing.assert_array_equal(source.get_initializer(init.name),
                                              transformed.get_initializer(init.name))
        report['weight_permutation_exact'] = True
        report['other_nodes_and_initializers_unchanged'] = True

        # Abstract MVAU np.matmul cannot consume [1,H,W,C] with [H*W*C,MH].
        # This adapter is only in a separate verification graph, never in 04.
        verification = deepcopy(transformed)
        adapter_tensor = verification.make_new_valueinfo_name()
        adapter_name = 'FCBoundaryVerificationFlatten'
        if adapter_name in remaining:
            raise ValueError('Verification adapter name collision')
        verify_fc = next(n for n in verification.graph.node if n.name == fc_name)
        position = list(verification.graph.node).index(verify_fc)
        verify_fc.input[0] = adapter_tensor
        verification.graph.node.insert(position, helper.make_node(
            'Flatten', [incoming], [adapter_tensor], axis=1, name=adapter_name))
        verification.graph.value_info.append(helper.make_tensor_value_info(
            adapter_tensor, TensorProto.FLOAT, [1, h * w * c]))
        verification.set_tensor_datatype(adapter_tensor, source.get_tensor_datatype(incoming))
        verification.save(str(out / 'verification_only.onnx'))

        reference, actual, rows = [], [], []
        for i, sample in enumerate(x):
            src_ctx = execute_onnx(source, {source.graph.input[0].name: sample[None].copy()},
                                   return_full_exec_context=True)
            pool = src_ctx[incoming]
            # Exact algebraic check before activation, using actual pooling outputs.
            old_product = pool.transpose(0, 3, 1, 2).reshape(1, -1) @ weights
            new_product = pool.reshape(1, -1) @ expected_weights
            np.testing.assert_allclose(new_product, old_product, rtol=0, atol=0)
            ref = src_ctx[source.graph.output[0].name]
            got = execute_onnx(verification, {verification.graph.input[0].name: sample[None].copy()})[
                verification.graph.output[0].name]
            if ref.shape != got.shape:
                raise AssertionError('Output shape mismatch')
            passed = bool(np.allclose(got, ref, rtol=1e-4, atol=1e-5))
            rows.append(dict(sample=i, passed=passed,
                             max_abs_error=float(np.max(np.abs(got-ref))),
                             same_argmax=bool(np.argmax(got) == np.argmax(ref))))
            reference.append(ref)
            actual.append(got)
        np.save(out / 'input.npy', x)
        np.save(out / 'source_output.npy', np.concatenate(reference))
        np.save(out / 'verification_output.npy', np.concatenate(actual))
        report.update(samples=len(x), comparisons=rows,
                      max_abs_error=max(r['max_abs_error'] for r in rows),
                      argmax_agreement=float(np.mean([r['same_argmax'] for r in rows])),
                      fc_products_exact=True)
        if not all(r['passed'] for r in rows):
            raise AssertionError('Output verification failed')
        if digest(model_path) != report['source_sha256']:
            raise AssertionError('Source model changed')
        transformed.save(str(out / '04_fc_boundary.onnx'))
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['total_seconds'] = perf_counter() - start
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('passed', 'samples', 'max_abs_error', 'argmax_agreement')},
                     indent=2))
    print('Hardware-preparation graph:', out / '04_fc_boundary.onnx')


def main():
    root = Path(__file__).resolve().parent.parent / 'mnist_hw_prepare' / 'local_check_2'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=str(root / '03_hw_layers.onnx'))
    p.add_argument('--input', default=str(root / 'input.npy'))
    p.add_argument('--output-dir', required=True, help='New directory; existing outputs are not overwritten')
    a = p.parse_args()
    run(a.model, a.input, a.output_dir)


if __name__ == '__main__':
    main()

