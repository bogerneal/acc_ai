"""Prepare FINN-ONNX for hardware mapping and verify each stage in Python.
No FPGA part or vendor tools needed. This does not generate HLS/RTL.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (GiveUniqueNodeNames, GiveReadableTensorNames,
                                         RemoveStaticGraphInputs, RemoveUnusedTensors)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw


def tidy(m):
    for t in [InferShapes(), FoldConstants(), GiveUniqueNodeNames(),
              GiveReadableTensorNames(), InferDataTypes(), RemoveStaticGraphInputs()]:
        m = m.transform(t)
    return m


def streamline(m):
    ts = [absorb.AbsorbSignBiasIntoMultiThreshold(), Streamline()]
    if m.get_nodes_by_op_type('Conv'):
        ts += [LowerConvsToMatMul(), MakeMaxPoolNHWC(),
               absorb.AbsorbTransposeIntoMultiThreshold(), MakeMaxPoolNHWC(),
               absorb.AbsorbConsecutiveTransposes()]
    ts += [ConvertBipolarMatMulToXnorPopcount(), Streamline(),
           absorb.AbsorbScalarMulAddIntoTopK(), InferDataLayouts(), RemoveUnusedTensors()]
    for t in ts:
        m = m.transform(t)
    return m.transform(InferShapes()).transform(InferDataTypes())


def hw_layers(m):
    ts = [to_hw.InferBinaryMatrixVectorActivation(),
          to_hw.InferQuantizedMatrixVectorActivation(), to_hw.InferLabelSelectLayer(),
          to_hw.InferThresholdingLayer()]
    if m.get_nodes_by_op_type('Im2Col'):
        # Keep the explicit NHWC->NCHW/Flatten boundary for Python verification.
        # RemoveCNVtoFCFlatten is deferred to backend preparation: the abstract
        # MVAU Python implementation does not implicitly flatten a 4D input.
        ts += [to_hw.InferConvInpGen(), to_hw.InferStreamingMaxPool()]
    ts += [absorb.AbsorbConsecutiveTransposes(), GiveUniqueNodeNames(), InferDataLayouts(),
           InferShapes(), InferDataTypes()]
    for t in ts:
        m = m.transform(t)
    return m


def infer(m, samples):
    return np.concatenate([execute_onnx(m, {m.graph.input[0].name: x[None].copy()})[
        m.graph.output[0].name] for x in samples])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--input', help='Optional original input.npy; otherwise generate smoke inputs')
    p.add_argument('--expected', help='Optional saved QAT expected_output.npy, requires --input')
    p.add_argument('--output-dir', required=True, help='Must be a new directory')
    a = p.parse_args()
    if a.expected and not a.input:
        p.error('--expected requires --input')
    out = Path(a.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(Path(a.model).resolve()), stages=[], rtol=1e-4, atol=1e-5,
                  validation='Python functional simulation; not C++ or RTL simulation')
    try:
        if a.input:
            x = np.load(a.input, allow_pickle=False)
            report['input_source'] = str(Path(a.input).resolve())
        else:
            rng = np.random.default_rng(42)
            x = np.concatenate([rng.random((8,1,28,28), dtype=np.float32),
                                rng.integers(0,256,(8,1,28,28)).astype(np.float32)/255,
                                np.zeros((1,1,28,28), np.float32),
                                np.ones((1,1,28,28), np.float32)])
            report['input_source'] = '18 synthetic samples: 8 random, 8 pixel-grid, zero, one'
        if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (1,28,28) or not len(x):
            raise ValueError('Input must be nonempty float32 [N,1,28,28]')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite input')
        m = ModelWrapper(a.model).transform(InferShapes())
        reference = infer(m, x)
        np.save(out/'input.npy', x)
        np.save(out/'source_output.npy', reference)
        if a.expected:
            expected = np.load(a.expected, allow_pickle=False)
            if expected.shape != reference.shape:
                raise ValueError('Expected output shape mismatch')
            np.testing.assert_allclose(reference, expected, rtol=1e-4, atol=1e-5)
            report['source_vs_qat_max_abs_error'] = float(np.max(np.abs(reference-expected)))
        for name, transform in [('01_tidy', tidy), ('02_streamlined', streamline),
                                ('03_hw_layers', hw_layers)]:
            m = transform(m)
            m.save(str(out/(name+'_candidate.onnx')))
            actual = infer(m, x)
            errors = np.max(np.abs(actual-reference), axis=1)
            passed = bool(np.allclose(actual, reference, rtol=1e-4, atol=1e-5))
            row = dict(stage=name, passed=passed, max_abs_error=float(errors.max()),
                       per_sample_max_abs_error=errors.tolist(),
                       argmax_agreement=float(np.mean(actual.argmax(1)==reference.argmax(1))),
                       nodes=dict(Counter(n.op_type for n in m.graph.node)))
            report['stages'].append(row)
            np.save(out/(name+'_output.npy'), actual)
            print(name, json.dumps(row), flush=True)
            if not passed:
                raise RuntimeError(f'{name}: numerical verification failed; inspect report.json')
            m.save(str(out/(name+'.onnx')))
        report['hardware_nodes'] = [dict(name=n.name, op=n.op_type) for n in m.graph.node
                                    if n.domain.startswith('finn.custom_op.fpgadataflow')]
        report['non_hardware_nodes'] = [dict(name=n.name, op=n.op_type) for n in m.graph.node
                                        if not n.domain.startswith('finn.custom_op.fpgadataflow')]
        report['passed'] = True
        print('Preparation PASS. Hardware abstraction graph:', out/'03_hw_layers.onnx')
        print('Remaining host/general nodes:', report['non_hardware_nodes'])
    except Exception as exc:
        report.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
