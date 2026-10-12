"""Smoke-test cnn.py -> QONNX. Run beside the user's original cnn.py.

Requires torch, brevitas, onnx, onnxoptimizer, qonnx.
This uses untrained weights; no accuracy or hardware compatibility is claimed.
Optional --finn also converts to FINN-ONNX when FINN is installed.
"""
import argparse
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.cleanup import cleanup

from cnn import QuantizedMNIST


def verify(graph, inputs, expected, label):
    input_name = graph.graph.input[0].name
    output_name = graph.graph.output[0].name
    errors = []
    for idx, (x, reference) in enumerate(zip(inputs, expected)):
        result = execute_onnx(graph, {input_name: x})[output_name]
        if result.shape != reference.shape:
            raise AssertionError(f'{label} sample {idx}: shape mismatch')
        np.testing.assert_allclose(
            result, reference, rtol=1e-4, atol=1e-4,
            err_msg=f'{label} sample {idx}: output mismatch',
        )
        errors.append(float(np.max(np.abs(result - reference))))
    print(f'{label}: PASS ({len(inputs)} samples, max error={max(errors):.6g})')
    return max(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/smoke_test')
    parser.add_argument('--finn', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Output directory is not empty. Use a new --output-dir.')

    torch.manual_seed(42)
    model = QuantizedMNIST(weight_bit_width=4, act_bit_width=4).cpu()
    # Synthetic initialization only, not MNIST calibration or training.
    with torch.no_grad():
        model.train()
        model(torch.rand(4, 1, 28, 28))
        model.eval()
        samples = [torch.zeros(1, 1, 28, 28), torch.ones(1, 1, 28, 28)]
        samples += [torch.rand(1, 1, 28, 28) for _ in range(2)]
        expected = [model(x).detach().cpu().numpy() for x in samples]
    for y in expected:
        assert y.shape == (1, 10) and np.isfinite(y).all()
    print('PyTorch forward: PASS, output shape (1, 10)')
    torch.save(model.state_dict(), out / 'untrained_smoke_state.pth')
    inputs = [x.numpy() for x in samples]
    np.save(out / 'input.npy', np.concatenate(inputs))
    np.save(out / 'expected_output.npy', np.concatenate(expected))

    raw = out / 'cnn_qonnx.onnx'
    clean = out / 'cnn_qonnx_clean.onnx'
    export_qonnx(model, samples[2], str(raw))
    cleanup(str(raw), out_file=str(clean))
    graph = ModelWrapper(str(clean))
    report = {
        'trained': False,
        'versions': {p: version(p) for p in ['torch', 'brevitas', 'onnx', 'qonnx']},
        'qonnx_nodes': dict(Counter(n.op_type for n in graph.graph.node)),
        'qonnx_max_abs_error': verify(graph, inputs, expected, 'QONNX'),
    }
    if args.finn:
        from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
        graph = graph.transform(ConvertQONNXtoFINN())
        graph.save(str(out / 'cnn_finn.onnx'))
        report['finn_max_abs_error'] = verify(graph, inputs, expected, 'FINN-ONNX')
        report['finn_nodes'] = dict(Counter(n.op_type for n in graph.graph.node))
    (out / 'verification.json').write_text(json.dumps(report, indent=2))
    print(f'Outputs: {out.resolve()}')
    print('Untrained smoke test only. Hardware mapping and HLS C++ generation are separate steps.')


if __name__ == '__main__':
    main()
