"""Run beside mnist_cnn.py. QAT fine-tuning and verified QONNX export.
Weights/hidden activations: --wbits/--abits (default 4); input: unsigned 8 bits.
Bias and final logits remain floating point in this training representation.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
import brevitas
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat
from mnist_cnn import DigitCNN, evaluate, seed_all, test_checkpoint


class QATCNN(nn.Module):
    def __init__(self, n1=16, n2=32, n3=128, dropout=0.5, wbits=4, abits=4):
        super().__init__()
        self.config = dict(n1=n1, n2=n2, n3=n3, dropout=dropout, wbits=wbits, abits=abits)
        self.input_quant = qnn.QuantIdentity(
            act_quant=Uint8ActPerTensorFloat, bit_width=8,
            scaling_impl_type='CONST', min_val=0.0, max_val=1.0,
            return_quant_tensor=False)
        kw = dict(weight_quant=Int8WeightPerTensorFloat, weight_bit_width=wbits,
                  bias=True, bias_quant=None, return_quant_tensor=False)
        self.conv1 = qnn.QuantConv2d(1, n1, 5, padding=0, **kw)
        self.relu1 = qnn.QuantReLU(bit_width=abits, return_quant_tensor=False)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = qnn.QuantConv2d(n1, n2, 5, padding=0, **kw)
        self.relu2 = qnn.QuantReLU(bit_width=abits, return_quant_tensor=False)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.fc3 = qnn.QuantLinear(n2 * 4 * 4, n3, **kw)
        self.relu3 = qnn.QuantReLU(bit_width=abits, return_quant_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.fc4 = qnn.QuantLinear(n3, 10, **kw)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout(self.relu3(self.fc3(self.flatten(x))))
        return self.fc4(x)

    @torch.no_grad()
    def copy_fp32(self, checkpoint):
        # Strictly validate the original architecture; do not silently ignore keys.
        fp = DigitCNN(**checkpoint['config'])
        fp.load_state_dict(checkpoint['model_state_dict'], strict=True)
        for name in ('conv1', 'conv2', 'fc3', 'fc4'):
            src, dst = getattr(fp, name), getattr(self, name)
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
            assert torch.equal(dst.weight, src.weight)
            assert torch.equal(dst.bias, src.bias)
        return fp


def loaders(args, seed, split_seed=None):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader, random_split
    ds = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    if split_seed is None:
        split_seed = seed
    tr, va = random_split(ds, [55000, 5000], generator=torch.Generator().manual_seed(split_seed))
    te = MNIST(args.data_dir, train=False, download=True, transform=ToTensor())
    return (DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(va, batch_size=256, num_workers=0),
            DataLoader(te, batch_size=256, num_workers=0))


def train(args):
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    split_seed = int(ck.get('seed', 42))
    seed = split_seed  # Always inherit the FP32 checkpoint seed.
    seed_all(seed)
    if 'wbits' in ck['config'] or 'abits' in ck['config']:
        raise ValueError('train requires an FP32 checkpoint, not a QAT checkpoint')
    model = QATCNN(**ck['config'], wbits=args.wbits, abits=args.abits)
    group = f'W{args.wbits}A{args.abits}'
    fp = model.copy_fp32(ck)
    print('FP32 weights and biases copied: PASS', flush=True)
    tr, va, te = loaders(args, seed, split_seed)
    _, fp_val = evaluate(fp, va, 'cpu')
    _, fp_test = evaluate(fp, te, 'cpu')
    del fp
    # Brevitas initializes/learns activation scales during training.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = Path(args.output_dir) / (f'{group}_seed{seed}_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    print('QAT run:', run.resolve(), flush=True)
    print(f'FP32 validation accuracy: {fp_val:.2%}', flush=True)
    print(f'Quantization: {group}; input=8 bits; seed={seed}; split_seed={split_seed}', flush=True)
    best = -1.0
    training_start = perf_counter()
    with (run / 'history.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_accuracy'])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = correct = 0
            loss_sum = 0.0
            for x, y in tr:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite QAT loss')
                loss.backward()
                optimizer.step()
                total += y.numel()
                correct += (logits.argmax(1) == y).sum().item()
                loss_sum += loss.item() * y.numel()
            _, acc = evaluate(model, va, 'cpu')
            writer.writerow([epoch, loss_sum/total, correct/total, acc])
            f.flush()
            print(f'Epoch {epoch}/{args.epochs}: loss={loss_sum/total:.4f} '
                  f'train_acc={correct/total:.2%} val_acc={acc:.2%}', flush=True)
            if acc > best:
                best = acc
                torch.save(dict(config=model.config, model_state_dict=model.state_dict(),
                                seed=seed, split_seed=split_seed, group=group,
                                epoch=epoch, val_accuracy=acc,
                                source_fp32=str(Path(args.checkpoint).resolve()),
                                brevitas=str(brevitas.__version__), torch=str(torch.__version__)),
                           run / 'best_qat.pth')
    training_seconds = perf_counter() - training_start
    saved = torch.load(run / 'best_qat.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    test_start = perf_counter()
    _, test = evaluate(model, te, 'cpu')
    test_seconds = perf_counter() - test_start
    (run / 'metrics.json').write_text(json.dumps(dict(
        fp32_val_accuracy=fp_val, qat_val_accuracy=best, qat_test_accuracy=test,
        fp32_test_accuracy=fp_test, accuracy_drop_pp=100.0 * (fp_test - test),
        group=group, input_bits=8, split_seed=split_seed,
        batch_size=args.batch_size, training_seconds=training_seconds,
        test_seconds=test_seconds,
        best_epoch=saved['epoch'], config=model.config, lr=args.lr,
        epochs=args.epochs, seed=seed, source_fp32=saved['source_fp32'],
        torch=str(torch.__version__), brevitas=str(brevitas.__version__)), indent=2))
    print(f'QAT test accuracy: {test:.2%}')
    print('Saved:', run / 'best_qat.pth')


def export(args):
    export_start = perf_counter()
    from brevitas.export import export_qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.util.cleanup import cleanup_model
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = QATCNN(**ck['config'])
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    out = Path(args.checkpoint).parent / 'export_qonnx'
    out.mkdir(exist_ok=False)
    _, va, _ = loaders(args, int(ck['seed']), int(ck.get('split_seed', ck['seed'])))
    real, _ = next(iter(va))
    torch.manual_seed(123)
    samples = torch.cat([real[:16], torch.rand(4, 1, 28, 28),
                         torch.zeros(1, 1, 28, 28), torch.ones(1, 1, 28, 28)])
    raw = out / 'mnist_qonnx.onnx'
    onnx_start = perf_counter()
    export_qonnx(model, args=torch.zeros(1, 1, 28, 28), export_path=str(raw))
    onnx_seconds = perf_counter() - onnx_start
    cleanup_start = perf_counter()
    graph = cleanup_model(ModelWrapper(str(raw)))
    graph.save(str(out / 'mnist_qonnx_clean.onnx'))
    cleanup_seconds = perf_counter() - cleanup_start
    verification_start = perf_counter()
    errors, expected, actual = [], [], []
    with torch.no_grad():
        for x in samples.split(1):
            ref = model(x).numpy()
            got = execute_onnx(graph, {graph.graph.input[0].name: x.numpy()})[graph.graph.output[0].name]
            expected.append(ref)
            actual.append(got)
            errors.append(float(np.max(np.abs(ref-got))))
    expected, actual = np.concatenate(expected), np.concatenate(actual)
    passed = bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5))
    verification_seconds = perf_counter() - verification_start
    np.save(out / 'input.npy', samples.numpy())
    np.save(out / 'expected_output.npy', expected)
    np.save(out / 'qonnx_output.npy', actual)
    total_seconds = perf_counter() - export_start
    (out / 'verification.json').write_text(json.dumps(dict(
        passed=passed, stage='Brevitas QAT vs cleaned QONNX; FINN not yet tested',
        config=model.config, input_bits=8,
        group=f"W{model.config['wbits']}A{model.config['abits']}",
        export_and_verification_seconds=total_seconds,
        timing_seconds=dict(onnx_export=onnx_seconds, cleanup=cleanup_seconds,
                            verification=verification_seconds, total=total_seconds),
        checkpoint=str(Path(args.checkpoint).resolve()),
        threads=torch.get_num_threads(), device='cpu',
        samples='16 validation digits + 4 random + zero + one',
        max_abs_error=max(errors), per_sample_max_abs_error=errors,
        rtol=1e-4, atol=1e-5), indent=2))
    if not passed:
        raise RuntimeError(f'QONNX verification FAILED, max error={max(errors)}. See {out}')
    print(f'QONNX verification: PASS (22 samples), max_error={max(errors):.8g}')
    print(f'Timing (s): export={onnx_seconds:.3f}, cleanup={cleanup_seconds:.3f}, '
          f'verification={verification_seconds:.3f}, total={total_seconds:.3f}')
    print('Saved:', out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'test', 'export'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--wbits', type=int, choices=[2, 3, 4, 6, 8],
                   help='Training weight bits (default: 4); export uses checkpoint config')
    p.add_argument('--abits', type=int, choices=[2, 3, 4, 6, 8],
                   help='Training hidden activation bits (default: 4); input stays 8 bits')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', default='runs_qat')
    args = p.parse_args()
    if args.mode in ('test', 'export') and any(v is not None for v in (args.wbits, args.abits)):
        p.error('test/export read bits from checkpoint; omit --wbits and --abits')
    if args.mode == 'train':
        args.wbits = 4 if args.wbits is None else args.wbits
        args.abits = 4 if args.abits is None else args.abits
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        p.error('epochs, batch-size, threads and lr must be positive')
    torch.set_num_threads(args.threads)
    if args.mode == 'test':
        test_checkpoint(args, QATCNN, quantized=True)
    else:
        (train if args.mode == 'train' else export)(args)


if __name__ == '__main__':
    main()
