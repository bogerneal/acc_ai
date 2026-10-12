"""MNIST FP32 baseline: 5x5 valid convs, pooling, FC-ReLU-dropout-FC.

python mnist_cnn.py check
python mnist_cnn.py train --epochs 5
python mnist_cnn.py test --checkpoint runs/<run>/best.pth
python mnist_cnn.py export --checkpoint runs/<run>/best.pth

This is a floating-point baseline, not QONNX or FINN-ONNX.
"""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from torch import nn


class DigitCNN(nn.Module):
    def __init__(self, n1=16, n2=32, n3=128, dropout=0.5):
        super().__init__()
        self.config = dict(n1=n1, n2=n2, n3=n3, dropout=dropout)
        self.conv1 = nn.Conv2d(1, n1, kernel_size=5, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc3 = nn.Linear(n2 * 4 * 4, n3)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc4 = nn.Linear(n3, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc3(x)))
        return self.fc4(x)  # logits; CrossEntropyLoss applies log-softmax internally


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_model(model):
    model.eval()
    x = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        print('input:', tuple(x.shape))
        for name, layer in model.named_children():
            x = layer(x)
            print(f'{name:10s}: {tuple(x.shape)}')
        assert x.shape == (1, 10)
        assert model(torch.rand(7, 1, 28, 28)).shape == (7, 10)
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    print('Shape check: PASS (batch sizes 1 and 7)')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += y.numel()
    return loss_sum / count, correct / count


def train(args):
    import torchvision
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    if args.epochs < 1:
        raise ValueError('--epochs must be positive')
    seed_all(args.seed)
    device = torch.device(args.device)
    model = DigitCNN().to(device)
    transform = transforms.ToTensor()  # grayscale [0,255] -> float32 [0,1]
    full_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    train_set, val_set = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(args.seed)
    )
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=256, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=0)
    run = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    print('Run directory:', run.resolve(), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    with (run / 'history.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = correct = count = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                count += y.numel()
                loss_sum += loss.item() * y.numel()
                correct += (logits.argmax(1) == y).sum().item()
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            writer.writerow([epoch, loss_sum/count, correct/count, val_loss, val_accuracy])
            file.flush()
            print(f'Epoch {epoch}/{args.epochs}: train_loss={loss_sum/count:.4f} '
                  f'train_acc={correct/count:.2%} val_acc={val_accuracy:.2%}', flush=True)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({'model_state_dict': {k: v.detach().cpu() for k,v in model.state_dict().items()},
                            'config': model.config, 'epoch': epoch, 'seed': args.seed,
                            'val_accuracy': val_accuracy, 'preprocessing': 'ToTensor: grayscale [0,1]'},
                           run / 'best.pth')
    checkpoint = torch.load(run / 'best.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    test_start = perf_counter()
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    test_seconds = perf_counter() - test_start
    report = {'model': model.config, 'seed': args.seed, 'epochs': args.epochs,
              'lr': args.lr, 'batch_size': args.batch_size, 'device': str(device),
              'split': {'train': 55000, 'validation': 5000, 'test': 10000},
              'best_epoch': checkpoint['epoch'], 'best_val_accuracy': best_accuracy,
              'test_loss': test_loss, 'test_accuracy': test_accuracy,
              'test_seconds': test_seconds,
              'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
              'preprocessing': 'ToTensor: grayscale [0,1]', 'quantized': False}
    (run / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(f'Best checkpoint test accuracy: {test_accuracy:.2%}', flush=True)
    print('Saved:', run / 'best.pth')



def test_checkpoint(args, model_class, quantized=False):
    """Evaluate a saved model on all MNIST test images without training."""
    import torchvision
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    is_qat = 'wbits' in config or 'abits' in config
    if is_qat != quantized:
        expected = 'QAT' if quantized else 'FP32'
        raise ValueError(f'This command requires a {expected} checkpoint')
    seed = int(checkpoint.get('seed', 42))
    seed_all(seed)
    device = torch.device(getattr(args, 'device', 'cpu'))
    model = model_class(**config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    dataset = MNIST(args.data_dir, train=False, download=False, transform=ToTensor())
    if len(dataset) != 10000:
        raise ValueError(f'Expected 10000 MNIST test samples, got {len(dataset)}')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    group = f"W{config['wbits']}A{config['abits']}" if quantized else 'FP32'
    print(f'Testing {group}: {checkpoint_path}', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = perf_counter()
    loss, accuracy = evaluate(model, loader, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    report = dict(
        checkpoint=str(checkpoint_path), group=group, config=config,
        quantized=quantized, seed=seed, dataset='MNIST', split='test',
        samples=len(dataset), test_loss=loss, test_accuracy=accuracy,
        test_accuracy_percent=100.0 * accuracy, test_seconds=elapsed,
        batch_size=args.batch_size, threads=torch.get_num_threads(),
        device=str(device), data_dir=str(Path(args.data_dir).resolve()),
        preprocessing='ToTensor: grayscale [0,1]',
        timing_scope='Full test DataLoader evaluation including data transforms, device transfer, '
                     'forward passes and metrics; no warmup; excludes checkpoint/dataset loading '
                     'and report writing. Not pure inference latency.',
        torch=str(torch.__version__), torchvision=str(torchvision.__version__),
    )
    if quantized:
        import brevitas
        report.update(input_bits=8, brevitas=str(brevitas.__version__))
    out = checkpoint_path.parent / ('test_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Test accuracy: {accuracy:.2%}; loss: {loss:.6f}; time: {elapsed:.3f} s')
    print('Report:', out / 'metrics.json')
    return report


def export_model(args):
    total_start = perf_counter()
    import numpy as np
    import onnx
    import onnxruntime as ort
    if not args.checkpoint:
        raise ValueError('export requires --checkpoint runs/<run>/best.pth')
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = DigitCNN(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # disables dropout
    out = checkpoint_path.parent / 'export_fp32'
    out.mkdir(exist_ok=False)
    torch.manual_seed(123)
    x = torch.rand(4, 1, 28, 28)
    path = out / 'mnist_fp32.onnx'
    export_start = perf_counter()
    torch.onnx.export(model, x[:1], str(path), opset_version=13,
                      input_names=['images'], output_names=['logits'],
                      dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}},
                      do_constant_folding=True)
    export_seconds = perf_counter() - export_start
    verification_start = perf_counter()
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        expected = model(x).numpy()
    actual = session.run(None, {'images': x.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    verification_seconds = perf_counter() - verification_start
    np.save(out / 'input.npy', x.numpy())
    np.save(out / 'expected_output.npy', expected)
    report = {'format': 'standard FP32 ONNX; not QONNX or FINN-ONNX',
              'timing_seconds': {'onnx_export': export_seconds,
                                 'verification': verification_seconds,
                                 'total': perf_counter() - total_start},
              'threads': torch.get_num_threads(), 'device': 'cpu',
              'max_abs_error': float(np.max(np.abs(actual-expected))),
              'validation': '4 synthetic samples', 'checkpoint': str(checkpoint_path)}
    (out / 'verification.json').write_text(json.dumps(report, indent=2))
    print('ONNX export check: PASS', report)
    print('Open in Netron:', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'train', 'test', 'export'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='runs')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error('--threads and --batch-size must be positive')
    if args.mode in ('test', 'export') and not args.checkpoint:
        parser.error(f'{args.mode} requires --checkpoint')
    torch.set_num_threads(args.threads)
    if args.mode == 'check':
        check_model(DigitCNN())
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_checkpoint(args, DigitCNN)
    else:
        export_model(args)


if __name__ == '__main__':
    main()
