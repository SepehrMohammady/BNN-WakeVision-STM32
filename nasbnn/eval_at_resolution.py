"""
Eval finetuned NAS-BNN checkpoint on WakeVision test at arbitrary input resolution.
Model uses AdaptiveAvgPool2d(1) so spatial size is flexible without retraining.

Usage:
  python eval_at_resolution.py \
      --ckpt work_dirs/wakevision_nasbnn_FULLEXP_run/full_eval_results/best_checkpoints/nasbnn_key4_best_ep29_acc80.41_f10.7934.pth.tar \
      --arch_key 4 \
      --search_info work_dirs/wakevision_nasbnn_LARGEXP_run/search/info.pth.tar \
      --data_dir data/WakeVision_Full \
      --img_size 32
"""
import argparse, os, sys, time
import torch
import torchvision.transforms as T
import torchvision.datasets as D
import models
import models.dynamic_operations as _dynops
from utils import tuple2cand


def topk_acc(out, tgt, ks=(1,)):
    with torch.no_grad():
        maxk = min(max(ks), out.size(1))
        _, pred = out.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(tgt.view(1, -1).expand_as(pred))
        return [correct[:min(k, maxk)].reshape(-1).float().sum().mul_(100.0 / tgt.size(0)).item() for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch_key', type=int, required=True)
    ap.add_argument('--search_info', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--img_size', type=int, required=True)
    ap.add_argument('--arch_name', default='superbnn_wakevision_large')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Architecture from search file
    si = torch.load(args.search_info, map_location='cpu', weights_only=False)
    if args.arch_key not in si.get('pareto_global', {}):
        sys.exit(f'Key {args.arch_key} not in pareto_global')
    arch = tuple2cand(si['pareto_global'][args.arch_key])

    # Build model. img_size=128 keeps the architecture geometry the supernet was trained for.
    # We still feed args.img_size at inference time; AdaptiveAvgPool absorbs spatial mismatch.
    model = models.__dict__[args.arch_name](sub_path=arch, img_size=128)
    model.to(device).eval()

    # Must call to_static BEFORE load_state_dict: checkpoint was saved after to_static()
    # and contains weight_s keys that only exist post-static-conversion.
    dummy = torch.randn(1, 3, 128, 128, device=device)
    model.to_static(dummy)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt['state_dict'].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'[warn] missing={len(missing)} unexpected={len(unexpected)}')

    # For non-native resolution: disable spatial-size assertions in forward()
    if args.img_size != 128:
        _dynops.SKIP_WH_ASSERT = True

    test_dir = os.path.join(args.data_dir, 'test')
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(args.data_dir, 'val')

    tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = D.ImageFolder(test_dir, tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)

    total = correct1 = 0
    tp = fp = fn = tn = 0
    person_class_idx = ds.class_to_idx.get('person', ds.class_to_idx.get('person_present', 1))

    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out, _ = model(x, model.sub_path)
            pred = out.argmax(1)
            correct1 += (pred == y).sum().item()
            total += y.size(0)
            tp += ((pred == person_class_idx) & (y == person_class_idx)).sum().item()
            fp += ((pred == person_class_idx) & (y != person_class_idx)).sum().item()
            fn += ((pred != person_class_idx) & (y == person_class_idx)).sum().item()
            tn += ((pred != person_class_idx) & (y != person_class_idx)).sum().item()
            if i % 50 == 0:
                print(f'  [{i*args.batch}/{len(ds)}] running acc={100*correct1/total:.2f}%')

    elapsed = time.time() - t0
    acc = 100.0 * correct1 / total
    prec = 100.0 * tp / max(tp + fp, 1)
    rec = 100.0 * tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print('=' * 60)
    print(f'Key {args.arch_key} @ {args.img_size}x{args.img_size}')
    print(f'  N           = {total}')
    print(f'  Top-1 Acc   = {acc:.4f}%')
    print(f'  Precision   = {prec:.4f}%')
    print(f'  Recall      = {rec:.4f}%')
    print(f'  Macro-F1    = {f1:.4f}%')
    print(f'  Wall time   = {elapsed:.1f}s')
    print('=' * 60)


if __name__ == '__main__':
    main()
