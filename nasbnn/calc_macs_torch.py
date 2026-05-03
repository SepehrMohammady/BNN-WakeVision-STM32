"""
Calculate MACs for NAS-BNN Key4 using PyTorch forward hooks.
Bypasses thop to avoid issues with supernet's tuple-returning forward passes.

MACs per layer type:
  DynamicQConv2d / DynamicBinConv2d:
      MACs = N * Hout * Wout * Cout * (Cin/groups) * Kh * Kw
  DynamicFPLinear (FC):
      MACs = N * Cout * Cin
  BatchNorm2d / DynamicBatchNorm2d:
      MACs = N * C * H * W  (1 mul + 1 add per element)
  DynamicPReLU / PReLU:
      MACs = N * C * H * W  (1 compare + 1 mul per element)
  AdaptiveAvgPool2d:
      MACs = N * C * H * W  (sum over window, 1 add per element)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import torch.nn as nn

sys.path.insert(0, ".")
import models
from utils.cand import tuple2cand
from models.dynamic_operations import (
    DynamicQConv2d, DynamicBinConv2d, DynamicFPLinear,
    DynamicBatchNorm2d, DynamicPReLU, DynamicLearnableBias,
)

# ── Configuration ──────────────────────────────────────────────────────────────
KEY             = 4
CHECKPOINT_PATH = ("work_dirs/wakevision_nasbnn_FULLEXP_run/full_eval_results/"
                   "best_checkpoints/nasbnn_key4_best_ep29_acc80.41_f10.7934.pth.tar")
IMG_SIZE        = 128

# ── Build static model ─────────────────────────────────────────────────────────
ckpt  = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
cand  = tuple2cand(ckpt["sub_path_tuple"])
model = models.superbnn_wakevision_large(sub_path=cand, img_size=IMG_SIZE)
model.eval()
model.to_static(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
state = {k.replace("module.", ""): v for k, v in ckpt.get("state_dict", ckpt).items()}
model.load_state_dict(state, strict=False)
model.eval()

# ── Hook registration ──────────────────────────────────────────────────────────
macs_by_module = {}  # name → macs
handles        = []

def _hook(name):
    def _fn(module, inp, out):
        # Unwrap tuple inputs (supernet blocks return (tensor, loss))
        x = inp[0] if isinstance(inp, (tuple, list)) else inp
        if isinstance(x, (tuple, list)):
            x = x[0]
        o = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(x, torch.Tensor) or not isinstance(o, torch.Tensor):
            return

        macs = 0
        t    = type(module)

        if t in (DynamicQConv2d, DynamicBinConv2d):
            N, Cin, _, _    = x.shape
            N, Cout, Ho, Wo = o.shape
            ks     = getattr(module, "active_ks",     getattr(module, "max_ks", 3))
            groups = getattr(module, "active_groups", 1)
            Cin_g  = Cin // max(int(groups), 1)
            macs   = N * Ho * Wo * Cout * Cin_g * ks * ks

        elif t is DynamicFPLinear or t is nn.Linear:
            N    = x.numel() // x.shape[-1]
            Cin  = x.shape[-1]
            Cout = o.shape[-1]
            macs = N * Cout * Cin

        elif t in (nn.BatchNorm2d, DynamicBatchNorm2d):
            macs = int(x.numel())  # 1 mul + 1 add ≈ 1 MAC per element

        elif t in (nn.PReLU, DynamicPReLU):
            macs = int(x.numel())  # 1 compare + 1 mul per element

        elif t is nn.AdaptiveAvgPool2d:
            macs = int(x.numel())  # sums over spatial window

        if macs:
            macs_by_module[name] = macs_by_module.get(name, 0) + int(macs)
    return _fn

for name, m in model.named_modules():
    h = m.register_forward_hook(_hook(name))
    handles.append(h)

# ── Forward pass ──────────────────────────────────────────────────────────────
with torch.no_grad():
    model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))

for h in handles:
    h.remove()

# ── Filter: keep only "leaf-like" layer names (no children with their own MACs)
# to avoid double-counting parent wrappers
leaf_names = set()
for name, m in model.named_modules():
    if type(m) in (DynamicQConv2d, DynamicBinConv2d, DynamicFPLinear,
                   nn.Linear, nn.BatchNorm2d, DynamicBatchNorm2d,
                   DynamicPReLU, nn.AdaptiveAvgPool2d):
        leaf_names.add(name)
    # plain PReLU only if NOT already wrapped inside a DynamicPReLU
    if type(m) is nn.PReLU and not name.endswith(".prelu_s"):
        leaf_names.add(name)

# ── Summary ───────────────────────────────────────────────────────────────────
conv_macs  = 0
bn_macs    = 0
act_macs   = 0
pool_macs  = 0
fc_macs    = 0

rows = []
for name, m in model.named_modules():
    if name not in leaf_names:
        continue
    m_macs = macs_by_module.get(name, 0)
    t = type(m).__name__
    rows.append((name, t, m_macs))

    if "Conv" in t:
        conv_macs += m_macs
    elif "BatchNorm" in t:
        bn_macs += m_macs
    elif "PReLU" in t or "Relu" in t.lower():
        act_macs += m_macs
    elif "Pool" in t:
        pool_macs += m_macs
    elif "Linear" in t:
        fc_macs += m_macs

total_macs = sum(v for _, v in macs_by_module.items()
                 if any(name == _ for name, _, _ in rows))

print()
print(f"  {'Layer':<55} {'Type':<24} {'MACs':>12}")
print("  " + "-" * 95)
for name, t, m in rows:
    if m > 0:
        print(f"  {name:<55} {t:<24} {m:>12,}")

print("  " + "-" * 95)
total_leaf = sum(m for _, _, m in rows)
print(f"  {'TOTAL':<55} {'':24} {total_leaf:>12,}")

print()
print("=" * 67)
print(f"  Model           : NAS-BNN Key {KEY} — WakeVision 128x128 RGB")
print(f"  Checkpoint      : ep29  |  Test acc: 80.41%  |  F1: 0.7934")
print(f"  Input           : (1, 3, {IMG_SIZE}, {IMG_SIZE})")
print("-" * 67)
print(f"  Breakdown:")
print(f"    Conv layers     : {conv_macs:>14,} MACs")
print(f"    BatchNorm       : {bn_macs:>14,} MACs")
print(f"    Activations     : {act_macs:>14,} MACs")
print(f"    Pool            : {pool_macs:>14,} MACs")
print(f"    FC              : {fc_macs:>14,} MACs")
print("-" * 67)
print(f"  Total MACs      : {total_leaf:>20,}")
print(f"  Total (M)       : {total_leaf/1e6:>20.3f} M MACs")
print(f"  GFLOPs (2xMACs) : {total_leaf*2/1e9:>20.4f} GFLOPS")
print("-" * 67)
print(f"  Note: Conv layers are 1-bit (binary). In hardware,")
print(f"  binary MACs use XNOR+popcount (64x fewer bit-ops vs float32).")
print(f"  BOPs (Conv MACs / 32) : {conv_macs//32:>12,}   ({conv_macs/32/1e6:.3f}M BOPs)")
print("=" * 67)
