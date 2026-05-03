#!/usr/bin/env python3
"""
build_lce_tflite.py — Build Larq Compute Engine (LCE) TFLite from NAS-BNN Key4 .npz
======================================================================================
Run inside WSL with the lce_env activated:

    source ~/lce_env/bin/activate
    cd /mnt/c/Projects/PhD/NAS-BNN/WakeVision
    python build_lce_tflite.py [--key 4]

Requirements (Python 3.10, Linux only):
    pip install tensorflow==2.13.0 larq==0.13.3 larq-compute-engine numpy

What this does:
    1. Loads the binary-packed weights from nasbnn_key4_larq_packed.npz
    2. Rebuilds the NAS-BNN Key4 architecture using larq.layers.QuantConv2D
       (true binary {-1,+1} weights + binary activations via Sign activation)
    3. Converts to LCE TFLite via larq_compute_engine.convert
    4. Saves nasbnn_key4_lce.tflite  (~500 KB expected)
    5. Runs a quick accuracy check on a sample of WakeVision test images
"""

import argparse
import os
import sys
import glob
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--key", type=int, default=4, choices=[3, 4, 5, 6])
parser.add_argument("--npz",  type=str, default=None)
parser.add_argument("--out",  type=str, default=None)
parser.add_argument("--test_dir", type=str, default="./data/WakeVision_Full/test")
parser.add_argument("--n_test", type=int, default=2000,
                    help="Number of test images for quick accuracy check (0 = skip)")
args = parser.parse_args()

BUNDLE = "./work_dirs/wakevision_nasbnn_FULLEXP_run/onnx_exports/deployment_bundle"

NPZ_PATH    = args.npz or (
    f"./work_dirs/wakevision_nasbnn_FULLEXP_run/onnx_exports/nasbnn_key{args.key}_larq_packed.npz"
)
OUT_PATH    = args.out or f"{BUNDLE}/nasbnn_key{args.key}_lce.tflite"
KERAS_PATH  = f"{BUNDLE}/nasbnn_key{args.key}_keras.h5"
os.makedirs(BUNDLE, exist_ok=True)

if not os.path.exists(NPZ_PATH):
    print(f"ERROR: NPZ not found: {NPZ_PATH}")
    print("       Run first:  python export_larq_npz.py --key {args.key}")
    sys.exit(1)

print(f"Loading packed weights from: {NPZ_PATH}")
npz = np.load(NPZ_PATH, allow_pickle=False)
print(f"  Keys: {sorted(npz.files)[:6]} ... ({len(npz.files)} total)")

# ── Imports ───────────────────────────────────────────────────────────────────
print("\nImporting TensorFlow / Larq ...")
import tensorflow as tf
import larq
try:
    import larq_compute_engine as lce
    print(f"  tensorflow           : {tf.__version__}")
    print(f"  larq                 : {larq.__version__}")
    print(f"  larq_compute_engine  : {lce.__version__}")
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install with: pip install larq-compute-engine")
    sys.exit(1)

IMG_SIZE = 128
N_CLASSES = 2

# ── Architecture specs per key  (out_ch, ks, g1, g2, stride, is_stem) ─────────
# Decoded from sub_path_tuple stored in each finetuned checkpoint.
# Format: groups of 6 values (stage, block, out_ch, ks, g1, g2) per slot;
# only slots with stage >= 0 are active; stride = 2 for block_idx=0, else 1.

ARCH_KEY3 = [
    # (out_ch, ks, g1, g2, stride, is_stem)
    (32,  3, 1, 1, 2, True),   # s0b0 stem  (32-ch, smaller than keys 4/5/6)
    (64,  5, 1, 1, 2, False),  # s1b0  (no s1b1 in Key3)
    (128, 5, 1, 1, 2, False),  # s2b0
    (192, 3, 2, 1, 1, False),  # s2b1
    (256, 5, 1, 1, 2, False),  # s3b0
    (384, 5, 2, 2, 1, False),  # s3b1
    (512, 3, 4, 2, 2, False),  # s4b0
    (512, 3, 4, 2, 1, False),  # s4b1
]
ARCH_KEY4 = [
    (48,  3, 1, 1, 2, True),   # s0b0 stem
    (64,  3, 1, 1, 2, False),  # s1b0
    (64,  3, 1, 1, 1, False),  # s1b1
    (128, 5, 1, 1, 2, False),  # s2b0
    (128, 3, 1, 1, 1, False),  # s2b1
    (256, 5, 1, 1, 2, False),  # s3b0
    (384, 5, 2, 2, 1, False),  # s3b1
    (512, 3, 4, 2, 2, False),  # s4b0
    (512, 3, 4, 2, 1, False),  # s4b1
]
ARCH_KEY5 = [
    (48,  3, 1, 1, 2, True),   # s0b0 stem
    (64,  5, 1, 1, 2, False),  # s1b0  (ks=5)
    (64,  3, 1, 1, 1, False),  # s1b1
    (128, 5, 1, 1, 2, False),  # s2b0
    (128, 3, 1, 1, 1, False),  # s2b1
    (256, 3, 1, 1, 2, False),  # s3b0  (ks=3)
    (384, 5, 2, 2, 1, False),  # s3b1
    (512, 3, 1, 2, 2, False),  # s4b0  (g1=1, not grouped)
    (512, 5, 4, 2, 1, False),  # s4b1  (ks=5)
]
ARCH_KEY6 = [
    (48,  3, 1, 1, 2, True),   # s0b0 stem
    (64,  5, 1, 1, 2, False),  # s1b0
    (96,  3, 1, 1, 1, False),  # s1b1  (out=96, differs from s1b0=64)
    (256, 3, 1, 1, 2, False),  # s2b0  (out=256)
    (192, 3, 2, 1, 1, False),  # s2b1
    (256, 3, 1, 2, 2, False),  # s3b0  (g2=2)
    (256, 5, 2, 1, 1, False),  # s3b1
    (512, 3, 1, 1, 2, False),  # s4b0  (g1=1, g2=1)
    (512, 5, 4, 1, 1, False),  # s4b1  (g2=1)
]
ARCH = {3: ARCH_KEY3, 4: ARCH_KEY4, 5: ARCH_KEY5, 6: ARCH_KEY6}

# Block name <-> NPZ prefix mapping (PyTorch features.{s}.{b} -> Keras layer prefix)
BLOCK_NPZ_PREFIXES = {
    3: [("features_1_0","s1b0"),("features_2_0","s2b0"),("features_2_1","s2b1"),
        ("features_3_0","s3b0"),("features_3_1","s3b1"),
        ("features_4_0","s4b0"),("features_4_1","s4b1")],
    4: [("features_1_0","s1b0"),("features_1_1","s1b1"),
        ("features_2_0","s2b0"),("features_2_1","s2b1"),
        ("features_3_0","s3b0"),("features_3_1","s3b1"),
        ("features_4_0","s4b0"),("features_4_1","s4b1")],
    5: [("features_1_0","s1b0"),("features_1_1","s1b1"),
        ("features_2_0","s2b0"),("features_2_1","s2b1"),
        ("features_3_0","s3b0"),("features_3_1","s3b1"),
        ("features_4_0","s4b0"),("features_4_1","s4b1")],
    6: [("features_1_0","s1b0"),("features_1_1","s1b1"),
        ("features_2_0","s2b0"),("features_2_1","s2b1"),
        ("features_3_0","s3b0"),("features_3_1","s3b1"),
        ("features_4_0","s4b0"),("features_4_1","s4b1")],
}

arch = ARCH[args.key]


def _npz_key(prefix, suffix):
    """Look up an array in the npz by a flexible prefix match."""
    for k in npz.files:
        if k.startswith(prefix) and k.endswith(suffix):
            return npz[k]
    return None


def _get_bn_weights(block_prefix):
    """Return (gamma, beta, mean, var) for a BN layer."""
    w    = _npz_key(block_prefix, "_bn_w")
    b    = _npz_key(block_prefix, "_bn_b")
    mean = _npz_key(block_prefix, "_bn_mean")
    var  = _npz_key(block_prefix, "_bn_var")
    return w, b, mean, var


# ── Build Keras model using larq ──────────────────────────────────────────────
print("\nBuilding Larq Keras model ...")

# Helper: binary sign activation (Larq uses "ste_sign" as the quantizer name)
def binary_sign():
    return larq.quantizers.SteSign(clip_value=1.0)

def binary_weight_quantizer():
    return larq.quantizers.SteSign(clip_value=1.0)


def build_nasbnn_larq():
    import math
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    stem_ch = arch[0][0]   # stem output channels (32 for Key3, 48 for Keys 4/5/6)
    stem_ks = arch[0][1]   # stem kernel size

    # ── Stem block (float Conv, stride=2) ───────────────────────────────────
    x = tf.keras.layers.Conv2D(
        stem_ch, stem_ks, strides=2, padding="same", use_bias=False, name="stem_conv"
    )(inp)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name="stem_prelu")(x)

    # ── Binary blocks ────────────────────────────────────────────────────────
    # NAS-BNN BasicBlock structure (from models/superbnn.py):
    #   binary_conv (ks×ks, in_ch→in_ch, stride, groups=g1)  ← same channel count
    #   shortcut1: avg_pool if stride>1, then direct add (always same channels)
    #   binary_conv1x1 (1×1, in_ch→out_ch, groups=g2)        ← channel expansion
    #   shortcut2: ceil-repeat + slice if out_ch != in_ch
    block_specs = arch[1:]  # skip stem entry
    block_names = [bname for _, bname in BLOCK_NPZ_PREFIXES[args.key]]

    for spec, bname in zip(block_specs, block_names):
        out_ch, ks, g1, g2, stride, _ = spec
        in_ch = x.shape[-1]

        # ── First binary path: in_ch → in_ch ────────────────────────────────
        residual = x
        act1 = larq.layers.QuantConv2D(
            in_ch, ks,
            strides=stride,
            padding="same",
            groups=g1,
            use_bias=False,
            input_quantizer=binary_sign(),
            kernel_quantizer=binary_weight_quantizer(),
            name=f"{bname}_binconv",
        )(x)
        act1 = tf.keras.layers.BatchNormalization(name=f"{bname}_bn1")(act1)

        # Shortcut1: downsample spatially if strided (channels always match)
        if stride != 1:
            residual = tf.keras.layers.AveragePooling2D(
                pool_size=stride, strides=stride, name=f"{bname}_shortcut1_pool"
            )(residual)
        # act1.shape[-1] == residual.shape[-1] always (binary_conv keeps in_ch)
        x = tf.keras.layers.Add(name=f"{bname}_add1")([act1, residual])
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{bname}_prelu1")(x)

        # ── Second binary path: in_ch → out_ch ──────────────────────────────
        residual2 = x
        act2 = larq.layers.QuantConv2D(
            out_ch, 1,
            groups=g2,
            padding="same",
            use_bias=False,
            input_quantizer=binary_sign(),
            kernel_quantizer=binary_weight_quantizer(),
            name=f"{bname}_binconv1x1",
        )(x)
        act2 = tf.keras.layers.BatchNormalization(name=f"{bname}_bn2")(act2)

        # Shortcut2: repeat + slice to match out_ch (NAS-BNN adaptive_add logic)
        if out_ch != in_ch:
            n_rep = math.ceil(out_ch / in_ch)
            residual2_exp = tf.keras.layers.Concatenate(
                name=f"{bname}_shortcut2_cat"
            )([residual2] * n_rep)
            # Slice to out_ch channels
            residual2 = tf.keras.layers.Lambda(
                lambda t, c=out_ch: t[:, :, :, :c],
                name=f"{bname}_shortcut2_slice"
            )(residual2_exp)
        x = tf.keras.layers.Add(name=f"{bname}_add2")([act2, residual2])
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{bname}_prelu2")(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(N_CLASSES, name="fc")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name=f"nasbnn_key{args.key}_larq")
    return model


model = build_nasbnn_larq()
model.summary(line_length=90)

# ── Load weights from NPZ ─────────────────────────────────────────────────────
print("\nLoading weights from NPZ ...")

def _set_layer_weights(layer, *arrays):
    """Set layer weights, ignoring None arrays."""
    arrays = [a for a in arrays if a is not None]
    if arrays:
        layer.set_weights(arrays)


# Stem
stem_conv_w = _npz_key("features_0_0_conv", "_weight_s")
if stem_conv_w is not None:
    # ONNX/PyTorch weights are (out, in/g, kH, kW) → Keras needs (kH, kW, in/g, out)
    stem_conv_w_keras = stem_conv_w.transpose(2, 3, 1, 0)
    model.get_layer("stem_conv").set_weights([stem_conv_w_keras])
    print(f"  stem_conv: {stem_conv_w_keras.shape}")

bn_w, bn_b, bn_mean, bn_var = _get_bn_weights("features_0_0_bn")
if all(v is not None for v in [bn_w, bn_b, bn_mean, bn_var]):
    model.get_layer("stem_bn").set_weights([bn_w, bn_b, bn_mean, bn_var])
    print(f"  stem_bn: {bn_w.shape}")

# Binary blocks
block_npz_prefixes = BLOCK_NPZ_PREFIXES[args.key]

for npz_prefix, bname in block_npz_prefixes:
    # BinConv weights: (out, in/g, kH, kW) → (kH, kW, in/g, out)
    bc_bits   = _npz_key(f"{npz_prefix}_binary_conv", "_bits")
    bc_shape  = _npz_key(f"{npz_prefix}_binary_conv", "_shape")
    bc_scales = _npz_key(f"{npz_prefix}_binary_conv", "_scales")

    if bc_bits is not None and bc_shape is not None:
        # Unpack bits → {-1, +1}
        total_bits = int(np.prod(bc_shape))
        flat_bits = np.unpackbits(bc_bits, bitorder="big")[:total_bits]
        binary_w = np.where(flat_bits == 1, 1.0, -1.0).reshape(bc_shape)
        # Scale by per-channel alpha (broadcast over spatial dims)
        if bc_scales is not None:
            binary_w = binary_w * bc_scales[:, np.newaxis, np.newaxis, np.newaxis]
        # Transpose to Keras format (kH, kW, in/g, out)
        binary_w_keras = binary_w.transpose(2, 3, 1, 0)
        try:
            model.get_layer(f"{bname}_binconv").set_weights([binary_w_keras])
            print(f"  {bname}_binconv: {binary_w_keras.shape}")
        except Exception as e:
            print(f"  WARNING {bname}_binconv: {e}")

    # BN1  (fold move12 bias into BN1 beta so PReLU1 gets the right offset)
    bn_w, bn_b, bn_mean, bn_var = _get_bn_weights(f"{npz_prefix}_bn1")
    move12 = _npz_key(f"{npz_prefix}_move12", "_bias")
    if all(v is not None for v in [bn_w, bn_b, bn_mean, bn_var]):
        if move12 is not None:
            bn_b = bn_b + move12   # fold additive post-BN shift into beta
        try:
            model.get_layer(f"{bname}_bn1").set_weights([bn_w, bn_b, bn_mean, bn_var])
        except Exception as e:
            print(f"  WARNING {bname}_bn1: {e}")

    # BinConv 1x1
    bc1_bits  = _npz_key(f"{npz_prefix}_binary_conv1x1", "_bits")
    bc1_shape = _npz_key(f"{npz_prefix}_binary_conv1x1", "_shape")
    bc1_scales = _npz_key(f"{npz_prefix}_binary_conv1x1", "_scales")

    if bc1_bits is not None and bc1_shape is not None:
        total_bits = int(np.prod(bc1_shape))
        flat_bits = np.unpackbits(bc1_bits, bitorder="big")[:total_bits]
        binary_w = np.where(flat_bits == 1, 1.0, -1.0).reshape(bc1_shape)
        if bc1_scales is not None:
            binary_w = binary_w * bc1_scales[:, np.newaxis, np.newaxis, np.newaxis]
        binary_w_keras = binary_w.transpose(2, 3, 1, 0)
        try:
            model.get_layer(f"{bname}_binconv1x1").set_weights([binary_w_keras])
            print(f"  {bname}_binconv1x1: {binary_w_keras.shape}")
        except Exception as e:
            print(f"  WARNING {bname}_binconv1x1: {e}")

    # BN2  (fold move22 bias into BN2 beta so PReLU2 gets the right offset)
    bn_w, bn_b, bn_mean, bn_var = _get_bn_weights(f"{npz_prefix}_bn2")
    move22 = _npz_key(f"{npz_prefix}_move22", "_bias")
    if all(v is not None for v in [bn_w, bn_b, bn_mean, bn_var]):
        if move22 is not None:
            bn_b = bn_b + move22   # fold additive post-BN shift into beta
        try:
            model.get_layer(f"{bname}_bn2").set_weights([bn_w, bn_b, bn_mean, bn_var])
        except Exception as e:
            print(f"  WARNING {bname}_bn2: {e}")

    # PReLU1 & PReLU2
    for prelu_name, npz_suffix in [
        (f"{bname}_prelu1", f"{npz_prefix}_prelu1_w"),
        (f"{bname}_prelu2", f"{npz_prefix}_prelu2_w"),
    ]:
        alpha = _npz_key(npz_suffix.rsplit('_', 1)[0],
                         f"_{npz_suffix.rsplit('_', 1)[1]}")
        if alpha is None:
            # Try direct lookup without prefix split
            for k in npz.files:
                if k == npz_suffix:
                    alpha = npz[k]
                    break
        if alpha is not None:
            try:
                # Keras PReLU(shared_axes=[1,2]) weight shape: (1,1,C)
                alpha_keras = alpha.reshape(1, 1, -1)
                model.get_layer(prelu_name).set_weights([alpha_keras])
            except Exception as e:
                print(f"  WARNING {prelu_name}: {e}")

# FC
fc_w = _npz_key("fc", "_weight_s")
fc_b = _npz_key("fc", "_bias")
if fc_w is not None:
    # PyTorch FC: (out, in) → Keras: (in, out)
    fc_w_keras = fc_w.T
    weights = [fc_w_keras]
    if fc_b is not None:
        weights.append(fc_b)
    model.get_layer("fc").set_weights(weights)
    print(f"  fc: {fc_w_keras.shape}")

# Stem PReLU
stem_prelu_alpha = None
for k in npz.files:
    if k == "features_0_0_relu_prelu_w":
        stem_prelu_alpha = npz[k]
        break
if stem_prelu_alpha is not None:
    try:
        model.get_layer("stem_prelu").set_weights(
            [stem_prelu_alpha.reshape(1, 1, -1)]
        )
        print(f"  stem_prelu: loaded alpha {stem_prelu_alpha.shape}")
    except Exception as e:
        print(f"  WARNING stem_prelu: {e}")

# Stem BN: fold move1 bias into BN beta if present
move1 = None
for k in npz.files:
    if k == "features_0_0_move1_bias":
        move1 = npz[k]
        break
if move1 is not None:
    try:
        w, b, mean, var = model.get_layer("stem_bn").get_weights()
        model.get_layer("stem_bn").set_weights([w, b + move1, mean, var])
        print(f"  stem_bn: folded move1 bias into beta")
    except Exception as e:
        print(f"  WARNING stem_bn move1 fold: {e}")

print("  Weight loading complete.")

# ── Save as TF SavedModel (preferred for PTQ — handles Lambda layers cleanly) ─
SM_PATH = f"{BUNDLE}/nasbnn_key{args.key}_saved_model"
print(f"\nSaving SavedModel to: {SM_PATH}")
try:
    model.save(SM_PATH)   # saves as directory (no custom_objects issue)
    print(f"  SavedModel saved OK")
except Exception as e:
    print(f"  WARNING: could not save SavedModel: {e}")

# ── Also save .h5 as fallback ────────────────────────────────────────────────
print(f"Saving Keras .h5 to: {KERAS_PATH}")
try:
    model.save(KERAS_PATH)
    print(f"  Saved {os.path.getsize(KERAS_PATH)/1024:.0f} KB")
except Exception as e:
    print(f"  WARNING: could not save .h5: {e}")

# ── Convert to LCE TFLite ─────────────────────────────────────────────────────
print(f"\nConverting to LCE TFLite ...")
lce_tflite = lce.convert_keras_model(model)

with open(OUT_PATH, "wb") as f:
    f.write(lce_tflite)

kb = len(lce_tflite) / 1024
print(f"\n✅ LCE TFLite saved: {OUT_PATH}")
print(f"   File size: {kb:.0f} KB  (expected ~500 KB with binary packing)")
print(f"   Compare  : mcunet-320kb = 923.8 KB")

# ── Quick accuracy check ───────────────────────────────────────────────────────
if args.n_test > 0 and os.path.isdir(args.test_dir):
    print(f"\nRunning quick accuracy check on {args.n_test} images ...")
    from PIL import Image

    # Collect test images
    paths, labels = [], []
    for cls_dir in sorted(os.listdir(args.test_dir)):
        full = os.path.join(args.test_dir, cls_dir)
        if not os.path.isdir(full):
            continue
        lbl = 1 if "person_present" in cls_dir and "no_" not in cls_dir else 0
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob.glob(os.path.join(full, ext)):
                paths.append(p)
                labels.append(lbl)
    paths = paths[:args.n_test]
    labels = np.array(labels[:args.n_test])

    # Run LCE interpreter
    interpreter = lce.testing.Interpreter(lce_tflite)
    input_idx  = interpreter.allocate_tensors()

    preds = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        arr = arr[np.newaxis]  # (1, H, W, C) NHWC
        out = interpreter.predict(arr)
        preds.append(np.argmax(out[0]))

    preds = np.array(preds)
    acc = (preds == labels).mean() * 100
    print(f"  Accuracy on {len(paths)} images: {acc:.2f}%")
    print(f"  Reference (PyTorch packed): 80.44%")
else:
    print(f"\nSkipping accuracy check (test_dir not found or n_test=0)")

print("\nDone. Upload the LCE TFLite to STM32AI Developer Cloud for Flash/RAM benchmarks.")
print(f"  File: {OUT_PATH}")
