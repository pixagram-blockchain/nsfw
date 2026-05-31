#!/usr/bin/env python
"""
export_model.py
Export viddexa/nsfw-detection-2-nano (EfficientNet-b0 classifier) to ONNX for
the browser, then quantize to INT8 — VERIFYING every artifact against PyTorch so
a broken graph can never ship.

THE PROBLEM THIS HANDLES
  EfficientNet is built on depthwise (grouped) convolutions. The ONNX
  constant-folding pass fuses BatchNorm into the preceding conv, and that fusion
  has historically been mis-applied to grouped convs — corrupting the weights so
  the features collapse and every logit comes out ~0 (a flat softmax). Both the
  Optimum exporter and a plain torch.onnx.export use folding by default, which is
  why they failed identically.

STRATEGY
  Try several export configurations, leading with constant-folding DISABLED, and
  keep the FIRST whose output matches PyTorch (max|PT-ONNX| < 1e-3). No guessing:
  if a config is wrong it's rejected automatically.

Outputs (consumed by scripts/embed-assets.mjs):
    model/nsfw.onnx          FP32 export (verified == PyTorch)
    model/nsfw.int8.onnx     INT8 static-quantized (verified) OR a copy of FP32
    model/labels.json        class order from config.id2label
    model/preprocess.json    params from the model's image processor

Install:
    pip install transformers torch onnx onnxruntime pillow numpy
    # optional, enables the dynamo fallback exporter:
    pip install onnxscript

Run (optionally pass a real image to see a sharp prediction in the check):
    python scripts/export_model.py [sample_image.jpg]
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import logging
for _n in ("onnxscript", "onnx", "torch.onnx"):
    logging.getLogger(_n).setLevel(logging.ERROR)
import glob
import json
import shutil
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import onnxruntime as ort

MODEL_ID = "viddexa/nsfw-detection-2-nano"
OUT_DIR = "model"
CALIB_DIR = os.path.join(OUT_DIR, "calib")
os.makedirs(OUT_DIR, exist_ok=True)
sample_image = sys.argv[1] if len(sys.argv) > 1 else None

# ── Load the (known-good) model in eval mode + the slow processor ────────
print(f"[export] loading {MODEL_ID}")
config = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID).eval()

# ── ROOT-CAUSE FIX: HF EfficientNet's global pooler is nn.AvgPool2d(hidden_dim),
# an oversized fixed kernel used as a "global" average pool. In PyTorch eager the
# window is clamped to the 7x7 feature map and divided by 49 (correct mean), but
# ONNX AveragePool divides by the literal kernel area (~1280*1280), so the pooled
# features come out ~0 and every logit collapses to the classifier bias — the flat
# softmax we saw. Swap it for AdaptiveAvgPool2d(1) -> ONNX GlobalAveragePool, which
# takes the true spatial mean at any size. Identical in PyTorch; correct in ONNX.
import torch.nn as nn  # noqa: E402
_pool_fixed = 0
def _fix_global_pool(mod):
    global _pool_fixed
    for _nm, _ch in list(mod.named_children()):
        if isinstance(_ch, nn.AvgPool2d):          # SE blocks use AdaptiveAvgPool2d, not this
            setattr(mod, _nm, nn.AdaptiveAvgPool2d(output_size=1))
            _pool_fixed += 1
        else:
            _fix_global_pool(_ch)
_fix_global_pool(model)
print(f"[export] ONNX-safe pooling: replaced {_pool_fixed} AvgPool2d -> AdaptiveAvgPool2d(1)")
try:
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
except TypeError:
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)

labels = [config.id2label[i] for i in sorted(config.id2label)]
with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
    json.dump(labels, f)
print(f"[export] labels: {labels}")

size_cfg = getattr(proc, "size", {}) or {}
crop_cfg = getattr(proc, "crop_size", {}) or {}
SIZE = int(size_cfg.get("height") or size_cfg.get("shortest_edge") or 224)
preprocess = {
    "size": SIZE,
    "cropSize": int(crop_cfg.get("height")) if crop_cfg else None,
    "doCenterCrop": bool(getattr(proc, "do_center_crop", False)),
    "rescaleFactor": float(getattr(proc, "rescale_factor", 1 / 255)),
    "rescaleOffset": bool(getattr(proc, "rescale_offset", False)),
    "doNormalize": bool(getattr(proc, "do_normalize", True)),
    "mean": [float(x) for x in getattr(proc, "image_mean", [0.485, 0.456, 0.406])],
    "std": [float(x) for x in getattr(proc, "image_std", [0.229, 0.224, 0.225])],
    "includeTop": bool(getattr(proc, "include_top", True)),
}
with open(os.path.join(OUT_DIR, "preprocess.json"), "w") as f:
    json.dump(preprocess, f, indent=2)
print(f"[export] preprocess: {preprocess}")


class LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).logits


wrapped = LogitsOnly(model).eval()
for mod in wrapped.modules():
    if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        mod.eval()

torch.manual_seed(0)
dummy = torch.randn(1, 3, SIZE, SIZE)
with torch.no_grad():
    pt_dummy = wrapped(dummy).numpy().reshape(1, -1)


def onnx_logits(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.asarray(sess.run(None, {name: x.numpy().astype(np.float32)})[0]).reshape(1, -1)


def softmax(z):
    z = np.asarray(z).reshape(-1)
    e = np.exp(z - z.max())
    return e / e.sum()


# ── Export strategies, most-likely-correct first ─────────────────────────
def exp_legacy(path, opset, fold):
    kw = dict(
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["logits"],
        do_constant_folding=fold,
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    )
    with torch.no_grad():
        try:
            torch.onnx.export(wrapped, (dummy,), path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(wrapped, (dummy,), path, **kw)


def exp_dynamo(path, opset, fold):
    # New FX/torch.export-based exporter. Use its NATIVE opset: forcing a
    # target opset makes onnxscript down-convert, which lacks a Pad adapter.
    with torch.no_grad():
        torch.onnx.export(wrapped, (dummy,), path, dynamo=True)


strategies = [
    ("legacy, fold=OFF, opset17", lambda p: exp_legacy(p, 17, False)),
    ("legacy, fold=OFF, opset14", lambda p: exp_legacy(p, 14, False)),
    ("legacy, fold=OFF, opset20", lambda p: exp_legacy(p, 20, False)),
    ("legacy, fold=OFF, opset19", lambda p: exp_legacy(p, 19, False)),
    ("legacy, fold=ON,  opset17", lambda p: exp_legacy(p, 17, True)),
    ("legacy, fold=ON,  opset13", lambda p: exp_legacy(p, 13, True)),
    ("dynamo, native opset     ", lambda p: exp_dynamo(p, 0, False)),
]

fp32_path = os.path.join(OUT_DIR, "nsfw.onnx")
tmp_path = os.path.join(OUT_DIR, "_try.onnx")
chosen = None
print("\n[export] searching for a faithful export configuration:")
for name, fn in strategies:
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        fn(tmp_path)
        diff = float(np.abs(pt_dummy - onnx_logits(tmp_path, dummy)).max())
        ok = diff < 1e-3
        print(f"  - {name:28s} max|PT-ONNX|={diff:.6f}  {'✓ PASS' if ok else 'reject'}")
        if ok:
            shutil.move(tmp_path, fp32_path)
            chosen = name
            break
    except Exception as e:  # noqa: BLE001
        print(f"  - {name:28s} export error: {str(e)[:80]}")

if os.path.exists(tmp_path):
    os.remove(tmp_path)

if not chosen:
    print("\n[export] FAILED: no configuration matched PyTorch. Do NOT ship.")
    print("[export] This points to an op-level tracing bug. Next step: run a")
    print("[export] layer-divergence probe (ask for diagnose_layers.py) to find")
    print("[export] the first diverging op, or try a different torch version.")
    sys.exit(2)

print(f"\n[verify] FP32 export OK via: {chosen}")
if sample_image and os.path.exists(sample_image):
    px = proc(images=Image.open(sample_image).convert("RGB"), return_tensors="pt")["pixel_values"]
    probs = softmax(onnx_logits(fp32_path, px))
    order = np.argsort(probs)[::-1]
    print(f"[verify] FP32 on {os.path.basename(sample_image)} spread={probs.max()-probs.min():.3f}  "
          + ", ".join(f"{labels[i]}={probs[i]:.3f}" for i in order))

# ── INT8 static quantization (correct for CNNs), with verification ───────
int8_path = os.path.join(OUT_DIR, "nsfw.int8.onnx")
from onnxruntime.quantization import (  # noqa: E402
    quantize_static, CalibrationDataReader, QuantType, QuantFormat,
)

src_for_quant = fp32_path
prepped = os.path.join(OUT_DIR, "nsfw.prep.onnx")
try:
    from onnxruntime.quantization import quant_pre_process

    quant_pre_process(fp32_path, prepped)
    src_for_quant = prepped
except Exception as e:  # noqa: BLE001
    print(f"[export] quant_pre_process unavailable ({e}); using raw FP32")

input_name = ort.InferenceSession(
    src_for_quant, providers=["CPUExecutionProvider"]
).get_inputs()[0].name


def preprocess_path(p):
    return proc(images=Image.open(p).convert("RGB"), return_tensors="np")["pixel_values"].astype(np.float32)


class ImageCalib(CalibrationDataReader):
    def __init__(self, files):
        self.files = files
        self.i = 0

    def get_next(self):
        while self.i < len(self.files):
            f = self.files[self.i]
            self.i += 1
            try:
                return {input_name: preprocess_path(f)}
            except Exception as e:  # noqa: BLE001
                print(f"[calib] skip {f}: {e}")
        return None

    def rewind(self):
        self.i = 0


calib_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
    calib_files += glob.glob(os.path.join(CALIB_DIR, ext))
    calib_files += glob.glob(os.path.join(CALIB_DIR, "**", ext), recursive=True)
calib_files = sorted(set(calib_files))[:400]

if calib_files:
    print(f"\n[export] static-quantizing INT8 with {len(calib_files)} calibration images")
    quantize_static(
        src_for_quant, int8_path,
        calibration_data_reader=ImageCalib(calib_files),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
    )
    probe = torch.from_numpy(
        preprocess_path(sample_image) if (sample_image and os.path.exists(sample_image))
        else dummy.numpy().astype(np.float32)
    )
    fp32_probs = softmax(onnx_logits(fp32_path, probe))
    int8_probs = softmax(onnx_logits(int8_path, probe))
    spread = float(int8_probs.max() - int8_probs.min())
    agree = int(np.argmax(fp32_probs)) == int(np.argmax(int8_probs))
    print(f"[verify] INT8 spread={spread:.3f}  argmax matches FP32: {agree}")
    if spread < 0.05 or not agree:
        print("[verify] INT8 degraded; shipping FP32 as the INT8 file instead.")
        shutil.copyfile(fp32_path, int8_path)
    else:
        print("[verify] INT8 OK ✓")
else:
    print(f"\n[export] no calibration images in {CALIB_DIR}/ — copying FP32 -> INT8 file")
    print("[export] (correct but ~16 MB; add images for the ~4 MB INT8 build)")
    shutil.copyfile(fp32_path, int8_path)

try:
    if os.path.exists(prepped):
        os.remove(prepped)
except OSError:
    pass

mb = os.path.getsize(int8_path) / (1024 * 1024)
print(f"\n[export] done. embedded model will be {mb:.2f} MB")
print("[export] next:  npm run embed-model  &&  npm run build")
