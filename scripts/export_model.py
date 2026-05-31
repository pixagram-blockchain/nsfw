#!/usr/bin/env python
"""
export_model.py
Export viddexa/nsfw-detection-2-nano (EfficientNet-b0 image classifier) to ONNX
and quantize to INT8 for the browser using STATIC quantization (calibrated,
per-channel, QDQ) — the correct method for conv-heavy CNNs.

Dynamic quantization is deliberately NOT used: it targets matmul ops and tends
to collapse a CNN's outputs toward uniform (every class ~= 1/N). If you saw a
near-uniform softmax, that was the dynamic-quant model.

Also emits the exact label order + preprocessing params so the JS side matches
the Python pipeline.

Outputs (consumed by scripts/embed-assets.mjs):
    model/nsfw.onnx          FP32 export
    model/nsfw.int8.onnx     INT8 static-quantized (embedded by default)
    model/labels.json        class order from config.id2label
    model/preprocess.json    params from the model's image processor

Install:
    pip install "optimum[onnxruntime]" transformers torch onnx pillow

Calibration images (REQUIRED for a good INT8 build):
    Put 100-300 representative images in  model/calib/  (any jpg/png/webp),
    ideally a mix that includes each class. Without them this script copies the
    FP32 model to the INT8 path and warns, so you never ship a collapsed model.

Run:
    python scripts/export_model.py
"""
import os
import glob
import json
import shutil
import numpy as np

from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoConfig, AutoImageProcessor
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
from PIL import Image

MODEL_ID = "viddexa/nsfw-detection-2-nano"
OUT_DIR = "model"
CALIB_DIR = os.path.join(OUT_DIR, "calib")
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Export to ONNX (FP32). The graph expects PREPROCESSED pixel_values
#    [1,3,H,W] and emits logits [1,num_labels]; the image processor is NOT
#    baked in, so JS must reproduce it (see preprocess.json below).
print(f"[export] exporting {MODEL_ID} -> ONNX (FP32)")
m = ORTModelForImageClassification.from_pretrained(MODEL_ID, export=True)
m.save_pretrained(OUT_DIR)
fp32_raw = os.path.join(OUT_DIR, "model.onnx")
nice_fp32 = os.path.join(OUT_DIR, "nsfw.onnx")
if os.path.exists(fp32_raw):
    os.replace(fp32_raw, nice_fp32)

# 2) Labels — authoritative order from id2label (model card prose is inconsistent).
config = AutoConfig.from_pretrained(MODEL_ID)
id2label = config.id2label
labels = [id2label[i] for i in sorted(id2label)]
with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
    json.dump(labels, f)
print(f"[export] labels: {labels}")

# 3) Preprocessing — read the model's actual image processor so JS matches it.
proc = AutoImageProcessor.from_pretrained(MODEL_ID)
size = getattr(proc, "size", {}) or {}
crop = getattr(proc, "crop_size", {}) or {}
preprocess = {
    "size": int(size.get("height") or size.get("shortest_edge") or 224),
    "cropSize": int(crop.get("height")) if crop else None,
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

# 4) Pre-process the graph for quantization (shape inference + folding). This
#    avoids a class of static-quant warnings/inaccuracies. Falls back cleanly
#    if the helper isn't available in your onnxruntime version.
src_for_quant = nice_fp32
prepped = os.path.join(OUT_DIR, "nsfw.prep.onnx")
try:
    from onnxruntime.quantization import quant_pre_process

    quant_pre_process(nice_fp32, prepped)
    src_for_quant = prepped
    print("[export] ran quant_pre_process")
except Exception as e:  # noqa: BLE001
    print(f"[export] quant_pre_process unavailable ({e}); quantizing raw FP32")

# 5) INT8 STATIC quantization (calibrated, per-channel, QDQ) — correct for CNNs.
int8_path = os.path.join(OUT_DIR, "nsfw.int8.onnx")
input_name = ort.InferenceSession(
    src_for_quant, providers=["CPUExecutionProvider"]
).get_inputs()[0].name


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    # Use the model's OWN processor so calibration matches training exactly.
    return proc(images=img, return_tensors="np")["pixel_values"].astype(np.float32)


class ImageCalib(CalibrationDataReader):
    def __init__(self, files):
        self.files = files
        self.i = 0

    def get_next(self):
        while self.i < len(self.files):
            f = self.files[self.i]
            self.i += 1
            try:
                return {input_name: preprocess_image(f)}
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
    print(f"[export] static-quantizing INT8 with {len(calib_files)} calibration images")
    quantize_static(
        src_for_quant,
        int8_path,
        calibration_data_reader=ImageCalib(calib_files),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"[export] wrote {int8_path}")
    print("[export] embed it:  npm run embed-model        (defaults to nsfw.int8.onnx)")
else:
    print(f"[export] NO calibration images found in {CALIB_DIR}/ — NOT quantizing.")
    print("[export] copying FP32 -> nsfw.int8.onnx so you ship a CORRECT (larger) model.")
    shutil.copyfile(nice_fp32, int8_path)
    print(f"[export] add ~200 images to {CALIB_DIR}/ and re-run to get the ~4 MB INT8 build.")

# Tidy the intermediate prepped graph.
try:
    if os.path.exists(prepped):
        os.remove(prepped)
except OSError:
    pass

print("\n[export] done.")
print("[export] then:  npm run embed-model  &&  npm run build")
print("[export] FP32 quick test (correct but ~16 MB):")
print("[export]   node scripts/embed-assets.mjs model/nsfw.onnx && npm run build")
