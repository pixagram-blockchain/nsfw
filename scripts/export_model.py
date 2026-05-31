#!/usr/bin/env python
"""
export_model.py
Export viddexa/nsfw-detection-2-nano (EfficientNet-b0 image classifier) to ONNX,
quantize to INT8 for the browser, and emit the EXACT label order + preprocessing
parameters so the JS side matches the Python pipeline.

Outputs (consumed by scripts/embed-assets.mjs):
    model/nsfw.onnx          FP32 export
    model/nsfw.int8.onnx     INT8 quantized (this is what gets embedded)
    model/labels.json        class order from config.id2label
    model/preprocess.json    params from the model's image processor

Install:
    pip install "optimum[exporters]" onnxruntime transformers torch onnx

Run:
    python scripts/export_model.py
"""
import json
import os

from optimum.onnxruntime import ORTModelForImageClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoConfig, AutoImageProcessor

MODEL_ID = "viddexa/nsfw-detection-2-nano"
OUT_DIR = "model"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Export to ONNX (FP32). Optimum produces a model expecting preprocessed
#    pixel_values [1, 3, H, W] and emitting logits [1, num_labels].
print(f"[export] exporting {MODEL_ID} -> ONNX")
ort_model = ORTModelForImageClassification.from_pretrained(MODEL_ID, export=True)
ort_model.save_pretrained(OUT_DIR)  # writes model.onnx + config

fp32_path = os.path.join(OUT_DIR, "model.onnx")
nice_fp32 = os.path.join(OUT_DIR, "nsfw.onnx")
if os.path.exists(fp32_path):
    os.replace(fp32_path, nice_fp32)

# 2) INT8 dynamic quantization (quick path). For best accuracy on this CNN,
#    swap for quantize_static() with a CalibrationDataReader over a few hundred
#    representative images — dynamic mainly helps matmul ops, less so convs.
int8_path = os.path.join(OUT_DIR, "nsfw.int8.onnx")
print("[export] quantizing INT8 (dynamic, QUInt8)")
quantize_dynamic(nice_fp32, int8_path, weight_type=QuantType.QUInt8)

# 3) Labels — the AUTHORITATIVE order (the model card prose is inconsistent).
config = AutoConfig.from_pretrained(MODEL_ID)
id2label = config.id2label  # {0: "...", 1: "...", ...}
labels = [id2label[i] for i in sorted(id2label)]
with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
    json.dump(labels, f)
print(f"[export] labels: {labels}")

# 4) Preprocessing — read the model's actual image processor so JS matches it.
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
    "mean": list(getattr(proc, "image_mean", [0.485, 0.456, 0.406])),
    "std": list(getattr(proc, "image_std", [0.229, 0.224, 0.225])),
    "includeTop": bool(getattr(proc, "include_top", True)),
}
with open(os.path.join(OUT_DIR, "preprocess.json"), "w") as f:
    json.dump(preprocess, f, indent=2)
print(f"[export] preprocess: {preprocess}")

print("\n[export] done. Next:  npm run embed-model  &&  npm run build")
