#!/usr/bin/env python
"""
sanity_check.py — isolate WHERE NSFW detection breaks.

Runs the FP32 export and the INT8 model on the SAME image(s) in Python, using
the model's own image processor. This removes JS preprocessing, the canvas, and
the Web Worker from the equation, so you learn exactly which layer is at fault.

    python scripts/sanity_check.py path/to/safe.jpg path/to/nsfw.jpg ...

How to read the output (look at `spread` = max prob - min prob):
  - INT8 spread ~0.00 (uniform) but FP32 spread large/decisive
        -> quantization collapsed the model. Re-quantize STATIC (export_model.py).
  - BOTH near-uniform
        -> the export/model itself is the problem. Re-export; check opset >= 17.
  - BOTH decisive here in Python, but your BROWSER shows ~0.20 each
        -> the problem is JS-side. Confirm src/assets.generated.ts actually has
           real PREPROCESS + LABELS (not the empty stub / fallback), and that
           wasmPaths is set so ORT loads. The canvas resize is then the place
           to look for parity drift.
"""
import sys
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoConfig
from PIL import Image

MODEL_ID = "viddexa/nsfw-detection-2-nano"
OUT_DIR = "model"

imgs = sys.argv[1:]
if not imgs:
    print("usage: python scripts/sanity_check.py <image> [<image> ...]")
    sys.exit(1)

proc = AutoImageProcessor.from_pretrained(MODEL_ID)
cfg = AutoConfig.from_pretrained(MODEL_ID)
labels = [cfg.id2label[i] for i in sorted(cfg.id2label)]
print(f"[sanity] labels (id order): {labels}")


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def load(path):
    full = os.path.join(OUT_DIR, path)
    if not os.path.exists(full):
        return None, full
    return ort.InferenceSession(full, providers=["CPUExecutionProvider"]), full


def run(sess, path):
    x = proc(images=Image.open(path).convert("RGB"), return_tensors="np")["pixel_values"]
    x = x.astype(np.float32)
    name = sess.get_inputs()[0].name
    logits = np.asarray(sess.run(None, {name: x})[0]).reshape(-1)
    return logits, softmax(logits)


def show(tag, sess, missing, path):
    if sess is None:
        print(f"  {tag}: (missing {missing})")
        return
    logits, probs = run(sess, path)
    order = np.argsort(probs)[::-1]
    spread = float(probs.max() - probs.min())
    top = ", ".join(f"{labels[i]}={probs[i]:.3f}" for i in order)
    flag = "  <-- COLLAPSED" if spread < 0.05 else ""
    print(f"  {tag}: spread={spread:.3f}{flag}")
    print(f"        probs : {top}")
    print(f"        logits: [{', '.join(f'{v:+.2f}' for v in logits)}]")


fp32_sess, fp32_path = load("nsfw.onnx")
int8_sess, int8_path = load("nsfw.int8.onnx")

for p in imgs:
    print(f"\n{os.path.basename(p)}")
    show("FP32", fp32_sess, fp32_path, p)
    show("INT8", int8_sess, int8_path, p)

print("\nspread < 0.05 = collapsed/uniform (bad). spread > ~0.3 = the model is discriminating.")
