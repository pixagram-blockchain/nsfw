#!/usr/bin/env python
"""
diagnose_model.py — pinpoint WHY the logits are flat (~0 for every class).

Runs the model three ways on ONE image and prints a verdict:
  1. Checks whether the fine-tuned weights actually LOADED (missing/mismatched
     keys + the classifier head's weight norm). Random/zero head = flat logits.
  2. Runs the OFFICIAL PyTorch path (AutoModelForImageClassification), the
     ground truth the model card documents.
  3. Runs the FP32 ONNX export on the SAME preprocessed tensor, if present.

    python scripts/diagnose_model.py path/to/any_image.jpg

Reading the verdict:
  - "HEAD NOT LOADED"            -> the checkpoint didn't map onto the model
                                    (wrong class / revision / repo). Fix loading.
  - PyTorch FLAT  + ONNX FLAT    -> model or preprocessing, NOT the export.
  - PyTorch SHARP + ONNX FLAT    -> the Optimum ONNX export is broken; re-export.
  - PyTorch SHARP + ONNX SHARP   -> Python is fine; your earlier flat result was
                                    the OLD dynamic-quant model or a JS issue.
"""
import sys
import os
import warnings
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

MODEL_ID = "viddexa/nsfw-detection-2-nano"
FP32_ONNX = os.path.join("model", "nsfw.onnx")

if len(sys.argv) < 2:
    print("usage: python scripts/diagnose_model.py <image>")
    sys.exit(1)
img_path = sys.argv[1]


def summarize(tag, logits, labels):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    spread = float(probs.max() - probs.min())
    order = np.argsort(probs)[::-1]
    verdict = "FLAT (broken)" if spread < 0.05 else "SHARP (working)"
    print(f"\n  [{tag}] {verdict}  spread={spread:.3f}")
    print("    probs : " + ", ".join(f"{labels[i]}={probs[i]:.3f}" for i in order))
    print("    logits: [" + ", ".join(f"{v:+.3f}" for v in logits) + "]")
    return spread


# ── 1) Load with loading-info so we can SEE if weights mapped ────────────
print(f"[diag] loading {MODEL_ID} (use_fast=False, per the model card)")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model, info = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, output_loading_info=True
    )
    model.eval()
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)

labels = [model.config.id2label[i] for i in sorted(model.config.id2label)]
print(f"[diag] labels (id order): {labels}")

missing = info.get("missing_keys", [])
unexpected = info.get("unexpected_keys", [])
mismatched = info.get("mismatched_keys", [])
print(f"[diag] missing_keys={len(missing)}  unexpected_keys={len(unexpected)}  mismatched_keys={len(mismatched)}")
head_related = [k for k in missing if "classif" in k.lower() or "logit" in k.lower()]
if missing:
    print(f"[diag]   first missing: {missing[:6]}")
if head_related:
    print(f"[diag]   !! classifier-head keys are MISSING -> head is randomly initialized: {head_related}")
for w in caught:
    if "newly initialized" in str(w.message) or "not used" in str(w.message):
        print(f"[diag]   load warning: {str(w.message)[:160]}")

# Classifier head weight norm — a near-zero or tiny norm is a red flag.
head_norm = None
for name, p in model.named_parameters():
    if name.endswith("classifier.weight") or name.endswith("classifier.1.weight"):
        head_norm = float(p.detach().float().norm())
        print(f"[diag] head weight '{name}' L2 norm = {head_norm:.4f}  shape={tuple(p.shape)}")
        break

# ── 2) Preprocess + official PyTorch inference ───────────────────────────
img = Image.open(img_path).convert("RGB")
inputs = processor(images=img, return_tensors="pt")
pv = inputs["pixel_values"]
print(
    f"\n[diag] pixel_values: shape={tuple(pv.shape)} "
    f"min={pv.min():.3f} max={pv.max():.3f} mean={pv.mean():.3f} std={pv.std():.3f}"
)
if float(pv.std()) < 1e-3:
    print("[diag]   !! pixel_values are nearly constant -> preprocessing produced degenerate input")

with torch.no_grad():
    pt_logits = model(**inputs).logits[0].cpu().numpy()
pt_spread = summarize("PyTorch (official)", pt_logits, labels)

# ── 3) FP32 ONNX on the SAME tensor ──────────────────────────────────────
onnx_spread = None
if os.path.exists(FP32_ONNX):
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        onnx_logits = np.asarray(sess.run(None, {name: pv.numpy().astype(np.float32)})[0]).reshape(-1)
        onnx_spread = summarize("ONNX FP32", onnx_logits, labels)
        diff = float(np.abs(pt_logits.reshape(-1) - onnx_logits).max())
        print(f"\n[diag] max |PyTorch - ONNX| logit diff = {diff:.4f}  (small = export is faithful)")
    except Exception as e:  # noqa: BLE001
        print(f"\n[diag] ONNX run failed: {e}")
else:
    print(f"\n[diag] {FP32_ONNX} not found — skipping ONNX comparison (run export_model.py to create it)")

# ── Verdict ──────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
if head_related or (head_norm is not None and head_norm < 1e-3):
    print("VERDICT: the classifier head did NOT load (random/zero weights).")
    print("  -> the checkpoint isn't mapping onto AutoModelForImageClassification.")
    print("     Check: exact repo id, that safetensors downloaded, transformers")
    print("     version supports this efficientnet config, and trust_remote_code.")
elif pt_spread < 0.05:
    print("VERDICT: PyTorch itself is FLAT -> NOT an ONNX/quantization problem.")
    print("  -> weights or preprocessing. Confirm pixel_values stats above look")
    print("     sane; try the model card's exact processor settings.")
elif onnx_spread is not None and onnx_spread < 0.05:
    print("VERDICT: PyTorch is SHARP but ONNX is FLAT -> the Optimum EXPORT broke.")
    print("  -> re-export (see notes); the PyTorch model and weights are fine.")
else:
    print("VERDICT: Python path is SHARP and correct.")
    print("  -> your earlier flat numbers were the OLD dynamic-quant model, or a")
    print("     JS-side issue (embedded preprocess/labels, wasmPaths, canvas).")
print("=" * 64)
