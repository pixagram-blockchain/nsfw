#!/usr/bin/env python
"""
diagnose_layers.py — find the EXACT op where the ONNX export diverges.

PyTorch is known-good; every ONNX export collapses the features to ~0. This
exports the broken graph, exposes every intermediate tensor, runs it with all
ORT graph optimizations DISABLED (so no node is fused away), and prints
per-tensor stats in topological order. The first ONNX tensor whose std -> 0 (or
%zeros -> 100) marks the offending op. The PyTorch activation table above it
should stay healthy the whole way, proving the op is fine in eager mode.

    python scripts/diagnose_layers.py [sample_image.jpg]

Paste the two tables back; the collapse point names the op to fix.
"""
import os
import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402
from PIL import Image  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

MODEL_ID = "viddexa/nsfw-detection-2-nano"
sample = sys.argv[1] if len(sys.argv) > 1 else None

print(f"[diag] loading {MODEL_ID}")
model = AutoModelForImageClassification.from_pretrained(MODEL_ID).eval()
try:
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
except TypeError:
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)

sc = getattr(proc, "size", {}) or {}
SIZE = int(sc.get("height") or sc.get("shortest_edge") or 224)
if sample and os.path.exists(sample):
    x = proc(images=Image.open(sample).convert("RGB"), return_tensors="pt")["pixel_values"].float()
    print(f"[diag] input: {os.path.basename(sample)}")
else:
    torch.manual_seed(0)
    x = torch.randn(1, 3, SIZE, SIZE)
    print(f"[diag] input: random {tuple(x.shape)}")


class LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pv):
        return self.m(pixel_values=pv).logits


wrapped = LogitsOnly(model).eval()

# ── PyTorch leaf-module activation stats (execution order) ───────────────
pt = []


def make_hook(nm):
    def hook(mod, inp, out):
        t = out if isinstance(out, torch.Tensor) else (
            out[0] if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor) else None
        )
        if t is None:
            return
        t = t.detach().float()
        pt.append((nm, type(mod).__name__, tuple(t.shape),
                   float(t.mean()), float(t.std()), float((t == 0).float().mean())))
    return hook


for nm, m in wrapped.named_modules():
    if len(list(m.children())) == 0:
        m.register_forward_hook(make_hook(nm))
with torch.no_grad():
    _ = wrapped(x)

# ── Export a broken ONNX to inspect (first config that writes a file) ────
os.makedirs("model", exist_ok=True)
tmp = os.path.join("model", "_probe.onnx")


def export_legacy(opset, fold):
    kw = dict(opset_version=opset, input_names=["pixel_values"], output_names=["logits"],
              do_constant_folding=fold, training=torch.onnx.TrainingMode.EVAL)
    with torch.no_grad():
        try:
            torch.onnx.export(wrapped, (x,), tmp, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(wrapped, (x,), tmp, **kw)


def export_dynamo():
    with torch.no_grad():
        torch.onnx.export(wrapped, (x,), tmp, dynamo=True)


wrote = None
for tag, fn in [("legacy fold=OFF op17", lambda: export_legacy(17, False)),
                ("legacy fold=ON op17", lambda: export_legacy(17, True)),
                ("legacy fold=OFF op14", lambda: export_legacy(14, False)),
                ("dynamo native", export_dynamo)]:
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        fn()
        if os.path.exists(tmp):
            wrote = tag
            break
    except Exception as e:  # noqa: BLE001
        print(f"[diag] export '{tag}' failed: {str(e)[:80]}")
if not wrote:
    print("[diag] could not produce any ONNX to inspect; the exporter errors outright.")
    print("[diag] this strongly implies a torch/exporter problem — try a stable env")
    print("[diag] (python 3.12 + torch==2.4.*) and re-run export_model.py.")
    sys.exit(2)
print(f"[diag] inspecting ONNX produced by: {wrote}")

# ── Expose ALL node outputs; run with optimizations OFF ──────────────────
mo = onnx.load(tmp)
existing = {o.name for o in mo.graph.output}
producer, order = {}, []
for node in mo.graph.node:
    for o in node.output:
        producer[o] = node
        order.append(o)
        if o not in existing:
            vi = onnx.ValueInfoProto()
            vi.name = o
            mo.graph.output.append(vi)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(mo.SerializeToString(), so, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
outs = sess.run(None, {in_name: x.numpy().astype(np.float32)})
name2val = {o.name: v for o, v in zip(sess.get_outputs(), outs)}


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return ((), 0.0, 0.0, 0.0, 0.0, 0.0)
    return (tuple(v.shape), float(v.mean()), float(v.std()), float(v.min()), float(v.max()), float((v == 0).mean()))


print("\n=== PyTorch leaf modules (execution order) — should stay HEALTHY ===")
print(f"{'#':>3} {'module':40} {'type':20} {'shape':20} {'mean':>9} {'std':>9} {'%0':>6}")
for i, (nm, ty, sh, me, sd, z) in enumerate(pt):
    print(f"{i:>3} {nm[-40:]:40} {ty[:20]:20} {str(sh):20} {me:9.3f} {sd:9.3f} {z*100:6.1f}")

# Constant/shape-style ops legitimately have zero variance; never flag them.
CONST_OPS = {"Constant", "ConstantOfShape", "Shape", "Size", "Range", "NonZero"}

xin = x.numpy().astype(np.float64)
print("\n[diag] input pixel_values: shape={} mean={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(
    tuple(xin.shape), xin.mean(), xin.std(), xin.min(), xin.max()))

print("\n=== ONNX FLOAT activations only (topological) — find where std -> 0 ===")
print(f"{'#':>3} {'op_type':16} {'tensor':32} {'shape':18} {'mean':>9} {'std':>9} {'%0':>6}")
collapse, seen, i, prev = None, set(), 0, None
for tname in order:
    if tname in seen or tname not in name2val:
        continue
    seen.add(tname)
    arr = np.asarray(name2val[tname])
    # Keep only real activations: float, >=2 dims, more than one element.
    if arr.dtype.kind != "f" or arr.ndim < 2 or arr.size <= 1:
        continue
    node = producer.get(tname)
    op = node.op_type if node else "?"
    sh, me, sd, mn, mx, z = stats(arr)
    dead = (sd < 1e-4 or z > 0.999)
    flag = ""
    if collapse is None and dead and op not in CONST_OPS:
        collapse = (i, op, tname, node, prev)
        flag = "   <<< FIRST DEAD ACTIVATION"
    if not dead:
        prev = (op, tname, sd)
    print(f"{i:>3} {op[:16]:16} {tname[-32:]:32} {str(sh):18} {me:9.3f} {sd:9.3f} {z*100:6.1f}{flag}")
    i += 1

print("\n" + "=" * 64)
if collapse:
    ci, op, tname, node, pv = collapse
    print(f"FIRST DEAD ACTIVATION: op #{ci} '{op}'  (tensor '{tname}')")
    if pv:
        print(f"  last HEALTHY before it: '{pv[0]}' tensor '{pv[1]}' (std={pv[2]:.3f})")
    if node is not None:
        print(f"  node name : {node.name}")
        print(f"  attrs     : {[a.name for a in node.attribute]}")
        print("  input tensor stats:")
        for inp in list(node.input):
            if inp in name2val:
                a = np.asarray(name2val[inp])
                if a.dtype.kind == 'f' and a.size > 1:
                    print(f"    {inp[-44:]:44} shape={str(tuple(a.shape)):16} std={float(a.std()):.4f}")
                else:
                    print(f"    {inp[-44:]:44} (const/shape: {a.dtype}, size {a.size})")
            else:
                print(f"    {inp[-44:]:44} (initializer / weight)")
    print("  => if this op's float inputs are HEALTHY but its output is DEAD,")
    print("     THIS op is the one that traces wrong.")
else:
    print("No dead float activation found — divergence is gradual; paste the full table.")
try:
    os.remove(tmp)
except OSError:
    pass
