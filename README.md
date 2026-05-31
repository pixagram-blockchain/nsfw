# @pixagram/nsfw

On-device NSFW image classification for the browser. Runs an EfficientNet
classifier through [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)
in a **Web Worker**, with an automatic **main-thread fallback** when a worker
isn't available. The model is **bundled in the package** (base64-embedded), so
there's no separate model fetch at runtime.

Nothing leaves the device — classification happens entirely client-side.

## Install

```bash
npm install @pixagram/nsfw onnxruntime-web
```

`onnxruntime-web` is a **peer dependency** — you install it in your app so there's
a single ORT copy and you control its version.

> **Heads up:** a freshly cloned copy of this package ships with an *empty* model
> stub and will throw `no model is embedded` at runtime until you run the export +
> embed pipeline below. This keeps the published tarball reproducible from source.

## Quick start

```ts
import { classify } from "@pixagram/nsfw";

const result = await classify(myImageElement);
// {
//   nsfw: true,
//   scores: { normal: 0.02, porn: 0.91, hentai: 0.01, sexy: 0.05, drawing: 0.01 },
//   top: { label: "porn", score: 0.91 },
//   triggers: ["porn>=0.35", "combined>=0.6"],
//   ms: 23,
//   backend: "wasm+simd"
// }

if (result.nsfw) {
  // block / blur / flag
}
```

For repeated use, create one detector and reuse it (one worker, one warm session):

```ts
import { NsfwDetector } from "@pixagram/nsfw";

const detector = await NsfwDetector.create({
  // serve ORT's own wasm assets from somewhere your bundler can reach:
  wasmPaths: "/ort/",
});

console.log(detector.backend); // "worker" or "direct"

const a = await detector.classify(imageBitmap);
const b = await detector.classify(blob);
const c = await detector.classify("https://same-origin/your.jpg");

detector.dispose(); // terminates the worker / releases the session
```

`classify()` accepts `ImageData`, `ImageBitmap`, `HTMLImageElement`,
`HTMLCanvasElement`, `OffscreenCanvas`, `Blob`/`File`, or a URL string. The main
thread decodes the source to pixels; only pixels are sent to the worker.

## Options

```ts
await NsfwDetector.create({
  useWorker: "auto",        // "auto" (default) | true | false
  wasmPaths: "/ort/",       // string dir, or per-file URL map for ORT's wasm
  numThreads: 1,            // >1 needs cross-origin isolation (COOP/COEP)
  thresholds: { porn: 0.35, hentai: 0.35, sexy: 0.5, combined: 0.6 },
  modelBytes,               // Uint8Array — override the embedded model
  preprocess,               // override embedded preprocessing params
  labels,                   // override embedded class order
});
```

An image is flagged `nsfw: true` if **any** gate trips:
`porn ≥ thresholds.porn`, `hentai ≥ thresholds.hentai`, `sexy ≥ thresholds.sexy`,
or `porn + hentai + sexy ≥ thresholds.combined`. Tune these to your tolerance.

## Building the package (export → embed → build)

The repo ships code but not the model binary. Produce it once:

```bash
# 1) Export the model to ONNX, INT8-quantize it, and emit the REAL label order
#    and preprocessing params (read from the model itself, not guessed).
pip install "optimum[exporters]" onnxruntime transformers torch onnx
python scripts/export_model.py
# -> model/nsfw.int8.onnx, model/labels.json, model/preprocess.json

# 2) Base64-embed the model + bake in labels/preprocess.
npm run embed-model
# -> overwrites src/assets.generated.ts (EMBEDDED = true)

# 3) Build dual ESM/CJS + types.
npm run build
# -> dist/
```

`scripts/export_model.py` targets `viddexa/nsfw-detection-2-nano` (an
EfficientNet-b0 fine-tune). Point it at any compatible 5-class image classifier;
the label order and preprocessing are read from that model's `config.json` and
image processor, so the JS pipeline always matches the Python one.

## Why these choices

**INT8, not FP8.** FP8 tensor types are rejected by onnxruntime-web's WASM
backend at session creation — FP8 is a server-GPU/TensorRT feature. FP16 also
gives no benefit on CPU/WASM. INT8 is the lever that works in-browser: it takes
the ~16 MB FP32 model to ~4 MB and runs on the WASM backend everywhere. For best
accuracy on a CNN, switch `export_model.py` to **static** quantization with a
small calibration set (dynamic quantization mostly helps matmul ops). Validate
per-class F1 after quantizing — the `sexy` class is the most fragile.

**Worker is truly off-thread only for ESM consumers.** The worker is spawned via
`new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, which
Vite / webpack 5 / Rollup detect statically and bundle. In CJS, SSR, or any
environment without `Worker` (or where that construction throws), the detector
**automatically falls back to main-thread inference** — same API, same results,
just on the main thread. Force one mode with `useWorker: true | false`.

**Bundled model = bigger tarball.** Because the model is base64-embedded and the
package builds both ESM and CJS, the model bytes appear in *both* `dist/index.js`
and `dist/index.cjs` — roughly model-size × 2 in the published tarball. Your app
bundle only pulls in the format it imports (ESM consumers ship one copy). The
worker does **not** embed the model; it receives the bytes via a zero-copy
transfer at init. If tarball size matters more than self-containment, ship the
`.onnx` as an external asset and load it with `modelBytes` instead.

**Canvas resampling ≠ PIL.** Preprocessing resizes via `OffscreenCanvas`
(bilinear), which is not bit-identical to PIL's `NEAREST`/`BICUBIC` used during
training/eval. Predictions can drift slightly at the margins. Mitigations: keep
the embedded `preprocess.json` in sync with the model (the pipeline does this),
validate on a labeled set, and — if you need exact parity — bake the
resize+normalize into the ONNX graph so JS feeds near-raw pixels.

## Notes

- `numThreads > 1` requires your page to be cross-origin isolated
  (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`).
  Default is `1`, which always works.
- Serve ORT's `.wasm`/`.mjs` files and point `wasmPaths` at them; see the
  [onnxruntime-web docs](https://onnxruntime.ai/docs/tutorials/web/) for bundler
  setups.

## License

Apache-2.0. The bundled model is subject to its own upstream license — check the
source model card before redistributing.
