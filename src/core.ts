/**
 * core.ts — runtime-agnostic inference helpers shared by the Web Worker and the
 * main-thread fallback. No DOM beyond OffscreenCanvas (available in both).
 */
// The /webgpu bundle includes BOTH the WebGPU and WASM execution providers,
// so we can prefer GPU and fall back to CPU from a single import.
import * as ort from "onnxruntime-web/webgpu";
import type { PreprocessConfig, Thresholds, NsfwResult } from "./types.js";

function now(): number {
  return typeof performance !== "undefined" && performance.now
    ? performance.now()
    : Date.now();
}

/** Raw pixels the main thread hands to inference. */
export interface PixelData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface Session {
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
  backend: string; // the execution provider that actually started, e.g. "webgpu"
}

export interface RuntimeOptions {
  wasmPaths?: string | Record<string, string>;
  numThreads?: number;
  /** "auto" tries WebGPU then WASM (default). Force one with "webgpu"/"wasm". */
  backend?: "auto" | "webgpu" | "wasm";
}

let configured = false;
export function configureRuntime(opts: RuntimeOptions = {}): void {
  if (configured) return;
  // >1 thread requires cross-origin isolation (COOP/COEP); 1 is always safe.
  try {
    ort.env.wasm.numThreads = opts.numThreads ?? 1;
  } catch {
    /* ignore */
  }
  try {
    ort.env.wasm.simd = true;
  } catch {
    /* ignore */
  }
  // We run inside a worker already (or accept main-thread cost): no proxy worker.
  try {
    (ort.env.wasm as { proxy?: boolean }).proxy = false;
  } catch {
    /* ignore */
  }
  if (opts.wasmPaths) {
    try {
      ort.env.wasm.wasmPaths = opts.wasmPaths as never;
    } catch {
      /* ignore */
    }
  }
  configured = true;
}

export async function loadSession(
  bytes: Uint8Array,
  opts: RuntimeOptions = {}
): Promise<Session> {
  configureRuntime(opts);

  const pref = opts.backend ?? "auto";
  const hasGPU =
    typeof navigator !== "undefined" &&
    !!(navigator as unknown as { gpu?: unknown }).gpu;

  // For the GPU path, pass ["webgpu","wasm"] as ONE provider list. ORT runs each
  // op on the WebGPU EP when it has a kernel and falls back to CPU per-op
  // otherwise — within a single session. This is what lets an INT8/QDQ model run
  // on WebGPU at all: listing "webgpu" alone would fail on any unsupported
  // quantize/dequantize op and drop the whole model to CPU. Pure "wasm" is used
  // when the GPU is unwanted or the WebGPU API isn't present.
  const useGPU = pref === "webgpu" || (pref === "auto" && hasGPU);
  const providers = useGPU ? ["webgpu", "wasm"] : ["wasm"];

  let session: ort.InferenceSession;
  let used = useGPU ? "webgpu" : "wasm";
  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    });
  } catch (e) {
    // WebGPU couldn't initialise at all (no adapter, etc.). If we weren't forced
    // onto it, retry pure CPU; if the caller forced "webgpu", surface the error.
    if (useGPU && pref !== "webgpu") {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      used = "wasm";
    } else {
      throw e;
    }
  }

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName || !outputName) {
    throw new Error("@pixagram/nsfw: model has no input/output names");
  }
  // `used` is the PRIMARY provider; with the GPU path some ops may still run on
  // CPU via fallback. The measured ms is the real signal of how well it landed.
  return { session, inputName, outputName, backend: used };
}

/**
 * Resize source pixels to the model's square input via OffscreenCanvas.
 * NOTE: canvas resampling (bilinear) is NOT identical to PIL's NEAREST/BICUBIC,
 * so predictions can drift slightly vs. the Python pipeline. Validate on a test
 * set; if you need pixel-exact parity, bake preprocessing into the ONNX graph.
 */
function resizeToSquare(px: PixelData, cfg: PreprocessConfig): Uint8ClampedArray {
  const S = cfg.size;

  const src = new OffscreenCanvas(px.width, px.height);
  const sctx = src.getContext("2d", { willReadFrequently: true });
  if (!sctx) throw new Error("@pixagram/nsfw: 2D canvas context unavailable");
  // Cast to any: newer TS types Uint8ClampedArray as generic over ArrayBufferLike
  // (which includes SharedArrayBuffer) and rejects it for ImageData's argument,
  // though at runtime our buffer is always a plain ArrayBuffer. Type-only, and
  // `any` keeps this compiling across TS 5.4–5.7+ (the generic form is 5.7-only).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sctx.putImageData(new ImageData(px.data as any, px.width, px.height), 0, 0);

  const dst = new OffscreenCanvas(S, S);
  const dctx = dst.getContext("2d", { willReadFrequently: true });
  if (!dctx) throw new Error("@pixagram/nsfw: 2D canvas context unavailable");

  if (cfg.doCenterCrop && cfg.cropSize) {
    const c = cfg.cropSize;
    const scale = c / Math.min(px.width, px.height);
    const rw = Math.round(px.width * scale);
    const rh = Math.round(px.height * scale);
    const tmp = new OffscreenCanvas(rw, rh);
    const tctx = tmp.getContext("2d", { willReadFrequently: true });
    if (!tctx) throw new Error("@pixagram/nsfw: 2D canvas context unavailable");
    tctx.drawImage(src, 0, 0, px.width, px.height, 0, 0, rw, rh);
    const cx = Math.floor((rw - S) / 2);
    const cy = Math.floor((rh - S) / 2);
    dctx.drawImage(tmp, cx, cy, S, S, 0, 0, S, S);
  } else {
    dctx.drawImage(src, 0, 0, px.width, px.height, 0, 0, S, S);
  }

  return dctx.getImageData(0, 0, S, S).data;
}

/** RGBA bytes -> NCHW float32, applying rescale / offset / normalize / include_top. */
function toTensorData(rgba: Uint8ClampedArray, cfg: PreprocessConfig): Float32Array {
  const S = cfg.size;
  const area = S * S;
  const out = new Float32Array(3 * area);
  const rO = 0;
  const gO = area;
  const bO = 2 * area;

  const m = cfg.mean;
  const s = cfg.std;
  const rf = cfg.rescaleFactor;
  const offset = cfg.rescaleOffset === true;
  const normalize = cfg.doNormalize !== false;
  const top = cfg.includeTop === true;

  for (let p = 0, j = 0; p < area; p++, j += 4) {
    let r = rgba[j]! * rf;
    let g = rgba[j + 1]! * rf;
    let b = rgba[j + 2]! * rf;
    if (offset) {
      r -= 1;
      g -= 1;
      b -= 1;
    }
    if (normalize) {
      r = (r - m[0]) / s[0];
      g = (g - m[1]) / s[1];
      b = (b - m[2]) / s[2];
      if (top) {
        // EfficientNet image-classification: normalize again by std only.
        r /= s[0];
        g /= s[1];
        b /= s[2];
      }
    }
    out[rO + p] = r;
    out[gO + p] = g;
    out[bO + p] = b;
  }
  return out;
}

/** Softmax logits -> per-class scores, then apply the NSFW gates. */
export function decode(
  logits: Float32Array | number[],
  labels: string[],
  t: Thresholds
): Pick<NsfwResult, "scores" | "top" | "nsfw" | "triggers"> {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;

  let sum = 0;
  const exps = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i]! - max);
    exps[i] = e;
    sum += e;
  }

  const scores: Record<string, number> = {};
  let topLabel = labels[0] ?? "0";
  let topScore = -1;
  for (let i = 0; i < exps.length; i++) {
    const prob = exps[i]! / sum;
    const label = (labels[i] ?? String(i)).toLowerCase();
    scores[label] = prob;
    if (prob > topScore) {
      topScore = prob;
      topLabel = label;
    }
  }

  const porn = scores["porn"] ?? 0;
  const hentai = scores["hentai"] ?? 0;
  const sexy = scores["sexy"] ?? 0;
  const combined = porn + hentai + sexy;

  const triggers: string[] = [];
  if (porn >= t.porn) triggers.push("porn>=" + t.porn);
  if (hentai >= t.hentai) triggers.push("hentai>=" + t.hentai);
  if (sexy >= t.sexy) triggers.push("sexy>=" + t.sexy);
  if (combined >= t.combined) triggers.push("combined>=" + t.combined);

  return { scores, top: { label: topLabel, score: topScore }, nsfw: triggers.length > 0, triggers };
}

/** Full pipeline for one image: preprocess -> run -> decode. */
export async function classifyImageData(
  sess: Session,
  px: PixelData,
  cfg: PreprocessConfig,
  labels: string[],
  thresholds: Thresholds
): Promise<NsfwResult> {
  const tStart = now();

  const rgba = resizeToSquare(px, cfg);
  const data = toTensorData(rgba, cfg);
  const S = cfg.size;
  const tensor = new ort.Tensor("float32", data, [1, 3, S, S]);

  const feeds: Record<string, ort.Tensor> = {};
  feeds[sess.inputName] = tensor;
  const results = await sess.session.run(feeds);
  const output = results[sess.outputName];
  if (!output) throw new Error("@pixagram/nsfw: missing model output");

  const decoded = decode(output.data as Float32Array, labels, thresholds);
  const simd = (() => {
    try {
      return ort.env.wasm.simd ? "+simd" : "";
    } catch {
      return "";
    }
  })();
  const backend = sess.backend === "wasm" ? "wasm" + simd : sess.backend;

  return { ...decoded, ms: Math.round(now() - tStart), backend };
}
