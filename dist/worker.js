// src/core.ts
import * as ort from "onnxruntime-web/webgpu";
function now() {
  return typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
}
var configured = false;
function configureRuntime(opts = {}) {
  if (configured) return;
  try {
    ort.env.wasm.numThreads = opts.numThreads ?? 1;
  } catch {
  }
  try {
    ort.env.wasm.simd = true;
  } catch {
  }
  try {
    ort.env.wasm.proxy = false;
  } catch {
  }
  if (opts.wasmPaths) {
    try {
      ort.env.wasm.wasmPaths = opts.wasmPaths;
    } catch {
    }
  }
  configured = true;
}
async function loadSession(bytes, opts = {}) {
  configureRuntime(opts);
  const pref = opts.backend ?? "auto";
  const hasGPU = typeof navigator !== "undefined" && !!navigator.gpu;
  const useGPU = pref === "webgpu" || pref === "auto" && hasGPU;
  const providers = useGPU ? ["webgpu", "wasm"] : ["wasm"];
  let session2;
  let used = useGPU ? "webgpu" : "wasm";
  try {
    session2 = await ort.InferenceSession.create(bytes, {
      executionProviders: providers,
      graphOptimizationLevel: "all"
    });
  } catch (e) {
    if (useGPU && pref !== "webgpu") {
      session2 = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
      used = "wasm";
    } else {
      throw e;
    }
  }
  const inputName = session2.inputNames[0];
  const outputName = session2.outputNames[0];
  if (!inputName || !outputName) {
    throw new Error("@pixagram/nsfw: model has no input/output names");
  }
  return { session: session2, inputName, outputName, backend: used };
}
function resizeToSquare(px, cfg2) {
  const S = cfg2.size;
  const src = new OffscreenCanvas(px.width, px.height);
  const sctx = src.getContext("2d", { willReadFrequently: true });
  if (!sctx) throw new Error("@pixagram/nsfw: 2D canvas context unavailable");
  sctx.putImageData(new ImageData(px.data, px.width, px.height), 0, 0);
  const dst = new OffscreenCanvas(S, S);
  const dctx = dst.getContext("2d", { willReadFrequently: true });
  if (!dctx) throw new Error("@pixagram/nsfw: 2D canvas context unavailable");
  if (cfg2.doCenterCrop && cfg2.cropSize) {
    const c = cfg2.cropSize;
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
function toTensorData(rgba, cfg2) {
  const S = cfg2.size;
  const area = S * S;
  const out = new Float32Array(3 * area);
  const rO = 0;
  const gO = area;
  const bO = 2 * area;
  const m = cfg2.mean;
  const s = cfg2.std;
  const rf = cfg2.rescaleFactor;
  const offset = cfg2.rescaleOffset === true;
  const normalize = cfg2.doNormalize !== false;
  const top = cfg2.includeTop === true;
  for (let p = 0, j = 0; p < area; p++, j += 4) {
    let r = rgba[j] * rf;
    let g = rgba[j + 1] * rf;
    let b = rgba[j + 2] * rf;
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
function decode(logits, labels2, t) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const exps = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }
  const scores = {};
  let topLabel = labels2[0] ?? "0";
  let topScore = -1;
  for (let i = 0; i < exps.length; i++) {
    const prob = exps[i] / sum;
    const label = (labels2[i] ?? String(i)).toLowerCase();
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
  const triggers = [];
  if (porn >= t.porn) triggers.push("porn>=" + t.porn);
  if (hentai >= t.hentai) triggers.push("hentai>=" + t.hentai);
  if (sexy >= t.sexy) triggers.push("sexy>=" + t.sexy);
  if (combined >= t.combined) triggers.push("combined>=" + t.combined);
  return { scores, top: { label: topLabel, score: topScore }, nsfw: triggers.length > 0, triggers };
}
async function classifyImageData(sess, px, cfg2, labels2, thresholds2) {
  const tStart = now();
  const rgba = resizeToSquare(px, cfg2);
  const data = toTensorData(rgba, cfg2);
  const S = cfg2.size;
  const tensor = new ort.Tensor("float32", data, [1, 3, S, S]);
  const feeds = {};
  feeds[sess.inputName] = tensor;
  const results = await sess.session.run(feeds);
  const output = results[sess.outputName];
  if (!output) throw new Error("@pixagram/nsfw: missing model output");
  const decoded = decode(output.data, labels2, thresholds2);
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

// src/worker.ts
var ctx = self;
var session = null;
var cfg = null;
var labels = [];
var thresholds = { porn: 0.35, hentai: 0.35, sexy: 0.5, combined: 0.6 };
ctx.onmessage = (event) => {
  const msg = event.data || {};
  const reqId = msg.reqId;
  void (async () => {
    try {
      if (msg.type === "init") {
        cfg = msg.cfg;
        labels = msg.labels;
        if (msg.thresholds) thresholds = msg.thresholds;
        session = await loadSession(new Uint8Array(msg.modelBuffer), msg.opts || {});
        ctx.postMessage({ reqId, ok: true, backend: "wasm" });
        return;
      }
      if (msg.type === "classify") {
        if (!session || !cfg) throw new Error("worker not initialized");
        const result = await classifyImageData(session, msg.payload, cfg, labels, thresholds);
        ctx.postMessage({ reqId, ok: true, result });
        return;
      }
    } catch (err) {
      ctx.postMessage({
        reqId,
        ok: false,
        error: String(err && err.message || err)
      });
    }
  })();
};
//# sourceMappingURL=worker.js.map