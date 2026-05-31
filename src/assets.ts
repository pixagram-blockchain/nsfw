import {
  MODEL_B64,
  LABELS as GEN_LABELS,
  PREPROCESS as GEN_PREPROCESS,
  EMBEDDED,
} from "./assets.generated.js";
import type { PreprocessConfig } from "./types.js";

/**
 * EfficientNet-style defaults (HF IMAGENET_STANDARD_MEAN/STD, include_top on).
 * These are a fallback only — the build pipeline emits the model's REAL values,
 * which override these. Do not rely on them for accuracy without verifying.
 */
export const DEFAULT_PREPROCESS: PreprocessConfig = {
  size: 224,
  doCenterCrop: false,
  rescaleFactor: 1 / 255,
  rescaleOffset: false,
  doNormalize: true,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  includeTop: true,
};

/**
 * Fallback label order. The model card lists the five classes in inconsistent
 * orders, so the REAL order MUST come from config.json id2label (emitted by the
 * build pipeline). This fallback is alphabetical and almost certainly wrong for
 * gating — replace it.
 */
export const DEFAULT_LABELS = ["drawing", "hentai", "normal", "porn", "sexy"];

export const EMBEDDED_LABELS: string[] =
  GEN_LABELS && GEN_LABELS.length ? GEN_LABELS : DEFAULT_LABELS;

export const EMBEDDED_PREPROCESS: PreprocessConfig = {
  ...DEFAULT_PREPROCESS,
  ...(GEN_PREPROCESS as Partial<PreprocessConfig>),
};

export function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

export function getModelBytes(): Uint8Array {
  if (!EMBEDDED || !MODEL_B64) {
    throw new Error(
      "@pixagram/nsfw: no model is embedded. Export + quantize your .onnx and " +
        "run `npm run embed-model` (see README), or pass options.modelBytes."
    );
  }
  return b64ToBytes(MODEL_B64);
}
