/**
 * Public types for @pixagram/nsfw.
 */

/** Per-class confidence gates. An image is flagged NSFW if ANY gate trips. */
export interface Thresholds {
  porn: number; // default 0.35
  hentai: number; // default 0.35
  sexy: number; // default 0.50
  combined: number; // porn + hentai + sexy, default 0.60
}

/**
 * EfficientNet preprocessing parameters. These MUST match the exported model's
 * image processor; the build pipeline (scripts/export_model.py) emits the real
 * values into the embedded assets. The fields mirror HF EfficientNetImageProcessor.
 */
export interface PreprocessConfig {
  /** Square model input edge, e.g. 224. */
  size: number;
  /** If center-cropping: resize shorter side to this, then crop to `size`. */
  cropSize?: number;
  doCenterCrop?: boolean;
  /** Pixel rescale, e.g. 1/255. */
  rescaleFactor: number;
  /** EfficientNet: offset rescaled values by -1 (towards [-1, 1]). */
  rescaleOffset?: boolean;
  doNormalize: boolean;
  mean: [number, number, number];
  std: [number, number, number];
  /** EfficientNet image-classification quirk: divide by std a SECOND time. */
  includeTop?: boolean;
}

/** Class label -> probability (labels are lowercased). */
export type NsfwScores = Record<string, number>;

export interface NsfwResult {
  /** True if any threshold gate tripped. */
  nsfw: boolean;
  /** Softmax probabilities for every class, keyed by lowercased label. */
  scores: NsfwScores;
  /** Highest-probability class. */
  top: { label: string; score: number };
  /** Which gate(s) tripped, e.g. ["porn>=0.35"]. Useful for debugging. */
  triggers: string[];
  /** Inference time in milliseconds. */
  ms: number;
  /** Backend that ran inference, e.g. "wasm+simd". */
  backend: string;
}

/** Anything the main thread can turn into pixels. (The worker only sees pixels.) */
export type ImageSource =
  | ImageData
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | Blob
  | string; // URL (fetched then decoded)

export interface DetectorOptions {
  /** "auto" (default) uses a Worker when available, else main-thread. */
  useWorker?: "auto" | boolean;
  /** Execution provider: "auto" tries WebGPU then WASM (default). */
  backend?: "auto" | "webgpu" | "wasm";
  /** Where onnxruntime-web's own .wasm/.mjs are served from. */
  wasmPaths?: string | Record<string, string>;
  /** WASM threads. >1 requires cross-origin isolation (COOP/COEP). Default 1. */
  numThreads?: number;
  /** Override the default per-class gates. */
  thresholds?: Partial<Thresholds>;
  /** Provide your own model bytes instead of the embedded one. */
  modelBytes?: Uint8Array;
  /** Override embedded preprocessing parameters. */
  preprocess?: Partial<PreprocessConfig>;
  /** Override embedded class label order (must match the model's id2label). */
  labels?: string[];
}
