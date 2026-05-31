/**
 * detector.ts — public NsfwDetector. Picks a Web Worker when available and
 * falls back to main-thread inference otherwise (and on any worker-spawn error).
 */
import * as core from "./core.js";
import { getModelBytes, EMBEDDED_LABELS, EMBEDDED_PREPROCESS } from "./assets.js";
import type {
  DetectorOptions,
  ImageSource,
  NsfwResult,
  PreprocessConfig,
  Thresholds,
} from "./types.js";

const DEFAULT_THRESHOLDS: Thresholds = {
  porn: 0.35,
  hentai: 0.35,
  sexy: 0.5,
  combined: 0.6,
};

// --- Image decoding (main thread only) -------------------------------------

function makeCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function isImageData(x: unknown): x is ImageData {
  return typeof ImageData !== "undefined" && x instanceof ImageData;
}

async function toPixelData(source: ImageSource): Promise<core.PixelData> {
  if (isImageData(source)) {
    // Copy: the worker path transfers (detaches) the buffer, and we must not
    // detach a caller-owned ImageData.
    return {
      data: new Uint8ClampedArray(source.data),
      width: source.width,
      height: source.height,
    };
  }

  let bitmap: ImageBitmap | null = null;
  let drawable: CanvasImageSource;

  if (typeof source === "string") {
    const res = await fetch(source);
    bitmap = await createImageBitmap(await res.blob());
    drawable = bitmap;
  } else if (typeof Blob !== "undefined" && source instanceof Blob) {
    bitmap = await createImageBitmap(source);
    drawable = bitmap;
  } else {
    drawable = source as CanvasImageSource;
  }

  const anyDrawable = drawable as unknown as {
    width?: number;
    height?: number;
    videoWidth?: number;
    videoHeight?: number;
  };
  const w = anyDrawable.width ?? anyDrawable.videoWidth ?? 0;
  const h = anyDrawable.height ?? anyDrawable.videoHeight ?? 0;
  if (!w || !h) throw new Error("@pixagram/nsfw: could not determine image dimensions");

  const canvas = makeCanvas(w, h);
  const ctx = (canvas as OffscreenCanvas).getContext("2d", {
    willReadFrequently: true,
  }) as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
  if (!ctx) throw new Error("@pixagram/nsfw: 2D canvas context unavailable");

  ctx.drawImage(drawable, 0, 0);
  const id = ctx.getImageData(0, 0, w, h);
  if (bitmap) bitmap.close();
  return { data: id.data, width: w, height: h };
}

// --- Implementations --------------------------------------------------------

interface Impl {
  classify(source: ImageSource): Promise<NsfwResult>;
  dispose(): void;
}

class DirectImpl implements Impl {
  private constructor(
    private sess: core.Session,
    private cfg: PreprocessConfig,
    private labels: string[],
    private thresholds: Thresholds
  ) {}

  static async create(
    bytes: Uint8Array,
    cfg: PreprocessConfig,
    labels: string[],
    thresholds: Thresholds,
    opts: DetectorOptions
  ): Promise<DirectImpl> {
    const sess = await core.loadSession(bytes, {
      wasmPaths: opts.wasmPaths,
      numThreads: opts.numThreads,
    });
    return new DirectImpl(sess, cfg, labels, thresholds);
  }

  async classify(source: ImageSource): Promise<NsfwResult> {
    const px = await toPixelData(source);
    return core.classifyImageData(this.sess, px, this.cfg, this.labels, this.thresholds);
  }

  dispose(): void {
    try {
      (this.sess.session as unknown as { release?: () => void }).release?.();
    } catch {
      /* ignore */
    }
  }
}

type Pending = { resolve: (v: NsfwResult) => void; reject: (e: Error) => void };

class WorkerImpl implements Impl {
  private seq = 0;
  private pending = new Map<number, Pending>();

  private constructor(private worker: Worker) {
    this.worker.onmessage = (e: MessageEvent) => {
      const { reqId, ok, result, error } = e.data || {};
      const p = this.pending.get(reqId);
      if (!p) return;
      this.pending.delete(reqId);
      if (ok) p.resolve(result as NsfwResult);
      else p.reject(new Error(error || "worker error"));
    };
    this.worker.onerror = (e: ErrorEvent) => {
      const err = new Error(e.message || "worker crashed");
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
  }

  static async create(
    bytes: Uint8Array,
    cfg: PreprocessConfig,
    labels: string[],
    thresholds: Thresholds,
    opts: DetectorOptions
  ): Promise<WorkerImpl> {
    // Statically detectable by Vite / webpack 5 / Rollup so the worker chunk
    // is bundled and its URL rewritten. Throws in CJS / no-Worker contexts,
    // which NsfwDetector.create() catches to fall back to main-thread.
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const impl = new WorkerImpl(worker);
    // Transfer the model buffer (zero-copy). The main thread no longer needs it.
    const buf = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength
    ) as ArrayBuffer;
    await impl.rpc(
      "init",
      {
        modelBuffer: buf,
        cfg,
        labels,
        thresholds,
        opts: { wasmPaths: opts.wasmPaths, numThreads: opts.numThreads },
      },
      [buf]
    );
    return impl;
  }

  private rpc(type: string, data: Record<string, unknown>, transfer: Transferable[] = []): Promise<NsfwResult> {
    return new Promise<NsfwResult>((resolve, reject) => {
      const reqId = ++this.seq;
      this.pending.set(reqId, { resolve, reject });
      this.worker.postMessage({ type, reqId, ...data }, transfer);
    });
  }

  async classify(source: ImageSource): Promise<NsfwResult> {
    const px = await toPixelData(source);
    // Transfer the pixel buffer (zero-copy); a fresh one is decoded each call.
    return this.rpc("classify", { payload: px }, [px.data.buffer as ArrayBuffer]);
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}

function canUseWorker(): boolean {
  try {
    return typeof Worker !== "undefined";
  } catch {
    return false;
  }
}

// --- Public class -----------------------------------------------------------

export class NsfwDetector {
  private constructor(private impl: Impl, public readonly backend: "worker" | "direct") {}

  static async create(opts: DetectorOptions = {}): Promise<NsfwDetector> {
    const cfg: PreprocessConfig = { ...EMBEDDED_PREPROCESS, ...(opts.preprocess || {}) };
    const labels = opts.labels ?? EMBEDDED_LABELS;
    const thresholds: Thresholds = { ...DEFAULT_THRESHOLDS, ...(opts.thresholds || {}) };
    const bytes = opts.modelBytes ?? getModelBytes();

    const mode = opts.useWorker ?? "auto";
    const wantWorker = mode === true || (mode === "auto" && canUseWorker());

    if (wantWorker) {
      try {
        const impl = await WorkerImpl.create(bytes, cfg, labels, thresholds, opts);
        return new NsfwDetector(impl, "worker");
      } catch {
        if (opts.useWorker === true) {
          throw new Error("@pixagram/nsfw: worker requested but unavailable in this environment");
        }
        // auto: fall through to main-thread
      }
    }

    const impl = await DirectImpl.create(bytes, cfg, labels, thresholds, opts);
    return new NsfwDetector(impl, "direct");
  }

  /** Classify one image. Accepts ImageData, ImageBitmap, <img>/<canvas>, Blob, or a URL. */
  classify(source: ImageSource): Promise<NsfwResult> {
    return this.impl.classify(source);
  }

  /** Release the worker / ORT session. */
  dispose(): void {
    this.impl.dispose();
  }
}
