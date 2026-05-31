/**
 * @pixagram/nsfw — on-device NSFW image classification for the browser.
 */
export { NsfwDetector } from "./detector.js";
export { classify, warmup, disposeShared } from "./oneshot.js";

export type {
  DetectorOptions,
  ImageSource,
  NsfwResult,
  NsfwScores,
  PreprocessConfig,
  Thresholds,
} from "./types.js";
