import type { OsAiBridge } from "./types.js";

declare global {
  interface Window {
    osai: OsAiBridge;
  }
}

export {};
