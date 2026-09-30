import fs from "node:fs/promises";
import path from "node:path";

const WEIGHT_EXTENSIONS = new Set([".gguf", ".safetensors", ".bin"]);

export async function modelTrainingSummary(source: string) {
  if (!source || !path.isAbsolute(source))
    throw new Error("Model folder must be an absolute path");
  const pending = [path.resolve(source)];
  let files = 0;
  let ggufBytes = 0;
  let tensorBytes = 0;
  while (pending.length) {
    const current = pending.pop()!;
    const entries = await fs.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isSymbolicLink()) continue;
      const child = path.join(current, entry.name);
      if (entry.isDirectory()) pending.push(child);
      else if (entry.isFile()) {
        files += 1;
        if (files > 20_000)
          throw new Error("Model folder contains too many files");
        const extension = path.extname(entry.name).toLowerCase();
        if (!WEIGHT_EXTENSIONS.has(extension)) continue;
        const size = (await fs.stat(child)).size;
        if (extension === ".gguf") ggufBytes += size;
        else tensorBytes += size;
      }
    }
  }
  const modelBytes = Math.max(ggufBytes, tensorBytes);
  if (!modelBytes)
    throw new Error("Model folder contains no supported weight files");
  return modelBytes;
}
