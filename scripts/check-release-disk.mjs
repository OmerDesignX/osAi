import { statfs } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const minimumGiB = Number(process.env.OSAI_RELEASE_MIN_FREE_GIB || "4");

if (!Number.isFinite(minimumGiB) || minimumGiB < 2)
  throw new Error("OSAI_RELEASE_MIN_FREE_GIB must be at least 2");

const disk = await statfs(root, { bigint: true });
const freeBytes = disk.bavail * disk.bsize;
const freeGiB = Number(freeBytes / 1024n / 1024n / 1024n);

if (freeBytes < BigInt(Math.ceil(minimumGiB * 1024 ** 3)))
  throw new Error(
    "Native releases require " +
      minimumGiB +
      " GiB free; this runner has about " +
      freeGiB +
      " GiB",
  );

console.log("Release disk check passed: about " + freeGiB + " GiB free");
