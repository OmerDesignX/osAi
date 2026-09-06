import { spawn } from "node:child_process";

const worker = spawn(process.execPath, [process.argv[2], process.argv[3]], {
  detached: true,
  stdio: "ignore",
});
worker.unref();
