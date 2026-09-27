import { spawn } from "node:child_process";
import path from "node:path";

export type TrainingProcess = {
  pid: number;
  name: string;
  commandLine: string;
};

function run(command: string, args: string[], timeoutMs = 15_000) {
  return new Promise<string>((resolve, reject) => {
    const child = spawn(command, args, {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(stdout);
    };
    const timer = setTimeout(() => {
      child.kill();
      finish(new Error(command + " timed out"));
    }, timeoutMs);
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
      if (stdout.length > 4_000_000) {
        child.kill();
        finish(new Error(command + " returned too much process information"));
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8").slice(0, 4_000);
    });
    child.once("error", (error) => finish(error));
    child.once("close", (code) => {
      if (code !== 0)
        finish(new Error(command + " failed: " + stderr.trim().slice(-300)));
      else finish();
    });
  });
}

function normalize(value: string, platform: NodeJS.Platform) {
  return platform === "win32"
    ? value.replaceAll("/", "\\").toLowerCase()
    : value;
}

export function sessionTrainerPids(
  processes: TrainingProcess[],
  sessionDirectory: string,
  platform: NodeJS.Platform = process.platform,
) {
  const session = normalize(path.resolve(sessionDirectory), platform);
  return processes
    .filter(({ pid, name, commandLine }) => {
      if (!Number.isSafeInteger(pid) || pid < 1 || pid === process.pid)
        return false;
      const executable =
        name.replaceAll("\\", "/").split("/").at(-1)?.toLowerCase() || "";
      const trainer =
        /^llama-[a-z0-9-]+(?:\.exe)?$/.test(executable) ||
        (/^python(?:w|3(?:\.\d+)?)?(?:\.exe)?$/.test(executable) &&
          /\bmlx_(?:lm|vlm)\b/.test(commandLine));
      if (!trainer) return false;
      const command = normalize(commandLine, platform);
      const position = command.indexOf(session);
      if (position < 0) return false;
      const following = command[position + session.length] || "";
      return !following || /[\\/"'\s]/.test(following);
    })
    .map(({ pid }) => pid);
}

async function listTrainingProcesses(): Promise<TrainingProcess[]> {
  if (process.platform === "win32") {
    const output = await run("powershell.exe", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      "Get-CimInstance Win32_Process | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress",
    ]);
    const parsed = JSON.parse(output || "null") as
      Record<string, unknown> | Record<string, unknown>[] | null;
    return (Array.isArray(parsed) ? parsed : parsed ? [parsed] : []).map(
      (entry) => ({
        pid: Number(entry.ProcessId),
        name: String(entry.Name || ""),
        commandLine: String(entry.CommandLine || ""),
      }),
    );
  }
  const output = await run("ps", ["-axo", "pid=,comm=,args="]);
  return output
    .split(/\r?\n/)
    .map((line) => /^\s*(\d+)\s+(\S+)\s+(.*)$/.exec(line))
    .filter((match): match is RegExpExecArray => Boolean(match))
    .map((match) => ({
      pid: Number(match[1]),
      name: match[2],
      commandLine: match[3],
    }));
}

function isRunning(pid: number) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

export async function terminateProcessTree(pid: number) {
  if (!Number.isSafeInteger(pid) || pid < 1 || pid === process.pid)
    throw new Error("Invalid training process");
  if (!isRunning(pid)) return;
  if (process.platform === "win32") {
    await run("taskkill", ["/PID", String(pid), "/T", "/F"], 10_000).catch(
      (error) => {
        if (isRunning(pid)) throw error;
      },
    );
  } else {
    try {
      process.kill(pid, "SIGTERM");
    } catch (error) {
      if (isRunning(pid)) throw error;
    }
  }
  const deadline = Date.now() + 5_000;
  while (isRunning(pid) && Date.now() < deadline)
    await new Promise((resolve) => setTimeout(resolve, 100));
  if (isRunning(pid) && process.platform !== "win32") {
    process.kill(pid, "SIGKILL");
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  if (isRunning(pid))
    throw new Error("Training process " + pid + " did not stop");
}

export async function stopSessionTrainers(sessionDirectory: string) {
  const processes = await listTrainingProcesses();
  const pids = sessionTrainerPids(processes, sessionDirectory);
  if (pids.length > 64)
    throw new Error("Too many training processes matched this session");
  let stopped = 0;
  for (const pid of pids) {
    const current = sessionTrainerPids(
      await listTrainingProcesses(),
      sessionDirectory,
    );
    if (!current.includes(pid)) continue;
    await terminateProcessTree(pid);
    stopped += 1;
  }
  return stopped;
}
