import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type {
  AutoBenchmarkResult,
  ClusterPeer,
  Preferences,
  SessionState,
  TrainingRequest,
} from "../types.js";
import { runAutoBenchmark } from "./auto-benchmark.js";
import { backendRuntimeEnvironment } from "./backend-source.js";
import { ClusterService } from "./cluster-service.js";
import { inspectDataset, saveDataset } from "./dataset-editor.js";
import { hardwareSnapshot } from "./hardware-snapshot.js";
import { SessionService } from "./session-service.js";

const activeStatuses = new Set([
  "queued",
  "running",
  "pausing",
  "paused",
  "stopping",
]);
const datasetNames = new Set(["train.jsonl", "valid.jsonl", "test.jsonl"]);

type EncodedFile = { name: string; data: string };
type NodeJob = { sessionId: string; recordCount: number; directory: string };
type NodeStartPayload = {
  jobId: string;
  request: TrainingRequest;
  files: EncodedFile[];
  recordCount: number;
};

type WorkerPlan = {
  id: string;
  peer: ClusterPeer | null;
  name: string;
  benchmark: AutoBenchmarkResult;
  recordCount: number;
  dataset: string;
  request: TrainingRequest;
  remoteSessionId?: string;
};

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The cluster request is malformed");
  return value as Record<string, unknown>;
}

function safeJobId(value: unknown) {
  if (typeof value !== "string" || !/^[a-f\d-]{20,80}$/i.test(value))
    throw new Error("The cluster job ID is invalid");
  return value;
}

function sha256(value: Buffer) {
  return createHash("sha256").update(value).digest("hex");
}

function capacityWeight(peer: ClusterPeer | null, localGpuMemory = 0) {
  const gpu = peer
    ? peer.gpuMemoryBytes.reduce((sum, value) => sum + value, 0)
    : localGpuMemory;
  const memory = peer?.memoryBytes ?? os.totalmem();
  return Math.max(1, gpu || Math.floor(memory / 2));
}

export function allocateRecords(total: number, weights: number[]) {
  if (!Number.isSafeInteger(total) || total < weights.length || !weights.length)
    throw new Error(
      "The dataset needs at least one row for every cluster machine",
    );
  if (weights.some((value) => !Number.isFinite(value) || value <= 0))
    throw new Error("Cluster capacity weights must be positive");
  const result = weights.map(() => 1);
  const remaining = total - weights.length;
  const sum = weights.reduce((left, right) => left + right, 0);
  const exact = weights.map((weight) => (remaining * weight) / sum);
  for (let index = 0; index < weights.length; index += 1)
    result[index] += Math.floor(exact[index]);
  let unassigned = total - result.reduce((left, right) => left + right, 0);
  for (const index of exact
    .map((value, index) => ({ index, remainder: value - Math.floor(value) }))
    .sort((left, right) => right.remainder - left.remainder)
    .map((item) => item.index)) {
    if (unassigned-- <= 0) break;
    result[index] += 1;
  }
  return result;
}

function compatibleRequest(
  input: TrainingRequest,
  benchmark: AutoBenchmarkResult,
  common: {
    rank: number;
    numLayers: number;
    targetModules: TrainingRequest["targetModules"];
  },
  dataset: string,
) {
  return {
    ...input,
    sessionsRoot: input.sessionsRoot,
    modelSource: "official" as const,
    customModelFolder: "",
    engine: "llama.cpp" as const,
    accelerator: benchmark.accelerator,
    stage: "fine-tuning" as const,
    fineTuneData: dataset,
    alignmentData: "",
    reuseDataset: false,
    adapter: "",
    autoSettings: true,
    calibrationApplied: true,
    fullContentContext: false,
    multiGpu:
      benchmark.devices.length > 1 ? ("on" as const) : ("auto" as const),
    batchSize: benchmark.settings.batch_size,
    maxSeqLength: benchmark.settings.max_seq_length,
    rank: common.rank,
    numLayers: common.numLayers,
    targetModules: common.targetModules,
    scale: input.scale ?? 8,
    ggufBatchSize: benchmark.settings.gguf_batch_size,
    ggufThreads: benchmark.settings.gguf_threads,
    devices: benchmark.devices.join(", "),
    tensorSplit: benchmark.tensor_split.join(", "),
    mainGpu: benchmark.main_gpu,
    distributedWorkers: 0,
  } satisfies TrainingRequest;
}

async function privateJson(file: string, value: unknown) {
  const temporary = `${file}.${process.pid}.${randomUUID()}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    mode: 0o600,
  });
  await fs.rename(temporary, file);
}

async function updateSession(
  sessionDirectory: string,
  update: Partial<SessionState>,
) {
  const statePath = path.join(sessionDirectory, "state.json");
  const state = JSON.parse(
    await fs.readFile(statePath, "utf8"),
  ) as SessionState;
  const next = { ...state, ...update };
  await privateJson(statePath, next);
  return next;
}

async function adapterPath(sessionDirectory: string) {
  const expected = path.join(
    sessionDirectory,
    ".internal",
    "base-plus-adapter",
    "adapters",
    "gguf",
    "adapter.gguf",
  );
  if ((await fs.stat(expected).catch(() => null))?.isFile()) return expected;
  throw new Error("The cluster worker produced no compatible GGUF adapter");
}

async function command(
  executable: string,
  args: string[],
  outputLimit = 64 * 1024,
) {
  const env = await backendRuntimeEnvironment(executable);
  return new Promise<string>((resolve, reject) => {
    const child = spawn(executable, args, {
      env,
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    const append = (chunk: Buffer) => {
      output = (output + chunk.toString("utf8")).slice(-outputLimit);
    };
    child.stdout?.on("data", append);
    child.stderr?.on("data", append);
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve(output);
      else
        reject(
          new Error(
            output.trim().split(/\r?\n/).at(-1) ||
              `Cluster aggregation exited with ${code}`,
          ),
        );
    });
  });
}

export class ClusterTrainingCoordinator {
  private nodeJobs = new Map<string, NodeJob>();
  private masterActive = false;

  constructor(
    private readonly cluster: ClusterService,
    private readonly sessions: SessionService,
    private readonly storageRoot: string,
    private readonly preferences: () => Promise<Preferences>,
    private readonly backendIsCurrent: (executable: string) => Promise<boolean>,
  ) {
    cluster.setRequestHandler((action, payload) =>
      this.handleNodeRequest(action, payload),
    );
  }

  async cancelNodeJobs() {
    await Promise.allSettled(
      [...this.nodeJobs.values()].map((job) =>
        this.sessions.stop(job.sessionId),
      ),
    );
    this.nodeJobs.clear();
  }

  async start(input: TrainingRequest, executable: string) {
    if (this.masterActive)
      throw new Error("A cluster session is already running");
    const status = this.cluster.getStatus();
    if (status.mode !== "master" || !status.ready)
      throw new Error("Cluster master mode is not ready");
    if (status.peers.some((peer) => peer.role === "master"))
      throw new Error("Turn off the other cluster master before starting");
    const peers = status.peers.filter((peer) => peer.role === "node");
    if (!peers.length)
      throw new Error(
        "No authenticated cluster nodes are connected on this LAN",
      );
    if (input.modelSource !== "official")
      throw new Error("Cluster training currently requires an osCode model");
    if (input.stage !== "fine-tuning")
      throw new Error(
        "Cluster training currently supports fine-tuning sessions",
      );

    this.masterActive = true;
    const jobId = randomUUID();
    const staging = path.join(this.storageRoot, "master", jobId);
    const normalized = path.join(staging, "normalized");
    const shardsRoot = path.join(staging, "shards");
    const startedPeers: Array<{ peer: ClusterPeer; jobId: string }> = [];
    try {
      await fs.mkdir(staging, { recursive: true, mode: 0o700 });
      const inspection = await inspectDataset(input.fineTuneData);
      if (inspection.modalities.length)
        throw new Error(
          "Cluster transfer currently supports text datasets; multimodal files must remain on one machine",
        );
      await saveDataset({
        source: input.fineTuneData,
        outputDirectory: normalized,
        task: inspection.recommendedTask,
        mapping: inspection.mapping,
        trimWhitespace: true,
        removeDuplicates: true,
        skipInvalid: true,
        validationPercent: 0,
        testPercent: 0,
        edits: [],
      });
      const trainLines = (
        await fs.readFile(path.join(normalized, "train.jsonl"), "utf8")
      )
        .split(/\r?\n/)
        .filter(Boolean);
      if (trainLines.length < peers.length + 1)
        throw new Error(
          "The dataset needs at least one valid row per cluster machine",
        );

      const baseRequest = {
        ...input,
        engine: "llama.cpp" as const,
        accelerator: "auto" as const,
        multiGpu: "auto" as const,
        devices: "",
      };
      const [localBenchmark, ...remoteBenchmarks] = await Promise.all([
        runAutoBenchmark(executable, baseRequest),
        ...peers.map(
          (peer) =>
            this.cluster.request(
              peer,
              "benchmark",
              baseRequest,
            ) as Promise<AutoBenchmarkResult>,
        ),
      ]);
      const benchmarks = [localBenchmark, ...remoteBenchmarks];
      const commonTargets = localBenchmark.settings.target_modules.filter(
        (target) =>
          remoteBenchmarks.every((benchmark) =>
            benchmark.settings.target_modules.includes(target),
          ),
      );
      if (!commonTargets.length)
        throw new Error(
          "The cluster machines have no compatible LoRA projections",
        );
      const common = {
        rank: Math.min(...benchmarks.map((item) => item.settings.rank)),
        numLayers: Math.min(
          ...benchmarks.map((item) => item.settings.num_layers),
        ),
        targetModules: commonTargets,
      };
      const localSnapshot = await hardwareSnapshot();
      const localGpuMemory = localSnapshot.gpus.reduce(
        (sum, gpu) => sum + (gpu.memoryTotalBytes ?? 0),
        0,
      );
      const counts = allocateRecords(trainLines.length, [
        capacityWeight(null, localGpuMemory),
        ...peers.map((peer) => capacityWeight(peer)),
      ]);
      await fs.mkdir(shardsRoot, { recursive: true, mode: 0o700 });
      let offset = 0;
      const plans: WorkerPlan[] = [];
      for (let index = 0; index < counts.length; index += 1) {
        const peer = index === 0 ? null : peers[index - 1];
        const directory = path.join(shardsRoot, String(index));
        await fs.mkdir(directory, { recursive: true, mode: 0o700 });
        const count = counts[index];
        await fs.writeFile(
          path.join(directory, "train.jsonl"),
          `${trainLines.slice(offset, offset + count).join("\n")}\n`,
          { mode: 0o600 },
        );
        offset += count;
        for (const name of ["valid.jsonl", "test.jsonl"]) {
          const source = path.join(normalized, name);
          if ((await fs.stat(source).catch(() => null))?.isFile())
            await fs.copyFile(source, path.join(directory, name));
        }
        plans.push({
          id: peer?.id ?? status.selfId,
          peer,
          name: peer?.name ?? (await this.preferences()).clusterName,
          benchmark: benchmarks[index],
          recordCount: count,
          dataset: directory,
          request: compatibleRequest(
            input,
            benchmarks[index],
            common,
            directory,
          ),
        });
      }

      for (const plan of plans.slice(1)) {
        const nodeJobId = randomUUID();
        const files = await this.encodeDataset(plan.dataset);
        const response = (await this.cluster.request(plan.peer!, "start-job", {
          jobId: nodeJobId,
          request: plan.request,
          files,
          recordCount: plan.recordCount,
        } satisfies NodeStartPayload)) as { sessionId: string };
        if (!response || typeof response.sessionId !== "string")
          throw new Error(`${plan.name} returned an invalid session`);
        plan.remoteSessionId = response.sessionId;
        startedPeers.push({ peer: plan.peer!, jobId: nodeJobId });
      }

      const local = await this.sessions.start(plans[0].request);
      await fs.writeFile(
        path.join(local.sessionDirectory, ".cluster-controller"),
        `${process.pid}\n`,
        { mode: 0o600 },
      );
      void this.monitor(
        executable,
        local,
        plans,
        startedPeers,
        staging,
      ).finally(() => {
        this.masterActive = false;
      });
      return {
        ...local,
        message: `Cluster training started on ${plans.length} machines`,
      };
    } catch (error) {
      await Promise.allSettled(
        startedPeers.map(({ peer, jobId }) =>
          this.cluster.request(peer, "cancel-job", { jobId }),
        ),
      );
      await fs.rm(staging, { recursive: true, force: true });
      this.masterActive = false;
      throw error;
    }
  }

  private async handleNodeRequest(action: string, payload: unknown) {
    const preferences = await this.preferences();
    if (preferences.clusterMode !== "node")
      throw new Error("This machine is not accepting cluster work");
    const backend = await this.sessions.backendStatus();
    if (
      !backend.available ||
      !(await this.backendIsCurrent(backend.executable))
    )
      throw new Error("Install the current osAi CLI on this node first");
    if (action === "benchmark") {
      const input = object(payload) as unknown as TrainingRequest;
      if (input.modelSource !== "official")
        throw new Error("Cluster nodes accept only official osCode models");
      return runAutoBenchmark(backend.executable, {
        ...input,
        engine: "llama.cpp",
        accelerator: "auto",
        multiGpu: "auto",
        devices: "",
      });
    }
    const value = object(payload);
    const jobId = safeJobId(value.jobId);
    if (action === "start-job") {
      for (const [existingId, existing] of this.nodeJobs) {
        const state = await this.sessions
          .find(existing.sessionId)
          .catch(() => null);
        if (state && activeStatuses.has(state.status))
          throw new Error("This node already has a cluster job");
        this.nodeJobs.delete(existingId);
      }
      const request = object(value.request) as unknown as TrainingRequest;
      if (
        request.modelSource !== "official" ||
        request.stage !== "fine-tuning" ||
        request.engine !== "llama.cpp"
      )
        throw new Error("The master requested an unsupported cluster job");
      if (!Array.isArray(value.files) || value.files.length > 3)
        throw new Error("The cluster dataset is invalid");
      const recordCount = Number(value.recordCount);
      if (!Number.isSafeInteger(recordCount) || recordCount < 1)
        throw new Error("The cluster record count is invalid");
      const directory = path.join(this.storageRoot, "node", jobId);
      const dataset = path.join(directory, "dataset");
      await fs.mkdir(dataset, { recursive: true, mode: 0o700 });
      let bytes = 0;
      for (const item of value.files as EncodedFile[]) {
        if (
          !item ||
          !datasetNames.has(item.name) ||
          typeof item.data !== "string" ||
          !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(
            item.data,
          )
        )
          throw new Error("The cluster dataset contains an unsafe file");
        const content = Buffer.from(item.data, "base64");
        bytes += content.byteLength;
        if (bytes > 240 * 1024 * 1024)
          throw new Error(
            "The cluster dataset exceeds this node's transfer limit",
          );
        await fs.writeFile(path.join(dataset, item.name), content, {
          flag: "wx",
          mode: 0o600,
        });
      }
      if (!(await fs.stat(path.join(dataset, "train.jsonl")).catch(() => null)))
        throw new Error("The cluster dataset has no training split");
      const session = await this.sessions.start({
        ...request,
        sessionsRoot: preferences.sessionsRoot,
        fineTuneData: dataset,
        sessionName: `${request.sessionName}-node`,
      });
      this.nodeJobs.set(jobId, {
        sessionId: session.id,
        recordCount,
        directory,
      });
      return { sessionId: session.id };
    }
    const job = this.nodeJobs.get(jobId);
    if (!job) throw new Error("The cluster job is no longer available");
    if (action === "job-status") {
      const state = await this.sessions.find(job.sessionId);
      return {
        id: state.id,
        status: state.status,
        phase: state.phase,
        progress: state.progress,
        message: state.message,
        error: state.error,
      };
    }
    if (action === "cancel-job") {
      const state = await this.sessions.stop(job.sessionId);
      return { status: state.status };
    }
    if (action === "job-artifact") {
      const state = await this.sessions.find(job.sessionId);
      if (state.status !== "completed")
        throw new Error("The cluster worker has not completed training");
      const adapter = await fs.readFile(
        await adapterPath(state.sessionDirectory),
      );
      return {
        recordCount: job.recordCount,
        sha256: sha256(adapter),
        data: adapter.toString("base64"),
      };
    }
    throw new Error("Unknown cluster action");
  }

  private async encodeDataset(directory: string) {
    const files: EncodedFile[] = [];
    for (const name of datasetNames) {
      const file = path.join(directory, name);
      if ((await fs.stat(file).catch(() => null))?.isFile())
        files.push({
          name,
          data: (await fs.readFile(file)).toString("base64"),
        });
    }
    return files;
  }

  private async monitor(
    executable: string,
    local: SessionState,
    plans: WorkerPlan[],
    remoteJobs: Array<{ peer: ClusterPeer; jobId: string }>,
    staging: string,
  ) {
    try {
      while (true) {
        const localState = await this.sessions.find(local.id);
        if (["stopping", "stopped"].includes(localState.status)) {
          await Promise.allSettled(
            remoteJobs.map(({ peer, jobId }) =>
              this.cluster.request(peer, "cancel-job", { jobId }),
            ),
          );
          return;
        }
        const remoteStates = await Promise.all(
          remoteJobs.map(({ peer, jobId }) =>
            this.cluster.request(peer, "job-status", { jobId }),
          ),
        );
        const failed = remoteStates.find(
          (state) => object(state).status === "failed",
        );
        if (localState.status === "failed" || failed) {
          const error =
            localState.error ||
            (failed
              ? String(object(failed).error || object(failed).message)
              : "") ||
            "A cluster worker failed";
          throw new Error(error);
        }
        const complete =
          localState.status === "completed" &&
          remoteStates.every((state) => object(state).status === "completed");
        if (complete) break;
        if (localState.status === "completed")
          await updateSession(local.sessionDirectory, {
            status: "running",
            phase: "publishing",
            progress: 99,
            indeterminate: true,
            message: "Waiting for the remaining cluster machines",
            endedAt: undefined,
          });
        await new Promise((resolve) => setTimeout(resolve, 2_000));
      }

      await updateSession(local.sessionDirectory, {
        status: "running",
        phase: "publishing",
        progress: 99,
        indeterminate: true,
        message: "Securely collecting and aggregating cluster adapters",
        endedAt: undefined,
      });
      const adaptersRoot = path.join(
        local.sessionDirectory,
        ".cluster",
        "adapters",
      );
      await fs.mkdir(adaptersRoot, { recursive: true, mode: 0o700 });
      const adapters = [await adapterPath(local.sessionDirectory)];
      const records = [plans[0].recordCount];
      for (let index = 0; index < remoteJobs.length; index += 1) {
        const { peer, jobId } = remoteJobs[index];
        const artifact = object(
          await this.cluster.request(peer, "job-artifact", { jobId }),
        );
        if (
          typeof artifact.data !== "string" ||
          typeof artifact.sha256 !== "string" ||
          !Number.isSafeInteger(artifact.recordCount) ||
          artifact.recordCount !== plans[index + 1].recordCount
        )
          throw new Error(`${peer.name} returned an invalid adapter`);
        const content = Buffer.from(artifact.data, "base64");
        if (sha256(content) !== artifact.sha256)
          throw new Error(`${peer.name}'s adapter failed its checksum`);
        const file = path.join(adaptersRoot, `${index + 1}-${peer.id}.gguf`);
        await fs.writeFile(file, content, { mode: 0o600 });
        adapters.push(file);
        records.push(Number(artifact.recordCount));
      }
      const outputDirectory = path.join(
        local.sessionDirectory,
        "outputs",
        "cluster",
      );
      await fs.mkdir(outputDirectory, { recursive: true, mode: 0o700 });
      const combined = path.join(outputDirectory, "adapter.gguf");
      const args = ["aggregate-gguf-adapters"];
      for (const adapter of adapters) args.push("--adapter", adapter);
      for (const count of records) args.push("--records", String(count));
      args.push("--output", combined);
      await command(executable, args);
      await privateJson(path.join(outputDirectory, "cluster.json"), {
        schemaVersion: 1,
        modelVersion: plans[0].request.modelVersion,
        tier: plans[0].request.tier,
        workers: plans.map((plan, index) => ({
          id: plan.id,
          name: plan.name,
          records: records[index],
          platform: plan.peer?.platform ?? process.platform,
          architecture: plan.peer?.architecture ?? process.arch,
          devices: plan.benchmark.devices,
        })),
        adapter: "adapter.gguf",
        adapterSha256: sha256(await fs.readFile(combined)),
      });
      await updateSession(local.sessionDirectory, {
        status: "completed",
        phase: "complete",
        progress: 100,
        trainingPercent: 100,
        indeterminate: false,
        message: `Cluster training completed on ${plans.length} machines`,
        endedAt: new Date().toISOString(),
        error: undefined,
      });
    } catch (error) {
      await Promise.allSettled(
        remoteJobs.map(({ peer, jobId }) =>
          this.cluster.request(peer, "cancel-job", { jobId }),
        ),
      );
      await this.sessions.stop(local.id).catch(() => null);
      await updateSession(local.sessionDirectory, {
        status: "failed",
        phase: "publishing",
        indeterminate: false,
        message: "Cluster training failed; open the log for details",
        error: error instanceof Error ? error.message : String(error),
        endedAt: new Date().toISOString(),
      }).catch(() => null);
    } finally {
      await fs
        .unlink(path.join(local.sessionDirectory, ".cluster-controller"))
        .catch(() => null);
      await fs.rm(staging, { recursive: true, force: true });
    }
  }
}
