export type Theme = "dark" | "blue-dark" | "blue-light";
export type TrainingStage = "fine-tuning" | "alignment" | "fine-tune-align";
export type AlignmentType =
  | "auto"
  | "dpo"
  | "ipo"
  | "simpo"
  | "orpo"
  | "cpo"
  | "kto"
  | "ppo"
  | "reinforce"
  | "rloo"
  | "grpo";
export type LoraTargetModule =
  | "self_attn.q_proj"
  | "self_attn.k_proj"
  | "self_attn.v_proj"
  | "self_attn.o_proj"
  | "mlp.gate_proj"
  | "mlp.up_proj"
  | "mlp.down_proj";

export type Preferences = {
  version: 1;
  theme: Theme;
  backendExecutable: string;
  autoUpdateEnabled: boolean;
  sessionsRoot: string;
  sessionRoots: string[];
};

export type TrainingRequest = {
  sessionsRoot: string;
  modelSource: "official" | "custom";
  tier: "small" | "medium" | "large";
  customModelFolder: string;
  engine: "auto" | "mlx" | "llama.cpp";
  accelerator: "auto" | "metal" | "mps" | "cuda" | "vulkan" | "cpu";
  stage: TrainingStage;
  fineTuneData: string;
  alignmentData: string;
  reuseDataset: boolean;
  adapter: string;
  alignmentType: AlignmentType;
  optimizer: "auto" | "sgd" | "adamw";
  autoSettings: boolean;
  multiGpu: "auto" | "on" | "off";
  liveRollouts: boolean;
  sessionName: string;
  iterations: number;
  alignmentIterations: number;
  batchSize: number | null;
  gradientAccumulationSteps: number | null;
  gradientCheckpoint: boolean;
  maxSeqLength: number | null;
  learningRate: number | null;
  alignmentLearningRate: number | null;
  rank: number | null;
  scale: number | null;
  numLayers: number | null;
  dropout: number | null;
  seed: number | null;
  saveEvery: number | null;
  stepsPerReport: number | null;
  stepsPerEval: number | null;
  validationBatches: number | null;
  maskPrompt: boolean;
  targetModules: LoraTargetModule[];
  alignmentBeta: number;
  alignmentGamma: number;
  ppoClip: number;
  rolloutMaxTokens: number;
  rolloutsPerPrompt: number;
  rolloutTemperature: number;
  rolloutTopP: number;
  rolloutSeed: number;
  ggufBatchSize: number | null;
  ggufThreads: number | null;
  distributedWorkers: number | null;
  splitMode: "auto" | "none" | "layer" | "row" | "tensor";
  tensorSplit: string;
  mainGpu: number | null;
  devices: string;
};

export type SessionState = {
  schemaVersion: 1;
  id: string;
  name: string;
  status:
    | "queued"
    | "running"
    | "paused"
    | "stopping"
    | "completed"
    | "failed"
    | "stopped";
  phase:
    | "preparing"
    | "download"
    | "fine-tuning"
    | "rollouts"
    | "alignment"
    | "publishing"
    | "complete";
  progress: number;
  indeterminate: boolean;
  message: string;
  createdAt: string;
  startedAt?: string;
  endedAt?: string;
  sessionDirectory: string;
  logPath: string;
  command: string;
  error?: string;
  request?: Partial<TrainingRequest>;
};

export type BackendStatus = {
  available: boolean;
  executable: string;
  version: string;
  message: string;
};

export type BackendInstallStatus = {
  state:
    | "idle"
    | "preparing-python"
    | "downloading"
    | "extracting"
    | "installing"
    | "ready"
    | "error";
  message: string;
  percent?: number;
  executable?: string;
  sourceDirectory?: string;
};

export type AppUpdateStatus = {
  state:
    | "disabled"
    | "idle"
    | "checking"
    | "available"
    | "downloading"
    | "ready"
    | "installing"
    | "current"
    | "error"
    | "unsupported";
  message: string;
  currentVersion: string;
  version?: string;
  percent?: number;
  channel?: string;
};

export type OsAiBridge = {
  platform: string;
  loadPreferences(): Promise<Preferences>;
  savePreferences(value: Preferences): Promise<Preferences>;
  chooseDirectory(title: string): Promise<string>;
  chooseDataset(title: string): Promise<string>;
  chooseFile(title: string): Promise<string>;
  chooseBackend(): Promise<string>;
  backendStatus(): Promise<BackendStatus>;
  backendInstallStatus(): Promise<BackendInstallStatus>;
  installBackend(): Promise<BackendInstallStatus>;
  startTraining(value: TrainingRequest): Promise<SessionState>;
  pauseTraining(id: string): Promise<SessionState>;
  resumeTraining(id: string): Promise<SessionState>;
  stopTraining(id: string): Promise<SessionState>;
  deleteSession(id: string): Promise<void>;
  listSessions(): Promise<SessionState[]>;
  sessionLog(id: string): Promise<string>;
  revealSession(id: string): Promise<void>;
  openSessionsFolder(): Promise<void>;
  appUpdateStatus(): Promise<AppUpdateStatus>;
  setAppAutoUpdate(enabled: boolean): Promise<AppUpdateStatus>;
  checkForAppUpdate(): Promise<AppUpdateStatus>;
  downloadAppUpdate(): Promise<AppUpdateStatus>;
  installAppUpdate(): Promise<AppUpdateStatus>;
  onAppUpdateStatus(callback: (status: AppUpdateStatus) => void): () => void;
  onBackendInstallStatus(
    callback: (status: BackendInstallStatus) => void,
  ): () => void;
};
