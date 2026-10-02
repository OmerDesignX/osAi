import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import feather from "feather-icons";
import type {
  AlignmentType,
  AutoCalibrationResult,
  AppUpdateStatus,
  BackendInstallStatus,
  BackendStatus,
  HardwareInfo,
  HardwareSnapshot,
  LoraTargetModule,
  Preferences,
  SessionArtifacts,
  SessionState,
  TrainingMetric,
  Theme,
  TrainingRequest,
  TrainingStage,
} from "./types.js";
import osAiIcon from "./assets/osai-icon.png";
import {
  estimatedModelBytes,
  selectHardwarePreset,
  type HardwarePreset,
} from "./hardware-presets.js";
import {
  calibratedLearningRateOptions,
  fittedChoices,
  learningRateOptions,
  type LearningPace,
} from "./lora-guidance.js";
import { DataEditor } from "./DataEditor.js";
import { TrainingWiki } from "./TrainingWiki.js";
import { LossChart, type LossAxis } from "./LossChart.js";

type IconName = keyof typeof feather.icons;

function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const markup = feather.icons[name].toSvg({
    width: size,
    height: size,
    "stroke-width": 1.8,
    "aria-hidden": "true",
  });
  return <span className="icon" dangerouslySetInnerHTML={{ __html: markup }} />;
}

function SettingInfo({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const appRoot =
    typeof document === "undefined" ? null : document.querySelector(".app");
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [open]);
  return (
    <span className="setting-info">
      <button
        type="button"
        aria-label={`About ${label}`}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="info" size={15} />
      </button>
      {open &&
        appRoot &&
        createPortal(
          <div
            className="setting-info-backdrop"
            onPointerDown={(event) => {
              if (event.target === event.currentTarget) setOpen(false);
            }}
          >
            <section
              className="setting-info-dialog"
              role="dialog"
              aria-modal="true"
              aria-label={label}
            >
              <header>
                <strong>{label}</strong>
                <button
                  type="button"
                  aria-label="Close explanation"
                  onClick={() => setOpen(false)}
                >
                  <Icon name="x" size={16} />
                </button>
              </header>
              <p>{children}</p>
            </section>
          </div>,
          appRoot,
        )}
    </span>
  );
}

function PathField({
  label,
  value,
  placeholder,
  onChange,
  onBrowse,
}: {
  label: string;
  value: string;
  placeholder: string;
  onChange(value: string): void;
  onBrowse(): void;
}) {
  return (
    <label className="field path-field">
      <span>{label}</span>
      <div>
        <input
          value={value}
          placeholder={placeholder}
          spellCheck={false}
          onChange={(event) => onChange(event.target.value)}
        />
        <button
          type="button"
          className="icon-button"
          onClick={onBrowse}
          aria-label={`Browse for ${label}`}
        >
          <Icon name="folder" />
        </button>
      </div>
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  placeholder = "Default",
  disabled = false,
  hint,
}: {
  label: string;
  value: number | null;
  onChange(value: number | null): void;
  min?: number;
  max?: number;
  step?: number | "any";
  placeholder?: string;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value ?? ""}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) =>
          onChange(
            event.target.value === "" ? null : Number(event.target.value),
          )
        }
      />
      {hint && <small>{hint}</small>}
    </label>
  );
}

function statusLabel(status: SessionState["status"]) {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function phaseLabel(phase: string) {
  const value = phase.replaceAll("-", " ");
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function friendlyTime(value?: string) {
  if (!value) return "—";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function readableError(error: unknown) {
  return (error instanceof Error ? error.message : String(error))
    .replace(/^Error invoking remote method '[^']+':\s*(?:Error:\s*)?/i, "")
    .replace(/^Error:\s*/i, "")
    .trim();
}

function memoryLabel(bytes: number | null) {
  return bytes === null
    ? "Unavailable"
    : `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

const defaults: TrainingRequest = {
  sessionsRoot: "",
  modelSource: "official",
  modelVersion: "v2",
  tier: "small",
  customModelFolder: "",
  engine: "auto",
  accelerator: "auto",
  stage: "fine-tuning",
  fineTuneData: "",
  alignmentData: "",
  reuseDataset: false,
  adapter: "",
  alignmentType: "auto",
  optimizer: "auto",
  autoSettings: true,
  fullContentContext: true,
  autoStop: true,
  multiGpu: "auto",
  liveRollouts: true,
  sessionName: "",
  iterations: 1,
  alignmentIterations: 10,
  batchSize: null,
  gradientAccumulationSteps: null,
  gradientCheckpoint: true,
  maxSeqLength: null,
  learningRate: null,
  alignmentLearningRate: null,
  rank: null,
  scale: null,
  numLayers: null,
  dropout: null,
  imageWidth: null,
  imageHeight: null,
  videoFps: 2,
  videoMaxFrames: 32,
  assistantTokenId: null,
  seed: null,
  saveEvery: null,
  stepsPerReport: null,
  stepsPerEval: null,
  validationBatches: null,
  maskPrompt: true,
  targetModules: [],
  alignmentBeta: 0.1,
  alignmentGamma: 0.5,
  ppoClip: 0.2,
  rolloutMaxTokens: 32,
  rolloutsPerPrompt: 2,
  rolloutTemperature: 0.8,
  rolloutTopP: 0.95,
  rolloutSeed: 0,
  ggufBatchSize: null,
  ggufThreads: null,
  distributedWorkers: null,
  splitMode: "auto",
  tensorSplit: "",
  mainGpu: null,
  devices: "",
};

const targetModules: Array<{
  value: LoraTargetModule;
  label: string;
}> = [
  { value: "self_attn.q_proj", label: "Attention Q" },
  { value: "self_attn.k_proj", label: "Attention K" },
  { value: "self_attn.v_proj", label: "Attention V" },
  { value: "self_attn.o_proj", label: "Attention O" },
  { value: "mlp.gate_proj", label: "MLP gate" },
  { value: "mlp.up_proj", label: "MLP up" },
  { value: "mlp.down_proj", label: "MLP down" },
];

const fallbackPreferences: Preferences = {
  version: 1,
  theme: "dark",
  backendExecutable: "",
  autoUpdateEnabled: false,
  sessionsRoot: "",
  sessionRoots: [],
};

const fallbackUpdate: AppUpdateStatus = {
  state: "disabled",
  message: "Automatic updates are off",
  currentVersion: "0.1.20",
};

const fallbackBackendInstall: BackendInstallStatus = {
  state: "idle",
  message: "osAi CLI is not installed",
};

const fallbackHardware: HardwareInfo = {
  platform: "unknown",
  architecture: "unknown",
  physicalMemoryBytes: 8 * 1024 ** 3,
  logicalCpuCount: 2,
};

function applyHardwarePreset(
  current: TrainingRequest,
  preset: HardwarePreset,
): TrainingRequest {
  return {
    ...current,
    autoSettings: true,
    batchSize: preset.batchSize,
    maxSeqLength: preset.maxSeqLength,
    numLayers: preset.numLayers,
    rank: preset.rank,
    ggufBatchSize: preset.ggufBatchSize,
    ggufThreads: preset.ggufThreads,
    targetModules: [...preset.targetModules],
  };
}

function resetAdvancedValues(
  current: TrainingRequest,
  preset: HardwarePreset,
): TrainingRequest {
  return applyHardwarePreset(
    {
      ...current,
      optimizer: defaults.optimizer,
      iterations: defaults.iterations,
      alignmentIterations: defaults.alignmentIterations,
      gradientAccumulationSteps: defaults.gradientAccumulationSteps,
      gradientCheckpoint: defaults.gradientCheckpoint,
      learningRate: defaults.learningRate,
      alignmentLearningRate: defaults.alignmentLearningRate,
      scale: defaults.scale,
      dropout: defaults.dropout,
      imageWidth: defaults.imageWidth,
      imageHeight: defaults.imageHeight,
      videoFps: defaults.videoFps,
      videoMaxFrames: defaults.videoMaxFrames,
      assistantTokenId: defaults.assistantTokenId,
      seed: defaults.seed,
      saveEvery: defaults.saveEvery,
      stepsPerReport: defaults.stepsPerReport,
      stepsPerEval: defaults.stepsPerEval,
      validationBatches: defaults.validationBatches,
      maskPrompt: defaults.maskPrompt,
      alignmentBeta: defaults.alignmentBeta,
      alignmentGamma: defaults.alignmentGamma,
      ppoClip: defaults.ppoClip,
      rolloutMaxTokens: defaults.rolloutMaxTokens,
      rolloutsPerPrompt: defaults.rolloutsPerPrompt,
      rolloutTemperature: defaults.rolloutTemperature,
      rolloutTopP: defaults.rolloutTopP,
      rolloutSeed: defaults.rolloutSeed,
      distributedWorkers: defaults.distributedWorkers,
      splitMode: defaults.splitMode,
      tensorSplit: defaults.tensorSplit,
      mainGpu: defaults.mainGpu,
      devices: defaults.devices,
    },
    preset,
  );
}

export function App() {
  const [preferences, setPreferences] = useState(fallbackPreferences);
  const [form, setForm] = useState(defaults);
  const [sameDataset, setSameDataset] = useState(true);
  const [advanced, setAdvanced] = useState(false);
  const [guidanceOpen, setGuidanceOpen] = useState(false);
  const [learningPace, setLearningPace] = useState<LearningPace | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<{
    source: string;
    fileCount: number;
    totalBytes: number;
    signature: string;
  } | null>(null);
  const [datasetSummaryError, setDatasetSummaryError] = useState("");
  const [datasetSummaryBusy, setDatasetSummaryBusy] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionState[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [log, setLog] = useState("");
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [lossAxis, setLossAxis] = useState<LossAxis>("progress");
  const [backend, setBackend] = useState<BackendStatus | null>(null);
  const [backendChecking, setBackendChecking] = useState(false);
  const [backendInstall, setBackendInstall] = useState(fallbackBackendInstall);
  const [hardware, setHardware] = useState(fallbackHardware);
  const [hardwareSheetOpen, setHardwareSheetOpen] = useState(false);
  const [hardwareSnapshot, setHardwareSnapshot] =
    useState<HardwareSnapshot | null>(null);
  const [update, setUpdate] = useState(fallbackUpdate);
  const [notice, setNotice] = useState("");
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [calibrationMessage, setCalibrationMessage] = useState("");
  const [calibration, setCalibration] = useState<{
    key: string;
    result: AutoCalibrationResult;
  } | null>(null);
  const [benchmarkPreset, setBenchmarkPreset] = useState<HardwarePreset | null>(
    null,
  );
  const [benchmarkModelBytes, setBenchmarkModelBytes] = useState<number | null>(
    null,
  );
  const [customModelBytes, setCustomModelBytes] = useState<number | null>(null);
  const [benchmarkDevices, setBenchmarkDevices] = useState<string[]>([]);
  const [deviceSignature, setDeviceSignature] = useState<string | null>(null);
  const [sessionArtifacts, setSessionArtifacts] =
    useState<SessionArtifacts | null>(null);
  const [noticeExpanded, setNoticeExpanded] = useState(false);
  const [sessionMenu, setSessionMenu] = useState<{
    id: string;
    top: number;
    left: number;
  } | null>(null);
  const [deleteCandidate, setDeleteCandidate] = useState<SessionState | null>(
    null,
  );
  const [restartCandidate, setRestartCandidate] = useState<SessionState | null>(
    null,
  );
  const [deletingSession, setDeletingSession] = useState(false);
  const [restartingSession, setRestartingSession] = useState(false);
  const [starting, setStarting] = useState(false);
  const [autoStopBusy, setAutoStopBusy] = useState(false);
  const [sessionControl, setSessionControl] = useState<
    "" | "pausing" | "resuming" | "saving" | "stopping"
  >("");
  const [showLogLatest, setShowLogLatest] = useState(false);
  const [wikiOpen, setWikiOpen] = useState(false);
  const [wikiActive, setWikiActive] = useState(false);
  const [dataEditorOpen, setDataEditorOpen] = useState(false);
  const [dataEditorActive, setDataEditorActive] = useState(false);
  const [dataEditorSource, setDataEditorSource] = useState("");
  const appRootRef = useRef<HTMLDivElement | null>(null);
  const quickSettingsRef = useRef<HTMLElement | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);
  const followLogRef = useRef(true);
  const selectionClearedRef = useRef(false);
  const backendCheckRef = useRef<Promise<BackendStatus> | null>(null);
  const autoInstallStartedRef = useRef(false);

  const selected = useMemo(
    () => sessions.find((session) => session.id === selectedId) || null,
    [selectedId, sessions],
  );
  const active = sessions.find((session) =>
    ["queued", "running", "pausing", "paused", "stopping"].includes(
      session.status,
    ),
  );
  const autoStopAvailable =
    (active?.request?.stage || form.stage) !== "alignment";
  useEffect(() => {
    if (active) setGuidanceOpen(false);
  }, [active?.id]);
  const sessionMenuSession = sessions.find(
    (session) => session.id === sessionMenu?.id,
  );
  const fallbackPreset = useMemo(
    () =>
      selectHardwarePreset(hardware, form.engine, form.tier, form.modelVersion),
    [hardware, form.engine, form.tier, form.modelVersion],
  );
  const hardwarePreset = benchmarkPreset || fallbackPreset;
  const guidanceBatchChoices = fittedChoices(
    hardwarePreset.batchSize,
    [1, 2, 4, 8, 16],
  );
  const guidanceRankChoices = fittedChoices(
    hardwarePreset.rank,
    [1, 2, 4, 8, 16, 32],
  );
  const datasetSource = form.fineTuneData.trim();
  const currentDatasetSummary =
    datasetSummary?.source === datasetSource ? datasetSummary : null;
  const datasetSummaryPending = Boolean(
    datasetSource &&
    (datasetSummaryBusy || (!currentDatasetSummary && !datasetSummaryError)),
  );
  const calibrationSignature = JSON.stringify([
    form.modelSource,
    form.modelVersion,
    form.tier,
    form.customModelFolder,
    form.engine,
    form.accelerator,
    form.multiGpu,
    form.devices,
    form.fineTuneData,
    form.fullContentContext,
    form.optimizer,
    form.iterations,
    form.scale,
    form.dropout,
    form.seed,
    form.gradientAccumulationSteps,
    form.gradientCheckpoint,
    form.maskPrompt,
    form.splitMode,
    form.tensorSplit,
    form.mainGpu,
    form.distributedWorkers,
    currentDatasetSummary?.signature ?? null,
    deviceSignature,
  ]);
  const currentCalibration =
    calibration?.key === calibrationSignature &&
    form.autoSettings &&
    form.learningRate !== null &&
    form.learningRate <= calibration.result.learning_rate * 1.0001
      ? calibration.result
      : null;
  const rateOptions = useMemo(
    () =>
      currentCalibration
        ? calibratedLearningRateOptions(currentCalibration.learning_rate)
        : learningRateOptions({
            datasetBytes: currentDatasetSummary?.totalBytes ?? null,
            modelBytes:
              form.modelSource === "custom"
                ? ((form.autoSettings ? benchmarkModelBytes : null) ??
                  customModelBytes ??
                  3 * 1024 ** 3)
                : ((form.autoSettings ? benchmarkModelBytes : null) ??
                  estimatedModelBytes(
                    hardware,
                    form.engine,
                    form.tier,
                    form.modelVersion,
                  )),
            memoryBytes: hardware.physicalMemoryBytes,
            batchSize: form.batchSize ?? hardwarePreset.batchSize,
            rank: form.rank ?? hardwarePreset.rank,
            epochs: form.iterations,
          }),
    [
      currentDatasetSummary?.totalBytes,
      benchmarkModelBytes,
      customModelBytes,
      form.modelSource,
      form.autoSettings,
      form.engine,
      form.tier,
      form.modelVersion,
      form.batchSize,
      form.rank,
      form.iterations,
      hardware,
      hardwarePreset.batchSize,
      hardwarePreset.rank,
      currentCalibration?.learning_rate,
    ],
  );
  const rateSelection =
    learningPace ?? (form.learningRate === null ? "backend" : "previous");
  const selectRate = (value: string) => {
    const selectedRate = rateOptions.find((option) => option.pace === value);
    setLearningPace(selectedRate?.pace ?? null);
    setForm((current) => ({
      ...current,
      learningRate: selectedRate?.rate ?? null,
    }));
  };

  useEffect(() => {
    setDatasetSummary(null);
    setDatasetSummaryError("");
    if (!datasetSource) {
      setDatasetSummaryBusy(false);
      return;
    }
    setDatasetSummaryBusy(true);
    let current = true;
    const timer = window.setTimeout(() => {
      void window.osai
        .datasetTrainingSummary(datasetSource)
        .then((summary) => {
          if (current) {
            setDatasetSummary({ ...summary, source: datasetSource });
            setDatasetSummaryError("");
          }
        })
        .catch((error) => {
          if (current) {
            setDatasetSummary(null);
            setDatasetSummaryError(readableError(error));
          }
        })
        .finally(() => {
          if (current) setDatasetSummaryBusy(false);
        });
    }, 300);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [datasetSource]);

  useEffect(() => {
    if (form.modelSource !== "custom") return;
    setCustomModelBytes(null);
    if (!form.customModelFolder.trim()) return;
    let current = true;
    const timer = window.setTimeout(() => {
      void window.osai
        .modelTrainingSummary(form.customModelFolder.trim())
        .then((bytes) => {
          if (current) setCustomModelBytes(bytes);
        })
        .catch(() => {
          if (current) setCustomModelBytes(null);
        });
    }, 300);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [form.modelSource, form.customModelFolder]);

  useEffect(() => {
    if (!learningPace || datasetSummaryPending || active || starting) return;
    const rate = rateOptions.find(
      (option) => option.pace === learningPace,
    )?.rate;
    if (rate !== undefined)
      setForm((current) =>
        current.learningRate === rate
          ? current
          : { ...current, learningRate: rate },
      );
  }, [learningPace, rateOptions, datasetSummaryPending, active?.id, starting]);

  useEffect(() => {
    if (!guidanceOpen) return;
    quickSettingsRef.current?.querySelector("select")?.focus();
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setGuidanceOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [guidanceOpen]);

  useEffect(() => window.osai.onCalibrationProgress(setCalibrationMessage), []);

  useEffect(() => {
    if (!hardwareSheetOpen) return;
    let current = true;
    const refresh = () => {
      void window.osai
        .hardwareSnapshot()
        .then((snapshot) => {
          if (current) setHardwareSnapshot(snapshot);
        })
        .catch((error) => {
          if (current) setNotice(readableError(error));
        });
    };
    refresh();
    const interval = window.setInterval(refresh, 5_000);
    return () => {
      current = false;
      window.clearInterval(interval);
    };
  }, [hardwareSheetOpen]);

  const refreshSessions = useCallback(async () => {
    const next = await window.osai.listSessions();
    setSessions(next);
    setSelectedId((current) =>
      current && next.some((session) => session.id === current)
        ? current
        : selectionClearedRef.current
          ? ""
          : next.find((session) =>
              ["queued", "running", "pausing", "paused", "stopping"].includes(
                session.status,
              ),
            )?.id ||
            next[0]?.id ||
            "",
    );
  }, []);

  const refreshBackend = useCallback(() => {
    if (backendCheckRef.current) return backendCheckRef.current;
    setBackendChecking(true);
    const request = window.osai
      .backendStatus()
      .catch((error): BackendStatus => ({
        available: false,
        executable: "osai",
        version: "",
        message: readableError(error) || "The osAi CLI could not be checked",
      }))
      .then((status) => {
        setBackend(status);
        return status;
      })
      .finally(() => {
        backendCheckRef.current = null;
        setBackendChecking(false);
      });
    backendCheckRef.current = request;
    return request;
  }, []);

  useEffect(() => {
    void Promise.all([
      window.osai.loadPreferences().then((value) => {
        setPreferences(value);
        setForm((current) => ({
          ...current,
          sessionsRoot: current.sessionsRoot || value.sessionsRoot,
        }));
      }),
      refreshSessions(),
      refreshBackend(),
      window.osai.backendInstallStatus().then(setBackendInstall),
      window.osai.appUpdateStatus().then(setUpdate),
      window.osai.hardwareInfo().then(setHardware),
    ]).catch((error) => setNotice(readableError(error)));
    const removeUpdateListener = window.osai.onAppUpdateStatus(setUpdate);
    const removeBackendInstallListener = window.osai.onBackendInstallStatus(
      (status) => {
        setBackendInstall(status);
        if (status.state === "error") setNotice(status.message);
        if (status.state === "ready") {
          void window.osai.loadPreferences().then(setPreferences);
          void refreshBackend();
        }
      },
    );
    const interval = window.setInterval(() => void refreshSessions(), 1_200);
    return () => {
      window.clearInterval(interval);
      removeUpdateListener();
      removeBackendInstallListener();
    };
  }, [refreshBackend, refreshSessions]);

  useEffect(() => {
    if (
      !form.autoSettings ||
      active ||
      starting ||
      calibrationBusy ||
      currentCalibration
    )
      return;
    setForm((current) => applyHardwarePreset(current, hardwarePreset));
  }, [
    hardwarePreset,
    selectedId,
    active?.id,
    starting,
    calibrationBusy,
    currentCalibration,
  ]);

  useEffect(() => {
    if (!form.autoSettings || !backend?.available || active || starting) {
      setDeviceSignature(null);
      return;
    }
    let current = true;
    const discover = () => {
      void window.osai
        .autoDevices(form.accelerator)
        .then((inventory) => {
          if (current) setDeviceSignature(JSON.stringify(inventory));
        })
        .catch((error) => {
          if (current) setNotice(readableError(error));
        });
    };
    setDeviceSignature(null);
    discover();
    const interval = window.setInterval(discover, 15_000);
    return () => {
      current = false;
      window.clearInterval(interval);
    };
  }, [
    backend?.available,
    form.autoSettings,
    form.accelerator,
    active?.id,
    starting,
  ]);

  useEffect(() => {
    if (
      !form.autoSettings ||
      !backend?.available ||
      deviceSignature === null ||
      calibrationBusy ||
      currentCalibration ||
      active ||
      starting
    ) {
      setBenchmarkBusy(false);
      return;
    }
    if (form.modelSource === "custom" && !form.customModelFolder) return;
    let current = true;
    setBenchmarkPreset(null);
    setBenchmarkModelBytes(null);
    setBenchmarkBusy(true);
    const timer = window.setTimeout(() => {
      void window.osai
        .autoBenchmark(form)
        .then((result) => {
          if (!current) return;
          const settings = result.settings;
          setBenchmarkPreset({
            profile: settings.profile,
            batchSize: settings.batch_size,
            maxSeqLength: settings.max_seq_length,
            numLayers: settings.num_layers,
            rank: settings.rank,
            ggufBatchSize: settings.gguf_batch_size,
            ggufThreads: settings.gguf_threads,
            targetModules: settings.target_modules,
          });
          setBenchmarkModelBytes(settings.model_size_bytes);
          setBenchmarkDevices(result.devices);
          void window.osai.hardwareInfo().then(setHardware);
        })
        .catch((error) => {
          if (current) setNotice(readableError(error));
        })
        .finally(() => {
          if (current) setBenchmarkBusy(false);
        });
    }, 350);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [
    backend?.available,
    form.autoSettings,
    form.modelSource,
    form.modelVersion,
    form.tier,
    form.customModelFolder,
    form.engine,
    form.accelerator,
    form.multiGpu,
    form.devices,
    form.splitMode,
    form.tensorSplit,
    form.mainGpu,
    deviceSignature,
    active?.id,
    starting,
    calibrationBusy,
    currentCalibration,
  ]);

  useEffect(() => setNoticeExpanded(false), [notice]);

  useEffect(() => {
    if (!selected || selected.status !== "completed") {
      setSessionArtifacts(null);
      return;
    }
    let current = true;
    void window.osai.sessionArtifacts(selected.id).then((result) => {
      if (current) setSessionArtifacts(result);
    });
    return () => {
      current = false;
    };
  }, [selected?.id, selected?.status]);

  useEffect(() => {
    if (!sessionMenu) return;
    const dismiss = (event: PointerEvent) => {
      const target = event.target;
      if (
        target instanceof Element &&
        target.closest(".session-tab-menu, .session-tab-more")
      )
        return;
      setSessionMenu(null);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSessionMenu(null);
    };
    window.addEventListener("pointerdown", dismiss);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("pointerdown", dismiss);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [sessionMenu]);

  useEffect(() => {
    if (!selected) {
      setLearningPace(null);
      setForm({
        ...defaults,
        sessionsRoot: preferences.sessionsRoot,
      });
      setSameDataset(true);
      setAdvanced(false);
      return;
    }
    if (!selected.request) return;
    setLearningPace(null);
    const restored = Object.fromEntries(
      Object.entries(selected.request).filter(
        ([, value]) => value !== undefined,
      ),
    ) as Partial<TrainingRequest>;
    setForm({
      ...defaults,
      ...restored,
      modelVersion: restored.modelVersion ?? "v1",
      targetModules: Array.isArray(selected.request.targetModules)
        ? selected.request.targetModules
        : [],
    });
    setSameDataset(
      Boolean(
        selected.request.stage === "fine-tune-align" &&
        selected.request.reuseDataset,
      ),
    );
  }, [selected?.id]);

  useEffect(() => {
    followLogRef.current = true;
    setShowLogLatest(false);
    if (!selected) {
      setLog("");
      return;
    }
    let cancelled = false;
    const load = async () => {
      const output = await window.osai.sessionLog(selected.id);
      if (!cancelled) setLog(output);
    };
    void load();
    const interval = window.setInterval(() => void load(), 1_200);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [selected?.id]);

  useEffect(() => {
    if (!selected) {
      setMetrics([]);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const history = await window.osai.sessionMetrics(selected.id);
        if (!cancelled) setMetrics(history);
      } catch (error) {
        if (!cancelled) setNotice(readableError(error));
      }
    };
    setMetrics([]);
    void load();
    const interval = window.setInterval(() => void load(), 4_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [selected?.id]);

  useEffect(() => {
    if (!logRef.current || !followLogRef.current) return;
    const frame = window.requestAnimationFrame(() => {
      if (!logRef.current || !followLogRef.current) return;
      logRef.current.scrollTop = logRef.current.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [log]);

  const handleLogScroll = () => {
    if (!logRef.current) return;
    const distanceFromBottom =
      logRef.current.scrollHeight -
      logRef.current.scrollTop -
      logRef.current.clientHeight;
    const atBottom = distanceFromBottom <= 24;
    followLogRef.current = atBottom;
    setShowLogLatest(!atBottom);
  };

  const jumpToLatestLog = () => {
    if (!logRef.current) return;
    followLogRef.current = true;
    setShowLogLatest(false);
    logRef.current.scrollTo({
      top: logRef.current.scrollHeight,
      behavior: "smooth",
    });
  };

  const savePreferences = async (next: Preferences) => {
    setPreferences(next);
    setPreferences(await window.osai.savePreferences(next));
  };

  const setTheme = (theme: Theme) => {
    void savePreferences({ ...preferences, theme });
  };

  const chooseDirectory = async (title: string, key: keyof TrainingRequest) => {
    const value = await window.osai.chooseDirectory(title);
    if (value) setForm((current) => ({ ...current, [key]: value }));
  };

  const chooseDataset = async (title: string, key: keyof TrainingRequest) => {
    const value = await window.osai.chooseDataset(title);
    if (value) setForm((current) => ({ ...current, [key]: value }));
  };

  const openDataEditor = (source = "") => {
    setDataEditorSource(source || form.fineTuneData || form.alignmentData);
    setDataEditorOpen(true);
    setDataEditorActive(true);
    setWikiActive(false);
    setSessionMenu(null);
  };

  const useEditedDataset = (
    destination: "fineTuneData" | "alignmentData",
    source: string,
    _recommendedTokenLimit: number,
  ) => {
    setForm((current) => ({
      ...current,
      [destination]: source,
      maxSeqLength: current.maxSeqLength,
    }));
    setDataEditorActive(false);
  };

  const chooseAdapter = async () => {
    const value = await window.osai.chooseFile("Choose an osAi adapter");
    if (value) setForm((current) => ({ ...current, adapter: value }));
  };

  const calibrate = async () => {
    if (!form.fineTuneData.trim()) {
      setNotice("Choose a fine-tuning dataset before calibration");
      return;
    }
    setCalibrationBusy(true);
    setCalibrationMessage("Checking model, dataset and available hardware…");
    setCalibration(null);
    setNotice("");
    try {
      const result = await window.osai.autoCalibration(form);
      setCalibration({ key: calibrationSignature, result });
      setBenchmarkPreset({
        profile: result.settings.profile,
        batchSize: result.settings.batch_size,
        maxSeqLength: result.settings.max_seq_length,
        numLayers: result.settings.num_layers,
        rank: result.settings.rank,
        ggufBatchSize: result.settings.gguf_batch_size,
        ggufThreads: result.settings.gguf_threads,
        targetModules: result.settings.target_modules,
      });
      setBenchmarkModelBytes(result.settings.model_size_bytes);
      setBenchmarkDevices(result.devices);
      setLearningPace(null);
      setForm((current) => ({
        ...current,
        learningRate: result.learning_rate,
      }));
      setCalibrationMessage(
        `Ready · pilot loss ${result.first_loss.toFixed(3)} → ${result.last_loss.toFixed(3)} · rate ${result.learning_rate.toExponential(2)}`,
      );
    } catch (error) {
      setCalibrationMessage("Calibration needs attention");
      setNotice(readableError(error));
    } finally {
      setCalibrationBusy(false);
    }
  };

  const startTraining = async () => {
    if (
      form.autoSettings &&
      form.stage !== "alignment" &&
      !currentCalibration
    ) {
      setNotice("Calibrate the selected model and dataset before training");
      return;
    }
    setNotice("");
    setStarting(true);
    try {
      const backendState = await window.osai.backendStatus();
      setBackend(backendState);
      if (!backendState.available)
        throw new Error(
          "Install the current osAi CLI from main before training.",
        );
      const dataSources = [
        ...(form.stage !== "alignment" ? [form.fineTuneData] : []),
        ...(form.stage !== "fine-tuning"
          ? [
              form.stage === "fine-tune-align" && sameDataset
                ? form.fineTuneData
                : form.alignmentData,
            ]
          : []),
      ].filter(
        (value, index, values) => value && values.indexOf(value) === index,
      );
      for (const source of dataSources) {
        if (source.toLowerCase().endsWith(".parquet")) continue;
        let inspection;
        try {
          inspection = await window.osai.inspectDataset(source);
        } catch (error) {
          if (readableError(error).includes("contains no JSON")) continue;
          throw error;
        }
        if (inspection.invalidRows > 0) {
          openDataEditor(source);
          throw new Error(
            `${inspection.invalidRows} of ${inspection.totalRows} dataset rows need review. The Data Editor has been opened with the first issues and inferred field mappings.`,
          );
        }
      }
      const request = {
        ...form,
        autoStop: form.stage !== "alignment" && form.autoStop,
        sessionsRoot: form.sessionsRoot || preferences.sessionsRoot,
        reuseDataset: form.stage === "fine-tune-align" && sameDataset,
        alignmentData:
          form.stage === "fine-tune-align" && sameDataset
            ? form.fineTuneData
            : form.alignmentData,
      };
      setPreferences(
        await window.osai.savePreferences({
          ...preferences,
          sessionsRoot: request.sessionsRoot,
        }),
      );
      const session = await window.osai.startTraining(request);
      selectionClearedRef.current = false;
      setSelectedId(session.id);
      await refreshSessions();
    } catch (error) {
      const message = readableError(error);
      if (
        message.includes("Calibrate this model") ||
        message.includes("Training data changed") ||
        message.includes("Available hardware changed")
      ) {
        setCalibration(null);
        setCalibrationMessage("Selections changed; run calibration again.");
        void window.osai
          .datasetTrainingSummary(form.fineTuneData)
          .then((summary) =>
            setDatasetSummary({ ...summary, source: form.fineTuneData.trim() }),
          )
          .catch(() => setDatasetSummary(null));
      }
      setNotice(message);
    } finally {
      setStarting(false);
    }
  };

  const stopTraining = async (id: string) => {
    setNotice("");
    setSessionControl("stopping");
    try {
      await window.osai.stopTraining(id);
      await refreshSessions();
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setSessionControl("");
    }
  };

  const changeAutoStop = async (enabled: boolean) => {
    if (!active) {
      setForm((current) => ({ ...current, autoStop: enabled }));
      return;
    }
    setAutoStopBusy(true);
    setNotice("");
    try {
      const updated = await window.osai.setAutoStop(active.id, enabled);
      setForm((current) => ({ ...current, autoStop: enabled }));
      setSessions((current) =>
        current.map((session) =>
          session.id === updated.id ? updated : session,
        ),
      );
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setAutoStopBusy(false);
    }
  };

  const pauseTraining = async (id: string) => {
    setNotice("");
    setSessionControl("pausing");
    try {
      await window.osai.pauseTraining(id);
      await refreshSessions();
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setSessionControl("");
    }
  };

  const resumeTraining = async (id: string) => {
    setNotice("");
    setSessionControl("resuming");
    try {
      await window.osai.resumeTraining(id);
      await refreshSessions();
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setSessionControl("");
    }
  };

  const saveCheckpoint = async (id: string) => {
    setNotice("");
    setSessionControl("saving");
    try {
      await window.osai.saveCheckpoint(id);
      setNotice("Checkpoint requested. Saving at the next safe training step.");
      await refreshSessions();
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setSessionControl("");
    }
  };

  const toggleSessionMenu = (id: string, button: HTMLButtonElement) => {
    if (sessionMenu?.id === id) {
      setSessionMenu(null);
      return;
    }
    const bounds = button.getBoundingClientRect();
    const width = 190;
    const height = 106;
    const left = Math.min(
      window.innerWidth - width - 12,
      Math.max(12, bounds.right - width),
    );
    const below = bounds.bottom + 8;
    setSessionMenu({
      id,
      left,
      top:
        below + height <= window.innerHeight - 12
          ? below
          : Math.max(12, bounds.top - height - 8),
    });
  };

  const deleteSession = async () => {
    if (!deleteCandidate) return;
    setDeletingSession(true);
    setNotice("");
    try {
      await window.osai.deleteSession(deleteCandidate.id);
      if (deleteCandidate.id === selectedId) {
        selectionClearedRef.current = true;
        setSelectedId("");
        setLog("");
        setForm({
          ...defaults,
          sessionsRoot: preferences.sessionsRoot,
        });
        setSameDataset(true);
        setAdvanced(false);
      }
      setDeleteCandidate(null);
      await refreshSessions();
    } catch (error) {
      setNotice(readableError(error));
    } finally {
      setDeletingSession(false);
    }
  };

  const restartSession = async () => {
    if (!restartCandidate?.request) return;
    setRestartingSession(true);
    setNotice("");
    const request: TrainingRequest = {
      ...defaults,
      ...restartCandidate.request,
      modelVersion: restartCandidate.request.modelVersion ?? "v1",
      targetModules: Array.isArray(restartCandidate.request.targetModules)
        ? restartCandidate.request.targetModules
        : [],
    };
    try {
      const session = await window.osai.restartSession(
        restartCandidate.id,
        request,
      );
      selectionClearedRef.current = false;
      setRestartCandidate(null);
      setSelectedId(session.id);
      await refreshSessions();
    } catch (error) {
      setRestartCandidate(null);
      setNotice(readableError(error));
      await refreshSessions();
    } finally {
      setRestartingSession(false);
    }
  };

  const runUpdateAction = async () => {
    if (["checking", "downloading", "installing"].includes(update.state))
      return;
    setNotice("");
    try {
      const next =
        update.state === "ready"
          ? await window.osai.installAppUpdate()
          : update.state === "available"
            ? await window.osai.downloadAppUpdate()
            : await window.osai.checkForAppUpdate();
      setUpdate(next);
    } catch (error) {
      setNotice(readableError(error));
    }
  };

  const downloadBackend = async () => {
    setNotice("");
    try {
      const installed = await window.osai.installBackend();
      setBackendInstall(installed);
      if (installed.state === "error") throw new Error(installed.message);
      const nextPreferences = await window.osai.loadPreferences();
      setPreferences(nextPreferences);
      await refreshBackend();
    } catch (error) {
      setNotice(readableError(error));
    }
  };

  useEffect(() => {
    if (
      backend?.available === false &&
      !backendChecking &&
      backendInstall.state === "idle" &&
      !autoInstallStartedRef.current
    ) {
      autoInstallStartedRef.current = true;
      void downloadBackend();
    }
  }, [backend, backendChecking, backendInstall.state]);

  const updateBusy = ["checking", "downloading", "installing"].includes(
    update.state,
  );
  const updateLabel =
    update.state === "available"
      ? "Update"
      : update.state === "downloading"
        ? `${update.percent || 0}%`
        : update.state === "ready"
          ? "Install"
          : update.state === "checking"
            ? "Checking"
            : update.state === "installing"
              ? "Opening"
              : update.state === "error"
                ? "Retry"
                : update.state === "current"
                  ? "Ready"
                  : "Download";
  const stage = form.stage;
  const needsFineTune = stage !== "alignment";
  const needsAlignment = stage !== "fine-tuning";
  const backendReady = backend?.available === true;
  const backendInstallBusy = [
    "preparing-python",
    "downloading",
    "extracting",
    "installing",
  ].includes(backendInstall.state);
  const backendInstallLabel =
    backendInstall.state === "preparing-python"
      ? "Preparing Python…"
      : backendInstall.state === "downloading"
        ? typeof backendInstall.percent === "number"
          ? `Downloading · ${backendInstall.percent}%`
          : "Downloading…"
        : backendInstall.state === "extracting"
          ? "Extracting…"
          : backendInstall.state === "installing"
            ? "Installing packages…"
            : backendInstall.state === "error"
              ? "Try installation again"
              : "Install osAi";
  const footerIndeterminate =
    calibrationBusy ||
    benchmarkBusy ||
    starting ||
    Boolean(active?.indeterminate) ||
    (backendInstallBusy && typeof backendInstall.percent !== "number");
  const footerPercent = active
    ? active.progress
    : backendInstallBusy && typeof backendInstall.percent === "number"
      ? backendInstall.percent
      : 0;

  return (
    <div
      ref={appRootRef}
      className={
        "app " + preferences.theme + " platform-" + window.osai.platform
      }
    >
      <div className="mac-titlebar-safe-area" aria-hidden="true" />
      <header className="topbar">
        <div
          className="auto-settings-strip"
          aria-label="Selected training settings"
        >
          {form.autoSettings ? (
            <>
              <span className="auto-settings-title">
                {benchmarkBusy ? "Measuring hardware…" : "Auto settings"}
              </span>
              {(selected?.autoSettingsSummary
                ? selected.autoSettingsSummary.split(" · ")
                : [
                    `Context ${hardwarePreset.maxSeqLength}`,
                    `Batch ${hardwarePreset.batchSize}`,
                    `Rank ${hardwarePreset.rank}`,
                    `Microbatch ${hardwarePreset.ggufBatchSize}`,
                    benchmarkDevices.length
                      ? benchmarkDevices.join(" + ")
                      : "CPU",
                  ]
              ).map((item, index) => (
                <span key={`${item}-${index}`}>{item}</span>
              ))}
            </>
          ) : (
            <span className="auto-settings-title">Manual settings</span>
          )}
          <button
            type="button"
            className="hardware-sheet-trigger"
            onClick={() => setHardwareSheetOpen(true)}
            aria-expanded={hardwareSheetOpen}
          >
            <Icon name="cpu" />
            Hardware
          </button>
        </div>

        <div className="global-activity" aria-live="polite">
          {notice && (
            <div className="top-status notification" role="status">
              <Icon name="alert-circle" />
              <button
                type="button"
                className="notice-message"
                title={notice}
                aria-label="Show complete error"
                onClick={() => setNoticeExpanded(true)}
              >
                <span>{notice}</span>
              </button>
              <button
                type="button"
                className="notice-dismiss"
                onClick={() => setNotice("")}
                aria-label="Dismiss error"
              >
                <Icon name="x" />
              </button>
            </div>
          )}
        </div>

        <nav className="topbar-actions" aria-label="Application">
          <button
            className="top-action"
            onClick={() => void window.osai.openSessionsFolder()}
          >
            <Icon name="archive" />
            Sessions
          </button>
          <button
            className={"top-action " + (settingsOpen ? "active" : "")}
            onClick={() => setSettingsOpen((open) => !open)}
            aria-expanded={settingsOpen}
          >
            <Icon name="settings" />
            Settings
          </button>
        </nav>
      </header>

      {backendReady ? (
        <main className="workspace">
          <aside className="training-panel">
            <header className="panel-header">
              <div>
                <h1>Training</h1>
                <p>Configure a new local model session.</p>
              </div>
            </header>

            <fieldset
              className="training-form"
              disabled={Boolean(active) || starting || calibrationBusy}
            >
              <section className="form-section session-controls">
                <div className="section-heading">
                  <h2>Session</h2>
                  <p>Name this run and choose where its files are saved.</p>
                </div>
                <label className="field session-name-control">
                  <span>Session name</span>
                  <input
                    value={form.sessionName}
                    placeholder={form.tier + "-" + form.stage}
                    onChange={(event) =>
                      setForm({ ...form, sessionName: event.target.value })
                    }
                  />
                </label>
                <PathField
                  label="Save sessions in"
                  value={form.sessionsRoot}
                  placeholder={preferences.sessionsRoot || "osAi/sessions"}
                  onChange={(sessionsRoot) =>
                    setForm({ ...form, sessionsRoot })
                  }
                  onBrowse={() =>
                    void chooseDirectory(
                      "Choose where to save osAi sessions",
                      "sessionsRoot",
                    )
                  }
                />
              </section>

              <section className="form-section">
                <div className="section-heading">
                  <h2>
                    <span className="workflow-step">1</span> Model
                  </h2>
                  <p>Use an osCode model or choose your own model folder.</p>
                </div>

                <div className="segmented two">
                  <button
                    type="button"
                    className={form.modelSource === "official" ? "active" : ""}
                    onClick={() =>
                      setForm({ ...form, modelSource: "official" })
                    }
                  >
                    osCode model
                  </button>
                  <button
                    type="button"
                    className={form.modelSource === "custom" ? "active" : ""}
                    onClick={() => setForm({ ...form, modelSource: "custom" })}
                  >
                    Custom model
                  </button>
                </div>

                {form.modelSource === "official" ? (
                  <>
                    <div
                      className="segmented two model-versions"
                      aria-label="osCode model generation"
                    >
                      {(["v2", "v1"] as const).map((modelVersion) => (
                        <button
                          type="button"
                          key={modelVersion}
                          className={
                            form.modelVersion === modelVersion ? "active" : ""
                          }
                          onClick={() =>
                            setForm({
                              ...form,
                              modelVersion,
                              tier:
                                modelVersion === "v1" && form.tier === "xsmall"
                                  ? "small"
                                  : form.tier,
                            })
                          }
                        >
                          osCode {modelVersion.toUpperCase()}
                        </button>
                      ))}
                    </div>
                    <div
                      className={`model-tiers${form.modelVersion === "v2" ? " four" : ""}`}
                    >
                      {(form.modelVersion === "v2"
                        ? ["xsmall", "small", "medium", "large"]
                        : ["small", "medium", "large"]
                      ).map((tier) => (
                        <button
                          type="button"
                          key={tier}
                          className={form.tier === tier ? "active" : ""}
                          onClick={() =>
                            setForm({
                              ...form,
                              tier: tier as TrainingRequest["tier"],
                            })
                          }
                        >
                          {tier === "xsmall"
                            ? "xSmall"
                            : tier.charAt(0).toUpperCase() + tier.slice(1)}
                        </button>
                      ))}
                    </div>
                    {(form.engine === "llama.cpp" ||
                      (form.engine === "auto" &&
                        !(
                          hardware.platform === "darwin" &&
                          hardware.architecture === "arm64"
                        ))) && (
                      <small>
                        GGUF osCode training adapts the final MLP block. Earlier
                        recurrent blocks have no llama.cpp backward pass.
                      </small>
                    )}
                  </>
                ) : (
                  <PathField
                    label="Model folder"
                    value={form.customModelFolder}
                    placeholder="Folder containing mlx/ or gguf/"
                    onChange={(customModelFolder) =>
                      setForm({ ...form, customModelFolder })
                    }
                    onBrowse={() =>
                      void chooseDirectory(
                        "Choose a custom model folder",
                        "customModelFolder",
                      )
                    }
                  />
                )}

                <div className="field-row three">
                  <label className="field">
                    <span>Engine</span>
                    <select
                      value={form.engine}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          engine: event.target
                            .value as TrainingRequest["engine"],
                        })
                      }
                    >
                      <option value="auto">Auto</option>
                      <option value="mlx">MLX</option>
                      <option value="llama.cpp">llama.cpp</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>Accelerator</span>
                    <select
                      value={form.accelerator}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          accelerator: event.target
                            .value as TrainingRequest["accelerator"],
                        })
                      }
                    >
                      <option value="auto">Auto · GPU first</option>
                      <option value="metal">Metal</option>
                      <option value="mps">MPS</option>
                      <option value="cuda">CUDA</option>
                      <option value="vulkan">Vulkan</option>
                      <option value="cpu">CPU</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>Multi-GPU</span>
                    <select
                      value={form.multiGpu}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          multiGpu: event.target
                            .value as TrainingRequest["multiGpu"],
                        })
                      }
                    >
                      <option value="auto">Auto</option>
                      <option value="on">Require</option>
                      <option value="off">Off</option>
                    </select>
                  </label>
                </div>
              </section>

              <section className="form-section">
                <div className="section-heading">
                  <h2>
                    <span className="workflow-step">2</span> Pipeline
                  </h2>
                  <p>Choose the work osAi should run.</p>
                </div>
                <div className="segmented three">
                  {(
                    [
                      ["fine-tuning", "Fine-tune"],
                      ["alignment", "Align"],
                      ["fine-tune-align", "Fine-tune + align"],
                    ] as [TrainingStage, string][]
                  ).map(([value, label]) => (
                    <button
                      type="button"
                      key={value}
                      className={stage === value ? "active" : ""}
                      onClick={() => setForm({ ...form, stage: value })}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </section>

              <section className="form-section">
                <div className="section-heading">
                  <h2>
                    <span className="workflow-step">3</span> Data &amp; context
                  </h2>
                  <p>
                    Text, image, video and compatible audio data stay on this
                    computer.
                  </p>
                </div>
                {needsFineTune && (
                  <PathField
                    label="Fine-tuning dataset"
                    value={form.fineTuneData}
                    placeholder="Dataset folder or JSON, JSONL, Parquet file"
                    onChange={(fineTuneData) =>
                      setForm({ ...form, fineTuneData })
                    }
                    onBrowse={() =>
                      void chooseDataset(
                        "Choose a fine-tuning dataset",
                        "fineTuneData",
                      )
                    }
                  />
                )}
                {needsFineTune && (
                  <div className="context-choice">
                    <div className="setting-heading">
                      <strong>Context</strong>
                      <SettingInfo label="context mode">
                        Full scans every selected training file before
                        calibration and fits the longest record in one context
                        if the model and memory allow it. Windowing splits long
                        records into overlapping windows without dropping answer
                        tokens.
                      </SettingInfo>
                    </div>
                    <div className="segmented two" aria-label="Context mode">
                      <button
                        type="button"
                        className={!form.fullContentContext ? "active" : ""}
                        aria-pressed={!form.fullContentContext}
                        onClick={() =>
                          setForm((current) => ({
                            ...current,
                            fullContentContext: false,
                          }))
                        }
                      >
                        Windowing
                      </button>
                      <button
                        type="button"
                        className={form.fullContentContext ? "active" : ""}
                        aria-pressed={form.fullContentContext}
                        onClick={() =>
                          setForm((current) => ({
                            ...current,
                            fullContentContext: true,
                          }))
                        }
                      >
                        Full
                      </button>
                    </div>
                    <small>
                      {form.fullContentContext
                        ? "Scan every record, then calibrate for the longest context."
                        : "Use overlapping windows for long records."}
                    </small>
                  </div>
                )}
                {stage === "fine-tune-align" && (
                  <label className="toggle-row compact">
                    <input
                      type="checkbox"
                      checked={sameDataset}
                      onChange={(event) => setSameDataset(event.target.checked)}
                    />
                    <span>
                      <b>Use the fine-tuning dataset for alignment</b>
                      <small>
                        Turn this off to choose a separate alignment dataset.
                      </small>
                    </span>
                  </label>
                )}
                {needsAlignment &&
                  !(stage === "fine-tune-align" && sameDataset) && (
                    <PathField
                      label="Alignment dataset"
                      value={form.alignmentData}
                      placeholder="Dataset folder or JSON file"
                      onChange={(alignmentData) =>
                        setForm({ ...form, alignmentData })
                      }
                      onBrowse={() =>
                        void chooseDataset(
                          "Choose an alignment dataset",
                          "alignmentData",
                        )
                      }
                    />
                  )}
                {stage === "alignment" && (
                  <PathField
                    label="Existing adapter"
                    value={form.adapter}
                    placeholder="Adapter file or directory"
                    onChange={(adapter) => setForm({ ...form, adapter })}
                    onBrowse={() => void chooseAdapter()}
                  />
                )}
                <button
                  type="button"
                  className="quiet-button data-editor-launch"
                  onClick={() =>
                    openDataEditor(
                      needsFineTune ? form.fineTuneData : form.alignmentData,
                    )
                  }
                >
                  <Icon name="edit-3" />
                  Inspect or repair training data
                </button>
              </section>

              {needsAlignment && (
                <section className="form-section">
                  <div className="section-heading">
                    <h2>Alignment</h2>
                    <p>Select the objective or leave it on automatic.</p>
                  </div>
                  <label className="field">
                    <span>Method</span>
                    <select
                      value={form.alignmentType}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          alignmentType: event.target.value as AlignmentType,
                        })
                      }
                    >
                      {[
                        "auto",
                        "dpo",
                        "ipo",
                        "simpo",
                        "orpo",
                        "cpo",
                        "kto",
                        "ppo",
                        "reinforce",
                        "rloo",
                        "grpo",
                      ].map((method) => (
                        <option key={method} value={method}>
                          {method === "auto" ? "Auto" : method.toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="toggle-row">
                    <input
                      type="checkbox"
                      checked={form.liveRollouts}
                      onChange={(event) =>
                        setForm({ ...form, liveRollouts: event.target.checked })
                      }
                    />
                    <span>
                      <b>Generate fresh answers locally</b>
                      <small>
                        The current model creates and scores its own rollouts.
                      </small>
                    </span>
                  </label>
                </section>
              )}

              <section className="form-section run-controls">
                <div className="section-heading">
                  <h2>
                    <span className="workflow-step">4</span> Calibration &amp;
                    settings
                  </h2>
                  <p>
                    Measure a short local training sample, then review the
                    fitted controls.
                  </p>
                </div>
                <div className="settings-strip">
                  <label className="toggle-row">
                    <input
                      type="checkbox"
                      checked={form.autoSettings}
                      onChange={(event) => {
                        const enabled = event.target.checked;
                        setForm((current) =>
                          enabled
                            ? applyHardwarePreset(current, hardwarePreset)
                            : { ...current, autoSettings: false },
                        );
                      }}
                    />
                    <span>
                      <b>Fit settings to this hardware</b>
                      <small>
                        Calibrate the selected model and data before training.
                        Turn off for manual control.
                      </small>
                    </span>
                  </label>
                  <div className="settings-actions">
                    {needsFineTune && (
                      <button
                        type="button"
                        className="quiet-button compact-button quick-settings-trigger"
                        disabled={form.autoSettings && !currentCalibration}
                        onClick={() => setGuidanceOpen(true)}
                        aria-haspopup="dialog"
                        title={
                          form.autoSettings && !currentCalibration
                            ? "Calibrate first to unlock Quick Settings"
                            : undefined
                        }
                      >
                        <Icon name="sliders" /> Quick Settings
                      </button>
                    )}
                    <button
                      type="button"
                      className="quiet-button compact-button"
                      disabled={form.autoSettings && !currentCalibration}
                      onClick={() => setAdvanced(!advanced)}
                      aria-expanded={advanced}
                    >
                      {advanced ? "Hide advanced" : "Advanced"}
                      <Icon name={advanced ? "chevron-up" : "chevron-down"} />
                    </button>
                  </div>
                </div>
                <div
                  className={`calibration-status${currentCalibration ? " ready" : ""}`}
                  role="status"
                >
                  <Icon
                    name={
                      calibrationBusy
                        ? "loader"
                        : currentCalibration
                          ? "check-circle"
                          : "activity"
                    }
                    size={15}
                  />
                  <span>
                    {!form.autoSettings
                      ? "Manual settings selected; calibration is optional."
                      : calibrationBusy
                        ? calibrationMessage
                        : currentCalibration
                          ? calibrationMessage ||
                            `Pilot loss fell ${currentCalibration.improvement_percent.toFixed(1)}%. Ready to train.`
                          : form.fineTuneData
                            ? "Calibrate to measure a safe learning rate on the selected data."
                            : "Select training data, then calibrate."}
                  </span>
                </div>
              </section>

              {guidanceOpen &&
                appRootRef.current &&
                createPortal(
                  <div
                    className="app-dialog-backdrop guidance-backdrop"
                    onPointerDown={(event) => {
                      if (event.target === event.currentTarget)
                        setGuidanceOpen(false);
                    }}
                  >
                    <section
                      ref={quickSettingsRef}
                      className="app-dialog guidance-card"
                      role="dialog"
                      aria-modal="true"
                      aria-labelledby="guidance-title"
                    >
                      <header className="guidance-header">
                        <div>
                          <h2 id="guidance-title">Quick Settings</h2>
                          <p>Core LoRA settings for this training run.</p>
                        </div>
                        <button
                          type="button"
                          className="dialog-close"
                          onClick={() => setGuidanceOpen(false)}
                          aria-label="Close Quick Settings"
                        >
                          <Icon name="x" />
                        </button>
                      </header>
                      <div className="guidance-context">
                        <span>
                          {form.modelSource === "custom"
                            ? "Custom model"
                            : `osCode ${form.modelVersion.toUpperCase()} · ${form.tier}`}
                        </span>
                        <span>
                          {currentDatasetSummary
                            ? `${currentDatasetSummary.fileCount} file${currentDatasetSummary.fileCount === 1 ? "" : "s"} · ${(currentDatasetSummary.totalBytes / 1024 ** 2).toFixed(1)} MiB`
                            : datasetSummaryPending
                              ? "Reading dataset size…"
                              : datasetSummaryError
                                ? "Dataset unavailable"
                                : "Select a dataset for tailored rates"}
                        </span>
                        <span>
                          {benchmarkBusy
                            ? "Measuring hardware…"
                            : `${phaseLabel(hardwarePreset.profile)} hardware fit`}
                        </span>
                      </div>
                      <div className="guidance-fields">
                        <label className="field">
                          <span>Epochs</span>
                          <select
                            value={form.iterations}
                            onChange={(event) =>
                              setForm((current) => ({
                                ...current,
                                iterations: Number(event.target.value),
                              }))
                            }
                          >
                            {[...new Set([1, 2, 3, 5, 8, form.iterations])]
                              .sort((a, b) => a - b)
                              .map((value) => (
                                <option key={value} value={value}>
                                  {value} pass{value === 1 ? "" : "es"}
                                </option>
                              ))}
                          </select>
                          <small>How many times the dataset is seen.</small>
                        </label>
                        <label className="field">
                          <span>Batch</span>
                          <select
                            value={
                              form.autoSettings
                                ? "auto"
                                : String(
                                    form.batchSize ?? hardwarePreset.batchSize,
                                  )
                            }
                            onChange={(event) => {
                              const value = event.target.value;
                              setForm((current) =>
                                value === "auto"
                                  ? applyHardwarePreset(current, hardwarePreset)
                                  : {
                                      ...(current.autoSettings
                                        ? applyHardwarePreset(
                                            current,
                                            hardwarePreset,
                                          )
                                        : current),
                                      autoSettings: false,
                                      batchSize: Number(value),
                                    },
                              );
                            }}
                          >
                            <option value="auto">
                              Auto · {hardwarePreset.batchSize}
                            </option>
                            {!form.autoSettings &&
                              form.batchSize !== null &&
                              !guidanceBatchChoices.includes(
                                form.batchSize,
                              ) && (
                                <option value={form.batchSize}>
                                  Current · {form.batchSize}
                                  {form.batchSize > hardwarePreset.batchSize
                                    ? " (above fit)"
                                    : ""}
                                </option>
                              )}
                            {guidanceBatchChoices.map((value) => (
                              <option key={value} value={value}>
                                {value}
                              </option>
                            ))}
                          </select>
                          <small>Choices stay within the fitted batch.</small>
                        </label>
                        <label className="field">
                          <span>LoRA rank</span>
                          <select
                            value={
                              form.autoSettings
                                ? "auto"
                                : String(form.rank ?? hardwarePreset.rank)
                            }
                            onChange={(event) => {
                              const value = event.target.value;
                              setForm((current) =>
                                value === "auto"
                                  ? applyHardwarePreset(current, hardwarePreset)
                                  : {
                                      ...(current.autoSettings
                                        ? applyHardwarePreset(
                                            current,
                                            hardwarePreset,
                                          )
                                        : current),
                                      autoSettings: false,
                                      rank: Number(value),
                                    },
                              );
                            }}
                          >
                            <option value="auto">
                              Auto · {hardwarePreset.rank}
                            </option>
                            {!form.autoSettings &&
                              form.rank !== null &&
                              !guidanceRankChoices.includes(form.rank) && (
                                <option value={form.rank}>
                                  Current · {form.rank}
                                  {form.rank > hardwarePreset.rank
                                    ? " (above fit)"
                                    : ""}
                                </option>
                              )}
                            {guidanceRankChoices.map((value) => (
                              <option key={value} value={value}>
                                {value}
                              </option>
                            ))}
                          </select>
                          <small>Adapter capacity and memory use.</small>
                        </label>
                        <label className="field">
                          <span>Learning rate</span>
                          <select
                            value={rateSelection}
                            onChange={(event) => selectRate(event.target.value)}
                            disabled={datasetSummaryPending}
                          >
                            <option value="backend">
                              Backend default · 1.00e-5
                            </option>
                            {rateSelection === "previous" && (
                              <option value="previous">
                                {currentCalibration
                                  ? "Calibrated setting"
                                  : "Previous setting"}{" "}
                                · {form.learningRate?.toExponential(2)}
                              </option>
                            )}
                            {rateOptions.map((option) => (
                              <option key={option.pace} value={option.pace}>
                                {option.label} · {option.rate.toExponential(2)}
                              </option>
                            ))}
                          </select>
                          <small>
                            {datasetSummaryPending
                              ? "Reading the dataset to tailor learning rates."
                              : "Sets the update pace. Rate alone does not affect memory."}
                          </small>
                        </label>
                      </div>
                      {datasetSummaryError && (
                        <p className="guidance-error">{datasetSummaryError}</p>
                      )}
                      <p className="guidance-note">
                        Calibration checks a short training sample and selects a
                        rate with decreasing pilot loss. Full training can still
                        vary across examples; Auto stop watches for sustained
                        loss increases.
                      </p>
                      <footer className="guidance-footer">
                        <button
                          type="button"
                          className="primary"
                          onClick={() => setGuidanceOpen(false)}
                        >
                          Done
                        </button>
                      </footer>
                    </section>
                  </div>,
                  appRootRef.current,
                )}

              {advanced && (
                <div className="advanced-panel">
                  <div className="advanced-toolbar">
                    <div className="advanced-auto-note">
                      <Icon name="cpu" />
                      <span>
                        {form.autoSettings
                          ? benchmarkBusy
                            ? "Running local inference to fit training settings to this hardware…"
                            : `${phaseLabel(hardwarePreset.profile)} profile selected${benchmarkDevices.length ? ` on ${benchmarkDevices.join(", ")}` : ""}. The recommended values are visible below; turn fitting off to edit them.`
                          : "Custom settings are active. Reset restores the recommended hardware profile."}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="quiet-button compact-button advanced-reset"
                      onClick={() =>
                        setForm((current) =>
                          resetAdvancedValues(current, hardwarePreset),
                        )
                      }
                      title="Restore recommended advanced settings"
                    >
                      <Icon name="rotate-ccw" />
                      Reset
                    </button>
                  </div>

                  <section className="advanced-group">
                    <div className="advanced-heading">
                      <h3>Optimization</h3>
                      <p>Update counts, batching and learning rates.</p>
                    </div>
                    <div className="advanced-grid">
                      {needsFineTune && (
                        <label className="field">
                          <span>Fine-tune epochs</span>
                          <input
                            type="number"
                            min="1"
                            value={form.iterations}
                            onChange={(event) =>
                              setForm({
                                ...form,
                                iterations: Number(event.target.value),
                              })
                            }
                          />
                        </label>
                      )}
                      {needsAlignment && (
                        <label className="field">
                          <span>Alignment iterations</span>
                          <input
                            type="number"
                            min="1"
                            value={form.alignmentIterations}
                            onChange={(event) =>
                              setForm({
                                ...form,
                                alignmentIterations: Number(event.target.value),
                              })
                            }
                          />
                        </label>
                      )}
                      <label className="field">
                        <span>Optimizer</span>
                        <select
                          value={form.optimizer}
                          onChange={(event) =>
                            setForm({
                              ...form,
                              optimizer: event.target
                                .value as TrainingRequest["optimizer"],
                            })
                          }
                        >
                          <option value="auto">Auto</option>
                          <option value="sgd">SGD</option>
                          <option value="adamw">AdamW</option>
                        </select>
                      </label>
                      <NumberField
                        label="Batch size"
                        value={form.batchSize}
                        min={1}
                        disabled={form.autoSettings}
                        placeholder="1"
                        onChange={(batchSize) =>
                          setForm({ ...form, batchSize })
                        }
                      />
                      <NumberField
                        label="Gradient accumulation"
                        value={form.gradientAccumulationSteps}
                        min={1}
                        placeholder="1"
                        onChange={(gradientAccumulationSteps) =>
                          setForm({ ...form, gradientAccumulationSteps })
                        }
                      />
                      <NumberField
                        label="Training token limit"
                        value={form.maxSeqLength}
                        min={32}
                        disabled={form.autoSettings || form.fullContentContext}
                        placeholder="64"
                        hint="Maximum prompt + answer tokens per training example."
                        onChange={(maxSeqLength) =>
                          setForm({ ...form, maxSeqLength })
                        }
                      />
                      {needsFineTune && (
                        <label className="field">
                          <span>Fine-tune learning rate</span>
                          <select
                            value={rateSelection}
                            onChange={(event) => selectRate(event.target.value)}
                            disabled={datasetSummaryPending}
                          >
                            <option value="backend">
                              Backend default · 1.00e-5
                            </option>
                            {rateSelection === "previous" && (
                              <option value="previous">
                                Previous setting ·{" "}
                                {form.learningRate?.toExponential(2)}
                              </option>
                            )}
                            {rateOptions.map((option) => (
                              <option key={option.pace} value={option.pace}>
                                {option.label} · {option.rate.toExponential(2)}
                              </option>
                            ))}
                          </select>
                          <small>
                            {datasetSummaryPending
                              ? "Reading the dataset to tailor learning rates."
                              : "Adaptive update pace; memory is fitted by batch, rank and context."}
                          </small>
                        </label>
                      )}
                      {needsAlignment && (
                        <NumberField
                          label="Alignment learning rate"
                          value={form.alignmentLearningRate}
                          min={0}
                          step="any"
                          placeholder="Backend default"
                          onChange={(alignmentLearningRate) =>
                            setForm({ ...form, alignmentLearningRate })
                          }
                        />
                      )}
                      <NumberField
                        label="Training seed"
                        value={form.seed}
                        min={0}
                        placeholder="0"
                        onChange={(seed) => setForm({ ...form, seed })}
                      />
                    </div>
                  </section>

                  {needsFineTune && (
                    <section className="advanced-group">
                      <div className="advanced-heading">
                        <h3>LoRA adapter</h3>
                        <p>Capacity and trainable projections.</p>
                      </div>
                      <div className="advanced-grid">
                        <NumberField
                          label="Rank"
                          value={form.rank}
                          min={1}
                          disabled={form.autoSettings}
                          placeholder="2"
                          onChange={(rank) => setForm({ ...form, rank })}
                        />
                        <NumberField
                          label="Scale / alpha"
                          value={form.scale}
                          min={0}
                          step="any"
                          placeholder="4"
                          onChange={(scale) => setForm({ ...form, scale })}
                        />
                        <NumberField
                          label="Layers to adapt"
                          value={form.numLayers}
                          min={1}
                          disabled={form.autoSettings}
                          placeholder="1"
                          onChange={(numLayers) =>
                            setForm({ ...form, numLayers })
                          }
                        />
                        <NumberField
                          label="Dropout · MLX"
                          value={form.dropout}
                          min={0}
                          max={0.999}
                          step="any"
                          placeholder="0"
                          onChange={(dropout) => setForm({ ...form, dropout })}
                        />
                      </div>
                      <div className="module-picker">
                        <span>Target projections</span>
                        <div>
                          <button
                            type="button"
                            className={
                              form.targetModules.length ? "" : "active"
                            }
                            disabled={form.autoSettings}
                            onClick={() =>
                              setForm({ ...form, targetModules: [] })
                            }
                          >
                            Automatic
                          </button>
                          {targetModules.map((target) => {
                            const selected = form.targetModules.includes(
                              target.value,
                            );
                            return (
                              <button
                                type="button"
                                key={target.value}
                                className={selected ? "active" : ""}
                                disabled={form.autoSettings}
                                title={target.value}
                                onClick={() =>
                                  setForm({
                                    ...form,
                                    targetModules: selected
                                      ? form.targetModules.filter(
                                          (value) => value !== target.value,
                                        )
                                      : [...form.targetModules, target.value],
                                  })
                                }
                              >
                                {target.label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </section>
                  )}

                  {needsFineTune && (
                    <section className="advanced-group">
                      <div className="advanced-heading">
                        <h3>Media</h3>
                        <p>
                          Local VLM preprocessing. Leave image size empty to use
                          the model processor default.
                        </p>
                      </div>
                      <div className="advanced-grid">
                        <NumberField
                          label="Image width"
                          value={form.imageWidth}
                          min={16}
                          placeholder="Model default"
                          onChange={(imageWidth) =>
                            setForm({ ...form, imageWidth })
                          }
                        />
                        <NumberField
                          label="Image height"
                          value={form.imageHeight}
                          min={16}
                          placeholder="Model default"
                          onChange={(imageHeight) =>
                            setForm({ ...form, imageHeight })
                          }
                        />
                        <NumberField
                          label="Video frames per second"
                          value={form.videoFps}
                          min={0.01}
                          step="any"
                          onChange={(videoFps) =>
                            videoFps !== null && setForm({ ...form, videoFps })
                          }
                        />
                        <NumberField
                          label="Maximum video frames"
                          value={form.videoMaxFrames}
                          min={2}
                          onChange={(videoMaxFrames) =>
                            videoMaxFrames !== null &&
                            setForm({ ...form, videoMaxFrames })
                          }
                        />
                        <NumberField
                          label="Assistant token ID"
                          value={form.assistantTokenId}
                          min={0}
                          placeholder="Detect automatically"
                          onChange={(assistantTokenId) =>
                            setForm({ ...form, assistantTokenId })
                          }
                        />
                      </div>
                    </section>
                  )}

                  {needsAlignment && (
                    <section className="advanced-group">
                      <div className="advanced-heading">
                        <h3>Alignment & rollouts</h3>
                        <p>Objective weights and local answer sampling.</p>
                      </div>
                      <div className="advanced-grid">
                        <NumberField
                          label="Beta"
                          value={form.alignmentBeta}
                          min={0}
                          step="any"
                          onChange={(alignmentBeta) =>
                            alignmentBeta !== null &&
                            setForm({ ...form, alignmentBeta })
                          }
                        />
                        <NumberField
                          label="SimPO gamma"
                          value={form.alignmentGamma}
                          min={0}
                          step="any"
                          onChange={(alignmentGamma) =>
                            alignmentGamma !== null &&
                            setForm({ ...form, alignmentGamma })
                          }
                        />
                        <NumberField
                          label="PPO clip"
                          value={form.ppoClip}
                          min={0}
                          step="any"
                          onChange={(ppoClip) =>
                            ppoClip !== null && setForm({ ...form, ppoClip })
                          }
                        />
                        <NumberField
                          label="Answers per prompt"
                          value={form.rolloutsPerPrompt}
                          min={1}
                          onChange={(rolloutsPerPrompt) =>
                            rolloutsPerPrompt !== null &&
                            setForm({ ...form, rolloutsPerPrompt })
                          }
                        />
                        <NumberField
                          label="Max new tokens"
                          value={form.rolloutMaxTokens}
                          min={1}
                          onChange={(rolloutMaxTokens) =>
                            rolloutMaxTokens !== null &&
                            setForm({ ...form, rolloutMaxTokens })
                          }
                        />
                        <NumberField
                          label="Temperature"
                          value={form.rolloutTemperature}
                          min={0}
                          step="any"
                          onChange={(rolloutTemperature) =>
                            rolloutTemperature !== null &&
                            setForm({ ...form, rolloutTemperature })
                          }
                        />
                        <NumberField
                          label="Top-p"
                          value={form.rolloutTopP}
                          min={0.000001}
                          max={1}
                          step="any"
                          onChange={(rolloutTopP) =>
                            rolloutTopP !== null &&
                            setForm({ ...form, rolloutTopP })
                          }
                        />
                        <NumberField
                          label="Rollout seed"
                          value={form.rolloutSeed}
                          min={0}
                          onChange={(rolloutSeed) =>
                            rolloutSeed !== null &&
                            setForm({ ...form, rolloutSeed })
                          }
                        />
                      </div>
                    </section>
                  )}

                  {needsFineTune && (
                    <section className="advanced-group">
                      <div className="advanced-heading">
                        <h3>Saving & evaluation · MLX</h3>
                        <p>Checkpoint, reporting and validation cadence.</p>
                      </div>
                      <div className="advanced-grid">
                        <NumberField
                          label="Save every"
                          value={form.saveEvery}
                          min={1}
                          placeholder="10"
                          onChange={(saveEvery) =>
                            setForm({ ...form, saveEvery })
                          }
                        />
                        <NumberField
                          label="Report every"
                          value={form.stepsPerReport}
                          min={1}
                          placeholder="1"
                          onChange={(stepsPerReport) =>
                            setForm({ ...form, stepsPerReport })
                          }
                        />
                        <NumberField
                          label="Evaluate every"
                          value={form.stepsPerEval}
                          min={1}
                          placeholder="10"
                          onChange={(stepsPerEval) =>
                            setForm({ ...form, stepsPerEval })
                          }
                        />
                        <NumberField
                          label="Validation batches"
                          value={form.validationBatches}
                          min={-1}
                          placeholder="1"
                          onChange={(validationBatches) =>
                            setForm({ ...form, validationBatches })
                          }
                        />
                      </div>
                      <label className="toggle-row compact">
                        <input
                          type="checkbox"
                          checked={form.maskPrompt}
                          onChange={(event) =>
                            setForm({
                              ...form,
                              maskPrompt: event.target.checked,
                            })
                          }
                        />
                        <span>
                          <b>Mask prompt tokens</b>
                          <small>
                            Train the adapter on answer tokens only.
                          </small>
                        </span>
                      </label>
                      <label className="toggle-row compact">
                        <input
                          type="checkbox"
                          checked={form.gradientCheckpoint}
                          onChange={(event) =>
                            setForm({
                              ...form,
                              gradientCheckpoint: event.target.checked,
                            })
                          }
                        />
                        <span>
                          <b>Gradient checkpointing</b>
                          <small>
                            Recompute MLX activations to reduce training memory.
                          </small>
                        </span>
                      </label>
                    </section>
                  )}

                  <section className="advanced-group">
                    <div className="advanced-heading">
                      <h3>Engine runtime</h3>
                      <p>GGUF execution and explicit multi-GPU placement.</p>
                    </div>
                    <div className="advanced-grid">
                      <NumberField
                        label="GGUF microbatch"
                        value={form.ggufBatchSize}
                        min={1}
                        disabled={form.autoSettings}
                        placeholder="8"
                        onChange={(ggufBatchSize) =>
                          setForm({ ...form, ggufBatchSize })
                        }
                      />
                      <NumberField
                        label="GGUF CPU threads"
                        value={form.ggufThreads}
                        min={1}
                        disabled={form.autoSettings}
                        placeholder="2"
                        onChange={(ggufThreads) =>
                          setForm({ ...form, ggufThreads })
                        }
                      />
                      <NumberField
                        label="MLX distributed workers"
                        value={form.distributedWorkers}
                        min={0}
                        placeholder="Auto"
                        onChange={(distributedWorkers) =>
                          setForm({ ...form, distributedWorkers })
                        }
                      />
                      <NumberField
                        label="Main GPU index"
                        value={form.mainGpu}
                        min={0}
                        placeholder="0"
                        onChange={(mainGpu) => setForm({ ...form, mainGpu })}
                      />
                      <label className="field">
                        <span>GGUF split mode</span>
                        <select
                          value={form.splitMode}
                          onChange={(event) =>
                            setForm({
                              ...form,
                              splitMode: event.target
                                .value as TrainingRequest["splitMode"],
                            })
                          }
                        >
                          <option value="auto">Auto</option>
                          <option value="none">None</option>
                          <option value="layer">Layer</option>
                          <option value="row">Row</option>
                          <option value="tensor">Tensor</option>
                        </select>
                      </label>
                      <label className="field">
                        <span>Tensor split</span>
                        <input
                          value={form.tensorSplit}
                          placeholder="Example: 3,1"
                          spellCheck={false}
                          onChange={(event) =>
                            setForm({
                              ...form,
                              tensorSplit: event.target.value,
                            })
                          }
                        />
                      </label>
                      <label className="field advanced-wide">
                        <span>Device order</span>
                        <input
                          value={form.devices}
                          placeholder="Example: CUDA0, CUDA1"
                          spellCheck={false}
                          onChange={(event) =>
                            setForm({ ...form, devices: event.target.value })
                          }
                        />
                      </label>
                    </div>
                  </section>
                </div>
              )}
            </fieldset>

            <div className="auto-stop-panel">
              <div className="setting-heading">
                <strong>Auto stop</strong>
                <SettingInfo label="Auto stop">
                  Watches average fine-tuning loss after five minutes of recent
                  readings. It stops only when four consecutive loss windows
                  rise on every reporting device, regardless of total dataset
                  progress. It saves and verifies a checkpoint before stopping.
                  Training loss can be noisy, so turn this off if you prefer to
                  decide from the graph.
                </SettingInfo>
              </div>
              <label className="toggle-row compact">
                <input
                  type="checkbox"
                  checked={
                    autoStopAvailable &&
                    (active ? Boolean(active.autoStopEnabled) : form.autoStop)
                  }
                  disabled={
                    !autoStopAvailable ||
                    autoStopBusy ||
                    active?.status === "stopping"
                  }
                  onChange={(event) =>
                    void changeAutoStop(event.target.checked)
                  }
                />
                <span>
                  <b>
                    {autoStopAvailable &&
                    (active?.autoStopEnabled || (!active && form.autoStop))
                      ? "On"
                      : "Off"}
                  </b>
                  <small>
                    {!autoStopAvailable
                      ? "Available for fine-tuning runs"
                      : active?.autoStopMessage ||
                        "Watch for sustained rising loss"}
                  </small>
                </span>
              </label>
            </div>

            <footer className="panel-footer">
              {active ? (
                <div className="active-run-controls">
                  {(active.status === "running" ||
                    active.status === "pausing") && (
                    <button
                      className={
                        "primary-button" +
                        (sessionControl === "pausing" ||
                        active.status === "pausing"
                          ? " busy-control"
                          : "")
                      }
                      disabled={
                        Boolean(sessionControl) || active.status === "pausing"
                      }
                      onClick={() => void pauseTraining(active.id)}
                    >
                      <Icon
                        name={
                          sessionControl === "pausing" ||
                          active.status === "pausing"
                            ? "loader"
                            : "pause"
                        }
                      />
                      {sessionControl === "pausing" ||
                      active.status === "pausing"
                        ? "Saving, then pausing…"
                        : "Pause training"}
                    </button>
                  )}
                  {active.status === "paused" && (
                    <button
                      className="primary-button"
                      disabled={Boolean(sessionControl)}
                      onClick={() => void resumeTraining(active.id)}
                    >
                      <Icon
                        name={sessionControl === "resuming" ? "loader" : "play"}
                      />
                      {sessionControl === "resuming"
                        ? "Resuming…"
                        : "Resume training"}
                    </button>
                  )}
                  {(active.status === "running" ||
                    active.status === "paused") && (
                    <button
                      type="button"
                      className="quiet-button"
                      disabled={Boolean(sessionControl)}
                      onClick={() => void saveCheckpoint(active.id)}
                    >
                      <Icon
                        name={sessionControl === "saving" ? "loader" : "save"}
                      />
                      {sessionControl === "saving"
                        ? "Requesting…"
                        : "Save checkpoint"}
                    </button>
                  )}
                  <button
                    className="danger-button"
                    disabled={
                      Boolean(sessionControl) || active.status === "stopping"
                    }
                    onClick={() => void stopTraining(active.id)}
                  >
                    <Icon
                      name={sessionControl === "stopping" ? "loader" : "square"}
                    />
                    {sessionControl === "stopping" ||
                    active.status === "stopping"
                      ? "Stopping…"
                      : "Stop training"}
                  </button>
                </div>
              ) : (
                <button
                  className="primary-button start-button"
                  disabled={
                    starting ||
                    (!calibrationBusy &&
                      form.autoSettings &&
                      (benchmarkBusy || datasetSummaryPending))
                  }
                  onClick={() => {
                    if (calibrationBusy) {
                      setCalibrationMessage("Stopping calibration…");
                      void window.osai.cancelAutoCalibration();
                    } else if (
                      form.autoSettings &&
                      needsFineTune &&
                      !currentCalibration
                    )
                      void calibrate();
                    else void startTraining();
                  }}
                >
                  <Icon
                    name={
                      starting || benchmarkBusy || calibrationBusy
                        ? "loader"
                        : form.autoSettings &&
                            needsFineTune &&
                            !currentCalibration
                          ? "activity"
                          : "play"
                    }
                  />
                  {starting
                    ? "Starting…"
                    : calibrationBusy
                      ? "Cancel calibration"
                      : benchmarkBusy
                        ? "Measuring hardware…"
                        : form.autoSettings &&
                            needsFineTune &&
                            !currentCalibration
                          ? "Calibrate"
                          : "Start training"}
                </button>
              )}
            </footer>
          </aside>

          <section className="session-workspace">
            <header className="session-toolbar">
              <div className="session-toolbar-title">
                <Icon name="activity" />
                <span>Training sessions</span>
              </div>
              <div className="session-toolbar-actions">
                <button
                  type="button"
                  className={
                    "quiet-button compact-button wiki-open-button " +
                    (dataEditorActive ? "active" : "")
                  }
                  onClick={() => openDataEditor()}
                >
                  <Icon name="edit-3" />
                  Data editor
                </button>
                <button
                  type="button"
                  className={
                    "quiet-button compact-button wiki-open-button " +
                    (wikiActive ? "active" : "")
                  }
                  onClick={() => {
                    setWikiOpen(true);
                    setWikiActive(true);
                    setDataEditorActive(false);
                    setSessionMenu(null);
                  }}
                >
                  <Icon name="book-open" />
                  Wiki
                </button>
              </div>
            </header>

            {(sessions.length > 0 || wikiOpen || dataEditorOpen) && (
              <div className="session-tabs" role="tablist">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={
                      "session-tab " +
                      (!wikiActive &&
                      !dataEditorActive &&
                      selected?.id === session.id
                        ? "active "
                        : "") +
                      session.status
                    }
                  >
                    <button
                      type="button"
                      className="session-tab-select"
                      role="tab"
                      aria-selected={
                        !wikiActive &&
                        !dataEditorActive &&
                        selected?.id === session.id
                      }
                      onClick={() => {
                        setWikiActive(false);
                        setDataEditorActive(false);
                        selectionClearedRef.current = false;
                        setSelectedId(session.id);
                        setSessionMenu(null);
                      }}
                    >
                      <span>
                        <b>{session.name}</b>
                        <small>{statusLabel(session.status)}</small>
                      </span>
                    </button>
                    <button
                      type="button"
                      className="session-tab-more"
                      aria-label={`Options for ${session.name}`}
                      aria-expanded={sessionMenu?.id === session.id}
                      onClick={(event) => {
                        event.stopPropagation();
                        toggleSessionMenu(session.id, event.currentTarget);
                      }}
                    >
                      <Icon name="more-horizontal" />
                    </button>
                  </div>
                ))}
                {dataEditorOpen && (
                  <div
                    className={
                      "session-tab wiki-session-tab " +
                      (dataEditorActive ? "active" : "")
                    }
                  >
                    <button
                      type="button"
                      className="session-tab-select"
                      role="tab"
                      aria-selected={dataEditorActive}
                      onClick={() => {
                        setDataEditorActive(true);
                        setWikiActive(false);
                        setSessionMenu(null);
                      }}
                    >
                      <Icon name="edit-3" />
                      <span>
                        <b>Data editor</b>
                        <small>Inspect and repair</small>
                      </span>
                    </button>
                    <button
                      type="button"
                      className="session-tab-more wiki-tab-close"
                      aria-label="Close Data editor"
                      title="Close Data editor"
                      onClick={() => {
                        setDataEditorOpen(false);
                        setDataEditorActive(false);
                      }}
                    >
                      <Icon name="x" size={15} />
                    </button>
                  </div>
                )}
                {wikiOpen && (
                  <div
                    className={
                      "session-tab wiki-session-tab " +
                      (wikiActive ? "active" : "")
                    }
                  >
                    <button
                      type="button"
                      className="session-tab-select"
                      role="tab"
                      aria-selected={wikiActive}
                      onClick={() => {
                        setWikiActive(true);
                        setDataEditorActive(false);
                        setSessionMenu(null);
                      }}
                    >
                      <Icon name="book-open" />
                      <span>
                        <b>Wiki</b>
                        <small>Training guide</small>
                      </span>
                    </button>
                    <button
                      type="button"
                      className="session-tab-more wiki-tab-close"
                      aria-label="Close Wiki"
                      title="Close Wiki"
                      onClick={() => {
                        setWikiOpen(false);
                        setWikiActive(false);
                      }}
                    >
                      <Icon name="x" size={15} />
                    </button>
                  </div>
                )}
              </div>
            )}

            {sessionMenu &&
              sessionMenuSession &&
              createPortal(
                <div
                  className="session-tab-menu"
                  role="menu"
                  aria-label={`Options for ${sessionMenuSession.name}`}
                  style={{ top: sessionMenu.top, left: sessionMenu.left }}
                >
                  <button
                    type="button"
                    role="menuitem"
                    disabled={
                      !sessionMenuSession.request ||
                      [
                        "queued",
                        "running",
                        "pausing",
                        "paused",
                        "stopping",
                      ].includes(sessionMenuSession.status)
                    }
                    title={
                      [
                        "queued",
                        "running",
                        "pausing",
                        "paused",
                        "stopping",
                      ].includes(sessionMenuSession.status)
                        ? "Stop this session before restarting it"
                        : sessionMenuSession.request
                          ? "Clear this pipeline and restart with its saved settings"
                          : "This session does not contain restorable settings"
                    }
                    onClick={() => {
                      setRestartCandidate(sessionMenuSession);
                      setSessionMenu(null);
                    }}
                  >
                    <Icon name="rotate-ccw" />
                    Restart session
                  </button>
                  <button
                    type="button"
                    className="danger"
                    role="menuitem"
                    disabled={[
                      "queued",
                      "running",
                      "pausing",
                      "paused",
                      "stopping",
                    ].includes(sessionMenuSession.status)}
                    title={
                      [
                        "queued",
                        "running",
                        "pausing",
                        "paused",
                        "stopping",
                      ].includes(sessionMenuSession.status)
                        ? "Stop this session before deleting it"
                        : "Move this session to Trash"
                    }
                    onClick={() => {
                      setDeleteCandidate(sessionMenuSession);
                      setSessionMenu(null);
                    }}
                  >
                    <Icon name="trash-2" />
                    Delete session
                  </button>
                </div>,
                document.querySelector(".app") || document.body,
              )}

            {dataEditorActive ? (
              <DataEditor
                initialSource={dataEditorSource}
                onUseForFineTune={(source, tokenLimit) =>
                  useEditedDataset("fineTuneData", source, tokenLimit)
                }
                onUseForAlignment={(source, tokenLimit) =>
                  useEditedDataset("alignmentData", source, tokenLimit)
                }
                onError={setNotice}
              />
            ) : wikiActive ? (
              <TrainingWiki />
            ) : selected ? (
              <div className="session-detail">
                <header className="session-summary">
                  <div>
                    <h2>{selected.name}</h2>
                    <p>
                      {phaseLabel(selected.phase)} ·{" "}
                      {friendlyTime(selected.startedAt || selected.createdAt)}
                    </p>
                  </div>
                </header>
                <div
                  className="session-status-details"
                  aria-label="Training status"
                >
                  <div>
                    <span>Current step</span>
                    <strong>{selected.message}</strong>
                  </div>
                  {selected.autoSettingsSummary && (
                    <div>
                      <span>Auto settings</span>
                      <strong>{selected.autoSettingsSummary}</strong>
                    </div>
                  )}
                  {selected.adjustment && (
                    <div>
                      <span>Memory adjustment</span>
                      <strong>{selected.adjustment}</strong>
                    </div>
                  )}
                  {selected.checkpointStatus && (
                    <div>
                      <span>Latest checkpoint</span>
                      <strong>
                        {selected.checkpointStatus === "requested"
                          ? selected.status === "paused"
                            ? "Requested; resume training to finish saving"
                            : "Waiting for the next safe training step"
                          : selected.checkpointStatus === "failed"
                            ? "Save failed; check the live output"
                            : `Adapter saved ${friendlyTime(selected.checkpointSavedAt || "")}`}
                      </strong>
                    </div>
                  )}
                  {selected.checkpointModelPath && (
                    <div>
                      <span>Reusable model</span>
                      <strong>Latest checkpoint model is ready</strong>
                    </div>
                  )}
                </div>
                {selected.checkpointModelPath &&
                  selected.status !== "completed" && (
                    <div className="session-artifacts">
                      <span>Saved adapter and reusable model</span>
                      <button
                        type="button"
                        onClick={() =>
                          setForm((current) => ({
                            ...current,
                            modelSource: "custom",
                            customModelFolder: selected.checkpointModelPath!,
                          }))
                        }
                      >
                        Use saved model
                      </button>
                    </div>
                  )}
                {selected.error && (
                  <div className="session-error">
                    <Icon name="alert-triangle" />
                    <span>{selected.error}</span>
                    {selected.status === "failed" && selected.request && (
                      <button
                        type="button"
                        onClick={() => setRestartCandidate(selected)}
                        title="Clear this pipeline and restart with its saved settings"
                      >
                        <Icon name="rotate-ccw" />
                        Restart session
                      </button>
                    )}
                  </div>
                )}
                {selected.status === "completed" && sessionArtifacts && (
                  <div className="session-artifacts">
                    <span>
                      {sessionArtifacts.adapterDirectory
                        ? "LoRA adapter saved"
                        : "No adapter was published"}
                      {sessionArtifacts.mergedModel
                        ? " · Reusable merged model saved"
                        : ""}
                    </span>
                    {sessionArtifacts.mergedModel && (
                      <button
                        type="button"
                        onClick={() =>
                          setForm((current) => ({
                            ...current,
                            modelSource: "custom",
                            customModelFolder: sessionArtifacts.mergedModel!,
                          }))
                        }
                      >
                        Use merged model
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() =>
                        void window.osai
                          .openSessionArtifacts(selected.id)
                          .catch((error) => setNotice(readableError(error)))
                      }
                    >
                      Open outputs
                    </button>
                  </div>
                )}
                <div className="training-monitor">
                  <section className="training-monitor-panel">
                    <div className="log-heading">
                      <span>Live output</span>
                    </div>
                    <div className="training-log-shell">
                      <pre
                        className="training-log"
                        ref={logRef}
                        onScroll={handleLogScroll}
                      >
                        {log || "Waiting for osAi output…"}
                      </pre>
                      {showLogLatest && (
                        <button
                          type="button"
                          className="log-latest-button"
                          onClick={jumpToLatestLog}
                          aria-label="Jump to latest output"
                          title="Jump to latest output"
                        >
                          <Icon name="arrow-down" size={15} />
                        </button>
                      )}
                    </div>
                  </section>
                  <section className="training-monitor-panel loss-monitor">
                    <div className="log-heading loss-heading">
                      <span>Training loss</span>
                      <button
                        type="button"
                        className="quiet-button"
                        disabled={!metrics.length}
                        onClick={() =>
                          void window.osai
                            .exportSessionMetrics(selected.id)
                            .catch((error) => setNotice(readableError(error)))
                        }
                      >
                        <Icon name="download" size={14} /> Save CSV
                      </button>
                    </div>
                    <div
                      className="loss-axis-tabs"
                      role="tablist"
                      aria-label="Loss chart horizontal axis"
                    >
                      {(
                        [
                          ["progress", "Progress"],
                          ["time", "Time"],
                          ["steps", "Steps"],
                        ] as const
                      ).map(([axis, label]) => (
                        <button
                          key={axis}
                          type="button"
                          role="tab"
                          aria-selected={lossAxis === axis}
                          className={lossAxis === axis ? "active" : ""}
                          onClick={() => setLossAxis(axis)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                    <div className="loss-chart-shell">
                      <LossChart metrics={metrics} axis={lossAxis} />
                    </div>
                  </section>
                </div>
                <p className="detached-note">
                  <Icon name="power" />
                  Training continues in a detached local worker when this app is
                  closed. Use Stop to end it.
                </p>
              </div>
            ) : (
              <div className="empty-session">
                <Icon name="activity" size={24} />
                <h2>No training session yet</h2>
                <p>
                  Choose a model and dataset on the left, then start training.
                  Progress and output will appear here.
                </p>
              </div>
            )}
          </section>
        </main>
      ) : (
        <main className="backend-onboarding">
          <section className="backend-onboarding-content" aria-live="polite">
            <img src={osAiIcon} alt="" aria-hidden="true" />
            <h1>Set up osAi</h1>
            <p>
              Download osAi CLI, install its Python packages, and compile
              llama.cpp for this computer. Setup uses the available GPU
              toolchain when possible.
            </p>
            <button
              className="primary-button onboarding-download"
              disabled={
                backend === null || backendChecking || backendInstallBusy
              }
              onClick={() => void downloadBackend()}
            >
              <Icon
                name={
                  backend === null || backendChecking || backendInstallBusy
                    ? "loader"
                    : "download"
                }
              />
              {backend === null || backendChecking
                ? "Checking setup…"
                : backendInstallLabel}
            </button>
            <div className="onboarding-state">
              <span>
                {backend === null || backendChecking
                  ? "Looking for an existing osAi CLI installation"
                  : backendInstall.state === "error"
                    ? "Setup failed. Open the notification above for details, then try again."
                    : backendInstall.state !== "idle"
                      ? backendInstall.message
                      : backend.message}
              </span>
            </div>
          </section>
        </main>
      )}

      <footer
        className={
          "statusbar " +
          (active
            ? "active"
            : backendReady
              ? "idle"
              : backendInstallBusy
                ? "installing"
                : "setup-needed")
        }
      >
        <div className="progress-copy">
          <Icon
            name={
              calibrationBusy || benchmarkBusy || active
                ? "activity"
                : backendReady
                  ? "check"
                  : "download"
            }
          />
          <span>
            {calibrationBusy
              ? `Calibration · ${calibrationMessage}`
              : starting
                ? "Preparing and checking the selected training data…"
                : active
                  ? active.name + " · " + active.message
                  : benchmarkBusy
                    ? "Measuring the model and available hardware…"
                    : backendReady
                      ? "Ready"
                      : backend === null || backendChecking
                        ? "Checking osAi CLI"
                        : backendInstall.state === "error"
                          ? "osAi CLI setup failed"
                          : backend?.message || backendInstall.message}
          </span>
        </div>
        <div className="status-track">
          <span
            className={footerIndeterminate ? "indeterminate" : ""}
            style={{ width: `${Math.max(0, Math.min(100, footerPercent))}%` }}
          />
        </div>
        <span className="status-percent">
          {calibrationBusy || starting || benchmarkBusy
            ? "Working"
            : active
              ? `${active.progress.toFixed(1)}%`
              : backendReady
                ? "Local"
                : typeof backendInstall.percent === "number"
                  ? `${backendInstall.percent}%`
                  : "Setup"}
        </span>
      </footer>

      {hardwareSheetOpen && (
        <div
          className="hardware-sheet-backdrop"
          role="presentation"
          onMouseDown={(event) =>
            event.target === event.currentTarget && setHardwareSheetOpen(false)
          }
        >
          <aside
            className="hardware-sheet"
            role="dialog"
            aria-modal="true"
            aria-labelledby="hardware-sheet-title"
          >
            <header>
              <div>
                <h2 id="hardware-sheet-title">Hardware</h2>
                <p>Live local readings refresh every five seconds.</p>
              </div>
              <button
                type="button"
                className="dialog-close"
                aria-label="Close hardware"
                onClick={() => setHardwareSheetOpen(false)}
              >
                <Icon name="x" />
              </button>
            </header>
            {hardwareSnapshot ? (
              <>
                <section className="hardware-card">
                  <div className="hardware-card-title">
                    <Icon name="cpu" />
                    <strong>{hardwareSnapshot.cpu.name}</strong>
                    <span>CPU</span>
                  </div>
                  <dl>
                    <div>
                      <dt>Load</dt>
                      <dd>
                        {hardwareSnapshot.cpu.utilizationPercent === null
                          ? "Measuring…"
                          : `${hardwareSnapshot.cpu.utilizationPercent.toFixed(0)}%`}
                      </dd>
                    </div>
                    <div>
                      <dt>Memory</dt>
                      <dd>
                        {memoryLabel(hardwareSnapshot.cpu.memoryUsedBytes)} /{" "}
                        {memoryLabel(hardwareSnapshot.cpu.memoryTotalBytes)}
                      </dd>
                    </div>
                    <div>
                      <dt>Logical cores</dt>
                      <dd>{hardwareSnapshot.cpu.logicalCores}</dd>
                    </div>
                    <div>
                      <dt>Temperature</dt>
                      <dd>
                        {hardwareSnapshot.cpu.temperatureC === null
                          ? "Unavailable"
                          : `${hardwareSnapshot.cpu.temperatureC} °C`}
                      </dd>
                    </div>
                  </dl>
                </section>
                <div className="hardware-divider">
                  GPUs · {hardwareSnapshot.gpus.length}
                </div>
                {hardwareSnapshot.gpus.length ? (
                  hardwareSnapshot.gpus.map((gpu) => (
                    <section className="hardware-card" key={gpu.id}>
                      <div className="hardware-card-title">
                        <Icon name="monitor" />
                        <strong>{gpu.name}</strong>
                        <span>{gpu.backend}</span>
                      </div>
                      <dl>
                        <div>
                          <dt>Status</dt>
                          <dd>
                            {gpu.utilizationPercent === null
                              ? gpu.note
                              : gpu.utilizationPercent > 1
                                ? "Active"
                                : "Idle"}
                          </dd>
                        </div>
                        <div>
                          <dt>Load</dt>
                          <dd>
                            {gpu.utilizationPercent === null
                              ? "Unavailable"
                              : `${gpu.utilizationPercent}%`}
                          </dd>
                        </div>
                        <div>
                          <dt>Memory</dt>
                          <dd>
                            {gpu.memoryTotalBytes === null
                              ? "Unavailable"
                              : `${memoryLabel(gpu.memoryUsedBytes)} / ${memoryLabel(gpu.memoryTotalBytes)}`}
                          </dd>
                        </div>
                        <div>
                          <dt>Temperature</dt>
                          <dd>
                            {gpu.temperatureC === null
                              ? "Unavailable"
                              : `${gpu.temperatureC} °C`}
                          </dd>
                        </div>
                      </dl>
                    </section>
                  ))
                ) : (
                  <p className="hardware-empty">
                    No GPU was reported by the available system tools.
                  </p>
                )}
              </>
            ) : (
              <p className="hardware-empty">Reading CPU and GPU devices…</p>
            )}
            <p className="hardware-note">
              GPU memory is per device. Metal on Apple silicon shares system
              memory. Some drivers do not expose temperature or load.
            </p>
          </aside>
        </div>
      )}

      {noticeExpanded && notice && (
        <div
          className="app-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) =>
            event.target === event.currentTarget && setNoticeExpanded(false)
          }
        >
          <section
            className="app-dialog error-details-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="error-details-title"
          >
            <header>
              <div>
                <h2 id="error-details-title">Error details</h2>
                <p>The complete local error is shown below.</p>
              </div>
              <button
                type="button"
                className="dialog-close"
                aria-label="Close error details"
                onClick={() => setNoticeExpanded(false)}
              >
                <Icon name="x" />
              </button>
            </header>
            <pre>{notice}</pre>
            <footer>
              <button
                type="button"
                className="primary"
                onClick={() => setNoticeExpanded(false)}
              >
                Close
              </button>
            </footer>
          </section>
        </div>
      )}

      {restartCandidate && (
        <div
          className="app-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !restartingSession)
              setRestartCandidate(null);
          }}
        >
          <section
            className="app-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="restart-session-title"
          >
            <header>
              <div>
                <h2 id="restart-session-title">Clear pipeline and restart?</h2>
                <p>
                  “{restartCandidate.name}” will be erased before a new run
                  starts with the same settings. Its current folder and outputs
                  will be moved to Trash.
                </p>
              </div>
              <button
                type="button"
                className="dialog-close"
                aria-label="Close"
                disabled={restartingSession}
                onClick={() => setRestartCandidate(null)}
              >
                <Icon name="x" />
              </button>
            </header>
            <footer>
              <button
                type="button"
                disabled={restartingSession}
                onClick={() => setRestartCandidate(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="danger"
                disabled={restartingSession}
                onClick={() => void restartSession()}
              >
                {restartingSession ? "Restarting…" : "Clear and restart"}
              </button>
            </footer>
          </section>
        </div>
      )}

      {deleteCandidate && (
        <div
          className="app-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !deletingSession)
              setDeleteCandidate(null);
          }}
        >
          <section
            className="app-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-session-title"
          >
            <header>
              <div>
                <h2 id="delete-session-title">Delete session?</h2>
                <p>
                  “{deleteCandidate.name}” and all of its local outputs will be
                  moved to Trash.
                </p>
              </div>
              <button
                type="button"
                className="dialog-close"
                aria-label="Close"
                disabled={deletingSession}
                onClick={() => setDeleteCandidate(null)}
              >
                <Icon name="x" />
              </button>
            </header>
            <footer>
              <button
                type="button"
                disabled={deletingSession}
                onClick={() => setDeleteCandidate(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="danger"
                disabled={deletingSession}
                onClick={() => void deleteSession()}
              >
                {deletingSession ? "Deleting…" : "Delete session"}
              </button>
            </footer>
          </section>
        </div>
      )}

      {settingsOpen && (
        <div
          className="overlay"
          role="presentation"
          onMouseDown={(event) =>
            event.target === event.currentTarget && setSettingsOpen(false)
          }
        >
          <aside className="settings-panel" role="dialog" aria-label="Settings">
            <div className="settings-title">
              <h2>Settings</h2>
              <button
                className="icon-button"
                onClick={() => setSettingsOpen(false)}
                aria-label="Close settings"
              >
                <Icon name="x" />
              </button>
            </div>

            <section>
              <div className="section-heading">
                <h3>Appearance</h3>
              </div>
              <div className="theme-choice">
                {(
                  [
                    ["dark", "droplet", "Gunmetal + blue"],
                    ["blue-dark", "moon", "Blue dark"],
                    ["blue-light", "sun", "Blue light"],
                  ] as [Theme, IconName, string][]
                ).map(([value, icon, label]) => (
                  <button
                    key={value}
                    className={preferences.theme === value ? "active" : ""}
                    onClick={() => setTheme(value)}
                  >
                    <Icon name={icon} />
                    {label}
                  </button>
                ))}
              </div>
            </section>

            <section>
              <div className="section-heading">
                <h3>osAi backend</h3>
              </div>
              <div
                className={
                  "backend-status " +
                  (backendChecking || backend === null
                    ? "checking"
                    : backend.available
                      ? "ready"
                      : "missing")
                }
              >
                <div>
                  <b>
                    {backendChecking || backend === null
                      ? "Checking…"
                      : backend.available
                        ? "Connected"
                        : "Not connected"}
                  </b>
                  <small>
                    {backendChecking || backend === null
                      ? "Verifying the managed osAi CLI"
                      : backend.message}
                  </small>
                </div>
              </div>
              <p className="field-hint">
                osAi downloads the CLI from its main branch and builds it for
                this computer.
              </p>
              <div className="button-row">
                <button
                  className="quiet-button"
                  disabled={backendChecking || backendInstallBusy}
                  onClick={() => void refreshBackend()}
                >
                  <Icon name={backendChecking ? "loader" : "check"} />
                  {backendChecking ? "Checking…" : "Check installation"}
                </button>
                <button
                  className="quiet-button"
                  disabled={backendInstallBusy}
                  onClick={() => void downloadBackend()}
                >
                  <Icon name={backendInstallBusy ? "loader" : "download"} />
                  {backendInstallBusy ? "Installing…" : "Install or repair"}
                </button>
              </div>
            </section>

            <section>
              <div className="section-heading">
                <h3>App updates</h3>
              </div>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={preferences.autoUpdateEnabled}
                  onChange={async (event) => {
                    const enabled = event.target.checked;
                    setPreferences({
                      ...preferences,
                      autoUpdateEnabled: enabled,
                    });
                    setUpdate(await window.osai.setAppAutoUpdate(enabled));
                  }}
                />
                <span>
                  <b>Install updates automatically</b>
                  <small>
                    Downloads verified releases, closes osAi, then opens the
                    installer.
                  </small>
                </span>
              </label>
              <div className="update-state">
                <Icon name="refresh-cw" />
                <span>{update.message}</span>
                {typeof update.percent === "number" && <b>{update.percent}%</b>}
              </div>
              <div className="button-row">
                <button
                  className="quiet-button"
                  disabled={updateBusy}
                  onClick={() => void runUpdateAction()}
                >
                  {updateLabel}
                </button>
              </div>
            </section>

            <p className="settings-footnote">
              Training, rollout generation and alignment run locally. Network
              access is used only for initial setup, app updates and optional
              model downloads.
            </p>
          </aside>
        </div>
      )}
    </div>
  );
}
