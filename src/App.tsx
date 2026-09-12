import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import feather from "feather-icons";
import type {
  AlignmentType,
  AppUpdateStatus,
  BackendInstallStatus,
  BackendStatus,
  LoraTargetModule,
  Preferences,
  SessionState,
  Theme,
  TrainingRequest,
  TrainingStage,
} from "./types.js";
import osAiIcon from "./assets/osai-icon.png";
import { TrainingWiki } from "./TrainingWiki.js";

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
}: {
  label: string;
  value: number | null;
  onChange(value: number | null): void;
  min?: number;
  max?: number;
  step?: number | "any";
  placeholder?: string;
  disabled?: boolean;
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

const defaults: TrainingRequest = {
  sessionsRoot: "",
  modelSource: "official",
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
  currentVersion: "0.1.15",
};

const fallbackBackendInstall: BackendInstallStatus = {
  state: "idle",
  message: "osAi CLI is not installed",
};

export function App() {
  const [preferences, setPreferences] = useState(fallbackPreferences);
  const [form, setForm] = useState(defaults);
  const [sameDataset, setSameDataset] = useState(true);
  const [advanced, setAdvanced] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionState[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [log, setLog] = useState("");
  const [backend, setBackend] = useState<BackendStatus | null>(null);
  const [backendInstall, setBackendInstall] = useState(fallbackBackendInstall);
  const [update, setUpdate] = useState(fallbackUpdate);
  const [notice, setNotice] = useState("");
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
  const [sessionControl, setSessionControl] = useState<
    "" | "pausing" | "resuming" | "stopping"
  >("");
  const [showLogLatest, setShowLogLatest] = useState(false);
  const [wikiOpen, setWikiOpen] = useState(false);
  const [wikiActive, setWikiActive] = useState(false);
  const logRef = useRef<HTMLPreElement | null>(null);
  const followLogRef = useRef(true);
  const selectionClearedRef = useRef(false);

  const selected = useMemo(
    () => sessions.find((session) => session.id === selectedId) || null,
    [selectedId, sessions],
  );
  const active = sessions.find((session) =>
    ["queued", "running", "paused", "stopping"].includes(session.status),
  );
  const sessionMenuSession = sessions.find(
    (session) => session.id === sessionMenu?.id,
  );

  const refreshSessions = useCallback(async () => {
    const next = await window.osai.listSessions();
    setSessions(next);
    setSelectedId((current) =>
      current && next.some((session) => session.id === current)
        ? current
        : selectionClearedRef.current
          ? ""
          : next.find((session) =>
              ["queued", "running", "paused", "stopping"].includes(
                session.status,
              ),
            )?.id ||
            next[0]?.id ||
            "",
    );
  }, []);

  const refreshBackend = useCallback(async () => {
    setBackend(await window.osai.backendStatus());
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
    if (backend?.available) return;
    const interval = window.setInterval(() => void refreshBackend(), 2_500);
    return () => window.clearInterval(interval);
  }, [backend?.available, refreshBackend]);

  useEffect(() => setNoticeExpanded(false), [notice]);

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
      setForm({
        ...defaults,
        sessionsRoot: preferences.sessionsRoot,
      });
      setSameDataset(true);
      setAdvanced(false);
      return;
    }
    if (!selected.request) return;
    const restored = Object.fromEntries(
      Object.entries(selected.request).filter(
        ([, value]) => value !== undefined,
      ),
    ) as Partial<TrainingRequest>;
    setForm({
      ...defaults,
      ...restored,
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

  const chooseAdapter = async () => {
    const value = await window.osai.chooseFile("Choose an osAi adapter");
    if (value) setForm((current) => ({ ...current, adapter: value }));
  };

  const startTraining = async () => {
    setNotice("");
    setStarting(true);
    try {
      const backendState = await window.osai.backendStatus();
      setBackend(backendState);
      if (!backendState.available)
        throw new Error(
          "Install osAi CLI or select its executable in Settings before training.",
        );
      const request = {
        ...form,
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
      setNotice(readableError(error));
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
  const showStatusbar =
    !active || selected?.id !== active.id || settingsOpen || !backendReady;

  return (
    <div
      className={
        "app " +
        preferences.theme +
        " platform-" +
        window.osai.platform +
        (showStatusbar ? "" : " statusbar-hidden")
      }
    >
      <div className="mac-titlebar-safe-area" aria-hidden="true" />
      <header className="topbar">
        <div className="brand" aria-label="osAi">
          <img src={osAiIcon} alt="" aria-hidden="true" />
          <div className="brand-wordmark">
            <span>os</span>
            <b>Ai</b>
          </div>
        </div>

        <div
          className={"global-activity " + (notice ? "has-status" : "")}
          aria-live="polite"
        >
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

            <div className="training-form">
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
                  <h2>Model</h2>
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
                  <div className="model-tiers">
                    {(["small", "medium", "large"] as const).map((tier) => (
                      <button
                        type="button"
                        key={tier}
                        className={form.tier === tier ? "active" : ""}
                        onClick={() => setForm({ ...form, tier })}
                      >
                        {tier.charAt(0).toUpperCase() + tier.slice(1)}
                      </button>
                    ))}
                  </div>
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
                  <h2>Pipeline</h2>
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
                  <h2>Data</h2>
                  <p>
                    Text, image, video and compatible audio data stay on this
                    computer.
                  </p>
                </div>
                {needsFineTune && (
                  <PathField
                    label="Fine-tuning dataset"
                    value={form.fineTuneData}
                    placeholder="Dataset folder or JSON file"
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
                <div className="settings-strip">
                  <label className="toggle-row">
                    <input
                      type="checkbox"
                      checked={form.autoSettings}
                      onChange={(event) =>
                        setForm({ ...form, autoSettings: event.target.checked })
                      }
                    />
                    <span>
                      <b>Fit settings to this hardware</b>
                      <small>Fit memory-sensitive values automatically.</small>
                    </span>
                  </label>
                  <button
                    type="button"
                    className="quiet-button compact-button"
                    onClick={() => setAdvanced(!advanced)}
                    aria-expanded={advanced}
                  >
                    {advanced ? "Hide advanced" : "Advanced"}
                    <Icon name={advanced ? "chevron-up" : "chevron-down"} />
                  </button>
                </div>
              </section>

              {advanced && (
                <div className="advanced-panel">
                  {form.autoSettings && (
                    <div className="advanced-auto-note">
                      <Icon name="cpu" />
                      <span>
                        Hardware fitting controls batch, context, LoRA size and
                        GGUF runtime values. Turn it off to edit those fields.
                      </span>
                    </div>
                  )}

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
                        placeholder={form.autoSettings ? "Hardware auto" : "1"}
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
                        label="Max sequence length"
                        value={form.maxSeqLength}
                        min={32}
                        disabled={form.autoSettings}
                        placeholder={form.autoSettings ? "Hardware auto" : "64"}
                        onChange={(maxSeqLength) =>
                          setForm({ ...form, maxSeqLength })
                        }
                      />
                      {needsFineTune && (
                        <NumberField
                          label="Fine-tune learning rate"
                          value={form.learningRate}
                          min={0}
                          step="any"
                          placeholder="Backend default"
                          onChange={(learningRate) =>
                            setForm({ ...form, learningRate })
                          }
                        />
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
                          placeholder={
                            form.autoSettings ? "Hardware auto" : "2"
                          }
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
                          placeholder={
                            form.autoSettings ? "Hardware auto" : "1"
                          }
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
                        placeholder={form.autoSettings ? "Hardware auto" : "8"}
                        onChange={(ggufBatchSize) =>
                          setForm({ ...form, ggufBatchSize })
                        }
                      />
                      <NumberField
                        label="GGUF CPU threads"
                        value={form.ggufThreads}
                        min={1}
                        disabled={form.autoSettings}
                        placeholder={form.autoSettings ? "Hardware auto" : "2"}
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
            </div>

            <footer className="panel-footer">
              {active ? (
                <div className="active-run-controls">
                  {active.status === "running" && (
                    <button
                      className="primary-button"
                      disabled={Boolean(sessionControl)}
                      onClick={() => void pauseTraining(active.id)}
                    >
                      <Icon
                        name={sessionControl === "pausing" ? "loader" : "pause"}
                      />
                      {sessionControl === "pausing"
                        ? "Pausing…"
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
                  disabled={starting}
                  onClick={() => void startTraining()}
                >
                  <Icon name={starting ? "loader" : "play"} />
                  {starting ? "Starting…" : "Start training"}
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
              <button
                type="button"
                className={
                  "quiet-button compact-button wiki-open-button " +
                  (wikiActive ? "active" : "")
                }
                onClick={() => {
                  setWikiOpen(true);
                  setWikiActive(true);
                  setSessionMenu(null);
                }}
              >
                <Icon name="book-open" />
                Wiki
              </button>
            </header>

            {(sessions.length > 0 || wikiOpen) && (
              <div className="session-tabs" role="tablist">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={
                      "session-tab " +
                      (!wikiActive && selected?.id === session.id
                        ? "active "
                        : "") +
                      session.status
                    }
                  >
                    <button
                      type="button"
                      className="session-tab-select"
                      role="tab"
                      aria-selected={!wikiActive && selected?.id === session.id}
                      onClick={() => {
                        setWikiActive(false);
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
                      ["queued", "running", "paused", "stopping"].includes(
                        sessionMenuSession.status,
                      )
                    }
                    title={
                      ["queued", "running", "paused", "stopping"].includes(
                        sessionMenuSession.status,
                      )
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
                      "paused",
                      "stopping",
                    ].includes(sessionMenuSession.status)}
                    title={
                      ["queued", "running", "paused", "stopping"].includes(
                        sessionMenuSession.status,
                      )
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

            {wikiActive ? (
              <TrainingWiki />
            ) : selected ? (
              <div className="session-detail">
                <header className="session-summary">
                  <div>
                    <h2>{selected.name}</h2>
                    <p>
                      {selected.message} · {phaseLabel(selected.phase)} ·{" "}
                      {friendlyTime(selected.startedAt || selected.createdAt)}
                    </p>
                  </div>
                  <strong>{Math.round(selected.progress)}%</strong>
                </header>
                <div className="inline-progress">
                  <span
                    style={{ width: Math.max(2, selected.progress) + "%" }}
                  />
                </div>
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
              Install the complete osAi CLI repository and its Python packages
              for this computer.
            </p>
            <button
              className="primary-button onboarding-download"
              disabled={backend === null || backendInstallBusy}
              onClick={() => void downloadBackend()}
            >
              <Icon
                name={
                  backend === null || backendInstallBusy ? "loader" : "download"
                }
              />
              {backend === null ? "Checking setup…" : backendInstallLabel}
            </button>
            <div className="onboarding-state">
              <span>
                {backend === null
                  ? "Looking for an existing osAi CLI installation"
                  : backendInstall.state === "idle"
                    ? "The training workspace opens automatically when setup finishes"
                    : backendInstall.message}
              </span>
            </div>
          </section>
        </main>
      )}

      {showStatusbar && (
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
              name={active ? "activity" : backendReady ? "check" : "download"}
            />
            <span>
              {active
                ? active.name + " · " + active.message
                : backendReady
                  ? "Ready"
                  : backend === null
                    ? "Checking osAi CLI"
                    : backendInstall.message}
            </span>
          </div>
          <div className="status-track">
            <span
              className={
                !active &&
                backendInstallBusy &&
                typeof backendInstall.percent !== "number"
                  ? "indeterminate"
                  : ""
              }
              style={{
                width:
                  (active
                    ? Math.max(2, active.progress)
                    : backendInstallBusy &&
                        typeof backendInstall.percent === "number"
                      ? Math.max(2, backendInstall.percent)
                      : 0) + "%",
              }}
            />
          </div>
          <span className="status-percent">
            {active
              ? Math.round(active.progress) + "%"
              : backendReady
                ? "Local"
                : typeof backendInstall.percent === "number"
                  ? `${backendInstall.percent}%`
                  : "Setup"}
          </span>
        </footer>
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
                  "backend-status " + (backend?.available ? "ready" : "missing")
                }
              >
                <div>
                  <b>{backend?.available ? "Connected" : "Not found"}</b>
                  <small>{backend?.message || "Checking the local CLI…"}</small>
                </div>
              </div>
              <label className="field">
                <span>Executable</span>
                <div className="input-button">
                  <input
                    value={preferences.backendExecutable}
                    placeholder="osai from PATH"
                    onChange={(event) =>
                      setPreferences({
                        ...preferences,
                        backendExecutable: event.target.value,
                      })
                    }
                  />
                  <button
                    className="icon-button"
                    onClick={async () => {
                      const value = await window.osai.chooseBackend();
                      if (value)
                        await savePreferences({
                          ...preferences,
                          backendExecutable: value,
                        });
                    }}
                    aria-label="Choose osAi executable"
                  >
                    <Icon name="folder" />
                  </button>
                </div>
              </label>
              <div className="button-row">
                <button
                  className="quiet-button"
                  onClick={() =>
                    void savePreferences(preferences).then(refreshBackend)
                  }
                >
                  <Icon name="check" />
                  Save and check
                </button>
                <button
                  className="quiet-button"
                  disabled={backendInstallBusy}
                  onClick={() => void downloadBackend()}
                >
                  <Icon name={backendInstallBusy ? "loader" : "download"} />
                  {backendInstallBusy ? "Installing…" : "Install locally"}
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
