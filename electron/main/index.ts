import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  Menu,
  session,
  shell,
} from "electron";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type {
  AutoCalibrationResult,
  Preferences,
  TrainingRequest,
} from "../types.js";
import { calibrationKey } from "../calibration-key.js";
import { BackendInstaller } from "./backend-installer.js";
import { readAutoDevices, runAutoBenchmark } from "./auto-benchmark.js";
import {
  cancelAutoCalibration,
  runAutoCalibration,
} from "./auto-calibration.js";
import { readPreferences, writePreferences } from "./preferences.js";
import { SessionService } from "./session-service.js";
import { stopSessionTrainers } from "./session-processes.js";
import {
  datasetTrainingSummary,
  inspectDataset,
  saveDataset,
} from "./dataset-editor.js";
import { modelTrainingSummary } from "./model-summary.js";
import { hardwareSnapshot } from "./hardware-snapshot.js";
import { AppUpdateService } from "./updater.js";
import { migrateLegacyV1Models } from "./model-migration.js";

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) app.quit();

let mainWindow: BrowserWindow | null = null;
let sessionService: SessionService;
let updateService: AppUpdateService;
let backendInstaller: BackendInstaller;
let pendingMacInstaller = "";
let approvedCalibration: {
  key: string;
  datasetSignature: string;
  deviceSignature: string;
  result: AutoCalibrationResult;
} | null = null;

async function moveSessionToTrash(directory: string) {
  let cleanupError: unknown;
  try {
    await stopSessionTrainers(directory);
  } catch (error) {
    cleanupError = error;
  }
  try {
    await shell.trashItem(directory);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    const cleanup = cleanupError
      ? ` Trainer cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}.`
      : "";
    throw new Error(
      `Could not move this session to Trash: ${reason}.${cleanup}`,
    );
  }
}

function userDataPath(...parts: string[]) {
  return path.join(app.getPath("userData"), ...parts);
}

function send(channel: string, value: unknown) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send(channel, value);
}

const macInstallerHandoffScript = [
  'parent_pid="$1"',
  'installer_path="$2"',
  "attempt=0",
  'while /bin/kill -0 "$parent_pid" 2>/dev/null && [ "$attempt" -lt 240 ]; do',
  "  /bin/sleep 0.25",
  "  attempt=$((attempt + 1))",
  "done",
  "/bin/sleep 2",
  'exec /usr/bin/open "$installer_path"',
].join("\n");

function openMacInstallerAfterExit(installerPath: string) {
  const handoff = spawn(
    "/bin/sh",
    [
      "-c",
      macInstallerHandoffScript,
      "osai-update-handoff",
      String(process.pid),
      installerPath,
    ],
    { cwd: "/", detached: true, stdio: "ignore" },
  );
  handoff.once("error", () => undefined);
  handoff.unref();
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1240,
    height: 820,
    minWidth: 940,
    minHeight: 680,
    title: "osAi",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    autoHideMenuBar: process.platform !== "darwin",
    backgroundColor: "#171819",
    icon: app.isPackaged
      ? undefined
      : path.join(app.getAppPath(), "build", "icon.png"),
    webPreferences: {
      preload: path.join(import.meta.dirname, "../preload/index.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });
  mainWindow = window;
  const devUrl = app.isPackaged ? "" : process.env.VITE_DEV_SERVER_URL || "";
  if (devUrl) void window.loadURL(devUrl);
  else void window.loadFile(path.join(app.getAppPath(), "dist", "index.html"));
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  window.on("closed", () => {
    if (mainWindow === window) mainWindow = null;
  });
  return window;
}

function createMenu() {
  const template: Electron.MenuItemConstructorOptions[] = [
    ...(process.platform === "darwin"
      ? [
          {
            label: "osAi",
            submenu: [
              { role: "about" as const },
              { type: "separator" as const },
              { role: "hide" as const },
              { role: "hideOthers" as const },
              { type: "separator" as const },
              { role: "quit" as const },
            ],
          },
        ]
      : []),
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
        ...(!app.isPackaged
          ? ([
              { type: "separator" },
              { role: "toggleDevTools" },
            ] as Electron.MenuItemConstructorOptions[])
          : []),
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "Install or repair osAi CLI",
          click: () => void backendInstaller.install(),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

async function preferences() {
  return readPreferences(userDataPath("preferences.json"));
}

async function migrateModelsIfIdle() {
  try {
    const active = (await sessionService.list()).some((state) =>
      ["queued", "running", "paused", "stopping"].includes(state.status),
    );
    await migrateLegacyV1Models(
      path.join(os.homedir(), "osAi", "models"),
      active,
    );
  } catch (error) {
    // The CLI can still use the old V1 layout. A migration problem must not
    // prevent opening the app or starting a session.
    console.warn("Could not migrate older V1 models:", error);
  }
}

function registerIpc() {
  ipcMain.handle("system:hardware", () => ({
    platform: process.platform,
    architecture: process.arch,
    physicalMemoryBytes: os.totalmem(),
    logicalCpuCount: Math.max(1, os.cpus().length),
  }));
  ipcMain.handle("system:hardware-snapshot", () => hardwareSnapshot());
  ipcMain.handle("preferences:get", () => preferences());
  ipcMain.handle("preferences:set", async (_event, value: unknown) => {
    const saved = await writePreferences(
      userDataPath("preferences.json"),
      value,
    );
    await updateService.setEnabled(saved.autoUpdateEnabled);
    return saved;
  });
  ipcMain.handle("dialog:choose-directory", async (_event, title: unknown) => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      title:
        typeof title === "string" ? title.slice(0, 100) : "Choose a folder",
      properties: ["openDirectory", "createDirectory"],
    });
    return result.canceled ? "" : result.filePaths[0] || "";
  });
  ipcMain.handle("dialog:choose-dataset", async (_event, title: unknown) => {
    const safeTitle =
      typeof title === "string" ? title.slice(0, 100) : "Choose a dataset";
    let properties: Array<"openFile" | "openDirectory"> = [
      "openFile",
      "openDirectory",
    ];
    if (process.platform !== "darwin") {
      const choice = await dialog.showMessageBox(mainWindow!, {
        type: "question",
        title: safeTitle,
        message: "Choose a dataset source",
        detail:
          "Select one JSON, JSONL, NDJSON, or Parquet file or a folder containing dataset files.",
        buttons: ["Choose data file", "Choose folder", "Cancel"],
        defaultId: 0,
        cancelId: 2,
        noLink: true,
      });
      if (choice.response === 2) return "";
      properties = choice.response === 0 ? ["openFile"] : ["openDirectory"];
    }
    const result = await dialog.showOpenDialog(mainWindow!, {
      title: safeTitle,
      properties,
      filters: properties.includes("openFile")
        ? [
            {
              name: "Datasets",
              extensions: ["json", "jsonl", "ndjson", "parquet"],
            },
          ]
        : undefined,
    });
    return result.canceled ? "" : result.filePaths[0] || "";
  });
  ipcMain.handle("dialog:choose-file", async (_event, title: unknown) => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      title: typeof title === "string" ? title.slice(0, 100) : "Choose a file",
      properties: ["openFile"],
    });
    return result.canceled ? "" : result.filePaths[0] || "";
  });
  ipcMain.handle("dataset:inspect", (_event, source: unknown) => {
    if (typeof source !== "string") throw new Error("Invalid dataset source");
    return inspectDataset(source);
  });
  ipcMain.handle("dataset:training-summary", (_event, source: unknown) => {
    if (typeof source !== "string") throw new Error("Invalid dataset source");
    return datasetTrainingSummary(source);
  });
  ipcMain.handle("model:training-summary", (_event, source: unknown) => {
    if (typeof source !== "string") throw new Error("Invalid model folder");
    return modelTrainingSummary(source);
  });
  ipcMain.handle("dataset:save", (_event, value: unknown) =>
    saveDataset(value as Parameters<typeof saveDataset>[0]),
  );
  ipcMain.handle("backend:status", async () => {
    const status = await sessionService.backendStatus();
    if (!status.available) return status;
    if (await backendInstaller.isCurrent(status.executable)) return status;
    return {
      ...status,
      available: false,
      version: "",
      message:
        "The installed osAi CLI needs an update. Choose Install locally.",
    };
  });
  ipcMain.handle("backend-install:status", () => backendInstaller.getStatus());
  ipcMain.handle("backend-install:start", () => backendInstaller.install());
  ipcMain.handle("training:auto-benchmark", async (_event, value: unknown) => {
    const status = await sessionService.backendStatus();
    if (
      !status.available ||
      !(await backendInstaller.isCurrent(status.executable))
    )
      throw new Error("Install the current osAi CLI before benchmarking");
    return runAutoBenchmark(status.executable, value as TrainingRequest);
  });
  ipcMain.handle(
    "training:auto-calibration",
    async (_event, value: unknown) => {
      const status = await sessionService.backendStatus();
      if (
        !status.available ||
        !(await backendInstaller.isCurrent(status.executable))
      )
        throw new Error("Install the current osAi CLI before calibration");
      approvedCalibration = null;
      const input = value as TrainingRequest;
      const [before, devicesBefore] = await Promise.all([
        datasetTrainingSummary(input.fineTuneData),
        readAutoDevices(status.executable, input.accelerator),
      ]);
      const result = await runAutoCalibration(
        status.executable,
        input,
        (message) => send("training:calibration-progress", message),
      );
      if (input.fullContentContext && !result.required_context)
        throw new Error(
          "The CLI did not return its verified full context; update osAi CLI and calibrate again",
        );
      const [after, devicesAfter] = await Promise.all([
        datasetTrainingSummary(input.fineTuneData),
        readAutoDevices(status.executable, input.accelerator),
      ]);
      if (after.signature !== before.signature)
        throw new Error(
          "Training data changed during calibration; run calibration again",
        );
      if (JSON.stringify(devicesAfter) !== JSON.stringify(devicesBefore))
        throw new Error(
          "Available hardware changed during calibration; run calibration again",
        );
      approvedCalibration = {
        key: calibrationKey(input),
        datasetSignature: after.signature,
        deviceSignature: JSON.stringify(devicesAfter),
        result,
      };
      return result;
    },
  );
  ipcMain.handle("training:auto-calibration-cancel", () => {
    cancelAutoCalibration();
    approvedCalibration = null;
  });
  ipcMain.handle("training:auto-devices", async (_event, value: unknown) => {
    const status = await sessionService.backendStatus();
    if (
      !status.available ||
      !(await backendInstaller.isCurrent(status.executable))
    )
      throw new Error("Install the current osAi CLI before detecting GPUs");
    return readAutoDevices(
      status.executable,
      value as TrainingRequest["accelerator"],
    );
  });
  ipcMain.handle("training:start", async (_event, value: unknown) => {
    const input = value as TrainingRequest;
    const status = await sessionService.backendStatus();
    if (
      !status.available ||
      !(await backendInstaller.isCurrent(status.executable))
    )
      throw new Error("Install the current osAi CLI from main before training");
    if (input.autoSettings && input.stage !== "alignment") {
      if (
        !approvedCalibration ||
        approvedCalibration.key !== calibrationKey(input) ||
        approvedCalibration.datasetSignature !==
          (await datasetTrainingSummary(input.fineTuneData)).signature ||
        approvedCalibration.deviceSignature !==
          JSON.stringify(
            await readAutoDevices(status.executable, input.accelerator),
          ) ||
        !input.learningRate ||
        input.learningRate > approvedCalibration.result.learning_rate * 1.0001
      )
        throw new Error(
          "Calibrate this model, dataset and hardware before training",
        );
    }
    await migrateModelsIfIdle();
    const request =
      input.autoSettings && input.stage !== "alignment" && approvedCalibration
        ? {
            ...input,
            calibrationApplied: true,
            batchSize: approvedCalibration.result.settings.batch_size,
            maxSeqLength: approvedCalibration.result.settings.max_seq_length,
            numLayers: approvedCalibration.result.settings.num_layers,
            rank: approvedCalibration.result.settings.rank,
            ggufBatchSize: approvedCalibration.result.settings.gguf_batch_size,
            ggufThreads: approvedCalibration.result.settings.gguf_threads,
            targetModules: approvedCalibration.result.settings.target_modules,
            devices: approvedCalibration.result.devices.join(", "),
            deviceSpeeds: approvedCalibration.result.device_speeds ?? [],
          }
        : input;
    return sessionService.start(
      request,
      input.fullContentContext &&
        input.autoSettings &&
        input.stage !== "alignment"
        ? (approvedCalibration?.result.required_context ?? undefined)
        : undefined,
    );
  });
  ipcMain.handle("training:pause", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.pause(id);
  });
  ipcMain.handle("training:resume", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.resume(id);
  });
  ipcMain.handle("training:checkpoint", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.checkpoint(id);
  });
  ipcMain.handle(
    "training:auto-stop",
    (_event, id: unknown, enabled: unknown) => {
      if (typeof id !== "string" || typeof enabled !== "boolean")
        throw new Error("Invalid Auto stop setting");
      return sessionService.setAutoStop(id, enabled);
    },
  );
  ipcMain.handle("training:stop", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.stop(id);
  });
  ipcMain.handle(
    "training:restart",
    async (_event, id: unknown, value: unknown) => {
      if (typeof id !== "string") throw new Error("Invalid training session");
      const status = await sessionService.backendStatus();
      if (
        !status.available ||
        !(await backendInstaller.isCurrent(status.executable))
      )
        throw new Error(
          "Install the current osAi CLI from main before training",
        );
      await migrateModelsIfIdle();
      return sessionService.restart(
        id,
        value as TrainingRequest,
        moveSessionToTrash,
      );
    },
  );
  ipcMain.handle("training:delete", async (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    await sessionService.remove(id, moveSessionToTrash);
  });
  ipcMain.handle("training:list", () => sessionService.list());
  ipcMain.handle("training:log", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.log(id);
  });
  ipcMain.handle("training:metrics", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.metrics(id);
  });
  ipcMain.handle("training:metrics-export", async (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    const state = await sessionService.find(id);
    const source = path.join(state.sessionDirectory, "metrics.csv");
    if (!(await fs.stat(source).catch(() => null)))
      throw new Error("This session has no recorded loss history yet");
    const safeName = state.name.replace(/[^A-Za-z0-9._-]+/g, "-");
    const result = await dialog.showSaveDialog({
      title: "Save training history as CSV",
      defaultPath: path.join(os.homedir(), `${safeName}-training.csv`),
      filters: [{ name: "CSV", extensions: ["csv"] }],
    });
    if (result.canceled || !result.filePath) return null;
    await fs.copyFile(source, result.filePath);
    return result.filePath;
  });
  ipcMain.handle("training:artifacts", async (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    const state = await sessionService.find(id);
    if (state.status !== "completed")
      return { mergedModel: null, adapterDirectory: null };
    const outputs = path.join(state.sessionDirectory, "outputs");
    const mergedModel = path.join(outputs, "merged-model");
    const ggufManifest = path.join(mergedModel, "gguf", "osai_fusion.json");
    const mlxManifest = path.join(mergedModel, "mlx", "osai_fusion.json");
    const adapterDirectory = path.join(
      outputs,
      "base-plus-adapter",
      "adapters",
    );
    const [gguf, mlx, adapters] = await Promise.all([
      fs.stat(ggufManifest).catch(() => null),
      fs.stat(mlxManifest).catch(() => null),
      fs.readdir(adapterDirectory).catch(() => []),
    ]);
    return {
      mergedModel: gguf?.isFile() || mlx?.isFile() ? mergedModel : null,
      adapterDirectory: adapters.length ? adapterDirectory : null,
    };
  });
  ipcMain.handle("training:open-artifacts", async (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    const state = await sessionService.find(id);
    const outputs = path.join(state.sessionDirectory, "outputs");
    if (!(await fs.stat(outputs).catch(() => null))?.isDirectory())
      throw new Error("This session has no published outputs");
    const error = await shell.openPath(outputs);
    if (error) throw new Error(error);
  });
  ipcMain.handle("training:reveal", async (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    shell.showItemInFolder((await sessionService.find(id)).sessionDirectory);
  });
  ipcMain.handle("training:open-root", async () => {
    const root = await sessionService.root();
    await fs.mkdir(root, { recursive: true });
    const error = await shell.openPath(root);
    if (error) throw new Error(error);
  });
  ipcMain.handle("updates:status", () => updateService.getStatus());
  ipcMain.handle("updates:set-enabled", async (_event, enabled: unknown) => {
    if (typeof enabled !== "boolean") throw new Error("Invalid update setting");
    const current = await preferences();
    await writePreferences(userDataPath("preferences.json"), {
      ...current,
      autoUpdateEnabled: enabled,
    } satisfies Preferences);
    return updateService.setEnabled(enabled);
  });
  ipcMain.handle("updates:check", () => updateService.check(true));
  ipcMain.handle("updates:download", () => updateService.downloadAvailable());
  ipcMain.handle("updates:install", () => updateService.installReadyUpdate());
}

app.whenReady().then(async () => {
  app.setName("osAi");
  const userData = app.getPath("userData");
  await fs.mkdir(userData, { recursive: true, mode: 0o700 });
  sessionService = new SessionService(
    userDataPath("sessions"),
    path.join(app.getAppPath(), "dist-electron", "main", "training-worker.js"),
    preferences,
  );
  await migrateModelsIfIdle();
  backendInstaller = new BackendInstaller(
    userDataPath("backend", "installations"),
    app.isPackaged
      ? path.join(process.resourcesPath, "python")
      : path.join(app.getAppPath(), "build", "python-runtime"),
    app.isPackaged
      ? path.join(process.resourcesPath, "backend-source.json")
      : path.join(app.getAppPath(), "releaseScripts", "backend-source.json"),
    (status) => send("backend-install:status-changed", status),
    async (executable) => {
      const current = await preferences();
      await writePreferences(userDataPath("preferences.json"), {
        ...current,
        backendExecutable: executable,
      } satisfies Preferences);
    },
  );
  updateService = new AppUpdateService(
    userDataPath("updates"),
    (status) => send("updates:status-changed", status),
    (installerPath) => {
      if (process.platform === "darwin") pendingMacInstaller = installerPath;
      app.quit();
    },
  );
  const devOrigin =
    !app.isPackaged && process.env.VITE_DEV_SERVER_URL
      ? new URL(process.env.VITE_DEV_SERVER_URL).origin
      : "";
  session.defaultSession.webRequest.onBeforeRequest(
    { urls: ["http://*/*", "https://*/*"] },
    (details, callback) => {
      let allowed = false;
      if (devOrigin) {
        try {
          allowed = new URL(details.url).origin === devOrigin;
        } catch {
          allowed = false;
        }
      }
      callback({ cancel: !allowed });
    },
  );
  registerIpc();
  createMenu();
  createWindow();
  updateService.initialize((await preferences()).autoUpdateEnabled);
});

app.on("second-instance", () => {
  if (!mainWindow || mainWindow.isDestroyed()) createWindow();
  else {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  }
});

app.on("activate", () => {
  if (!mainWindow || mainWindow.isDestroyed()) createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  cancelAutoCalibration();
  updateService?.dispose();
  if (process.platform === "darwin" && pendingMacInstaller) {
    const installer = pendingMacInstaller;
    pendingMacInstaller = "";
    openMacInstallerAfterExit(installer);
  }
});
