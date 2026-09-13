import { contextBridge, ipcRenderer } from "electron";
import type {
  AppUpdateStatus,
  BackendInstallStatus,
  OsAiBridge,
  Preferences,
  TrainingRequest,
} from "../types.js";

const bridge: OsAiBridge = {
  platform: process.platform,
  hardwareInfo: () => ipcRenderer.invoke("system:hardware"),
  loadPreferences: () => ipcRenderer.invoke("preferences:get"),
  savePreferences: (value: Preferences) =>
    ipcRenderer.invoke("preferences:set", value),
  chooseDirectory: (title: string) =>
    ipcRenderer.invoke("dialog:choose-directory", title),
  chooseDataset: (title: string) =>
    ipcRenderer.invoke("dialog:choose-dataset", title),
  chooseFile: (title: string) =>
    ipcRenderer.invoke("dialog:choose-file", title),
  chooseBackend: () => ipcRenderer.invoke("dialog:choose-backend"),
  inspectDataset: (source: string) =>
    ipcRenderer.invoke("dataset:inspect", source),
  saveDataset: (value) => ipcRenderer.invoke("dataset:save", value),
  backendStatus: () => ipcRenderer.invoke("backend:status"),
  backendInstallStatus: () => ipcRenderer.invoke("backend-install:status"),
  installBackend: () => ipcRenderer.invoke("backend-install:start"),
  startTraining: (value: TrainingRequest) =>
    ipcRenderer.invoke("training:start", value),
  pauseTraining: (id: string) => ipcRenderer.invoke("training:pause", id),
  resumeTraining: (id: string) => ipcRenderer.invoke("training:resume", id),
  stopTraining: (id: string) => ipcRenderer.invoke("training:stop", id),
  restartSession: (id: string, value: TrainingRequest) =>
    ipcRenderer.invoke("training:restart", id, value),
  deleteSession: (id: string) => ipcRenderer.invoke("training:delete", id),
  listSessions: () => ipcRenderer.invoke("training:list"),
  sessionLog: (id: string) => ipcRenderer.invoke("training:log", id),
  revealSession: (id: string) => ipcRenderer.invoke("training:reveal", id),
  openSessionsFolder: () => ipcRenderer.invoke("training:open-root"),
  appUpdateStatus: () => ipcRenderer.invoke("updates:status"),
  setAppAutoUpdate: (enabled: boolean) =>
    ipcRenderer.invoke("updates:set-enabled", enabled),
  checkForAppUpdate: () => ipcRenderer.invoke("updates:check"),
  downloadAppUpdate: () => ipcRenderer.invoke("updates:download"),
  installAppUpdate: () => ipcRenderer.invoke("updates:install"),
  onAppUpdateStatus: (callback: (status: AppUpdateStatus) => void) => {
    const listener = (_event: unknown, status: AppUpdateStatus) =>
      callback(status);
    ipcRenderer.on("updates:status-changed", listener);
    return () => ipcRenderer.removeListener("updates:status-changed", listener);
  },
  onBackendInstallStatus: (
    callback: (status: BackendInstallStatus) => void,
  ) => {
    const listener = (_event: unknown, status: BackendInstallStatus) =>
      callback(status);
    ipcRenderer.on("backend-install:status-changed", listener);
    return () =>
      ipcRenderer.removeListener("backend-install:status-changed", listener);
  },
};

contextBridge.exposeInMainWorld("osai", bridge);
