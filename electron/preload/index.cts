import { contextBridge, ipcRenderer } from "electron";
import type {
  AppUpdateStatus,
  OsAiBridge,
  Preferences,
  TrainingRequest,
} from "../types.js";

const bridge: OsAiBridge = {
  platform: process.platform,
  loadPreferences: () => ipcRenderer.invoke("preferences:get"),
  savePreferences: (value: Preferences) =>
    ipcRenderer.invoke("preferences:set", value),
  chooseDirectory: (title: string) =>
    ipcRenderer.invoke("dialog:choose-directory", title),
  chooseFile: (title: string) =>
    ipcRenderer.invoke("dialog:choose-file", title),
  chooseBackend: () => ipcRenderer.invoke("dialog:choose-backend"),
  backendStatus: () => ipcRenderer.invoke("backend:status"),
  openBackendDownload: () => ipcRenderer.invoke("backend:download"),
  startTraining: (value: TrainingRequest) =>
    ipcRenderer.invoke("training:start", value),
  stopTraining: (id: string) => ipcRenderer.invoke("training:stop", id),
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
};

contextBridge.exposeInMainWorld("osai", bridge);
