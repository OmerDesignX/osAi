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
import path from "node:path";
import type { Preferences, TrainingRequest } from "../types.js";
import { BackendInstaller } from "./backend-installer.js";
import { readPreferences, writePreferences } from "./preferences.js";
import { SessionService } from "./session-service.js";
import { AppUpdateService } from "./updater.js";

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) app.quit();

let mainWindow: BrowserWindow | null = null;
let sessionService: SessionService;
let updateService: AppUpdateService;
let backendInstaller: BackendInstaller;
let pendingMacInstaller = "";

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

function registerIpc() {
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
          "Select one JSON/JSONL file or a folder containing dataset splits.",
        buttons: ["Choose JSON file", "Choose folder", "Cancel"],
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
        ? [{ name: "JSON datasets", extensions: ["json", "jsonl"] }]
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
  ipcMain.handle("dialog:choose-backend", async () => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      title: "Choose the osAi executable",
      properties: ["openFile"],
      filters:
        process.platform === "win32"
          ? [{ name: "Applications", extensions: ["exe"] }]
          : [{ name: "Executable", extensions: ["*"] }],
    });
    return result.canceled ? "" : result.filePaths[0] || "";
  });
  ipcMain.handle("backend:status", () => sessionService.backendStatus());
  ipcMain.handle("backend-install:status", () => backendInstaller.getStatus());
  ipcMain.handle("backend-install:start", () => backendInstaller.install());
  ipcMain.handle("training:start", (_event, value: unknown) =>
    sessionService.start(value as TrainingRequest),
  );
  ipcMain.handle("training:pause", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.pause(id);
  });
  ipcMain.handle("training:resume", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.resume(id);
  });
  ipcMain.handle("training:stop", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.stop(id);
  });
  ipcMain.handle("training:delete", async (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    await sessionService.remove(id, (directory) => shell.trashItem(directory));
  });
  ipcMain.handle("training:list", () => sessionService.list());
  ipcMain.handle("training:log", (_event, id: unknown) => {
    if (typeof id !== "string") throw new Error("Invalid training session");
    return sessionService.log(id);
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
  backendInstaller = new BackendInstaller(
    userDataPath("backend", "installations"),
    app.isPackaged
      ? path.join(process.resourcesPath, "python")
      : path.join(app.getAppPath(), "build", "python-runtime"),
    app.isPackaged
      ? path.join(process.resourcesPath, "backend")
      : path.join(app.getAppPath(), "build", "backend-bundle"),
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
  updateService?.dispose();
  if (process.platform === "darwin" && pendingMacInstaller) {
    const installer = pendingMacInstaller;
    pendingMacInstaller = "";
    openMacInstallerAfterExit(installer);
  }
});
