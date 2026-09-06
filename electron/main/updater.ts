import { app, shell } from "electron";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { AppUpdateStatus } from "../types.js";
import {
  isNewerVersion,
  isTrustedUpdateUrl,
  selectUpdateAsset,
  updateAssetName,
  updateAssetVersion,
  updateChannel,
  type UpdateChannel,
} from "./updater-policy.js";

const CHECK_INTERVAL_MS = 4 * 60 * 60 * 1_000;
const RELEASE_API_ROOT =
  "https://api.github.com/repos/OmerDesignX/osAi-CLI/releases/tags";
const MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024;

type ReleaseAsset = {
  name?: unknown;
  size?: unknown;
  digest?: unknown;
  browser_download_url?: unknown;
};

type AvailableUpdate = {
  version: string;
  name: string;
  asset: ReleaseAsset;
  channel: UpdateChannel;
};

type ReadyUpdate = {
  version: string;
  name: string;
  digest: string;
  bytes: number;
  channelTag: string;
};

export class AppUpdateService {
  private enabled = false;
  private checking = false;
  private downloading = false;
  private timer: ReturnType<typeof setInterval> | null = null;
  private automaticInstallTimer: ReturnType<typeof setTimeout> | null = null;
  private downloadedPackage = "";
  private available: AvailableUpdate | null = null;
  private status: AppUpdateStatus = {
    state: "disabled",
    message: "Automatic updates are off; manual checks remain available",
    currentVersion: app.getVersion(),
  };

  constructor(
    private readonly updatesRoot: string,
    private readonly emit: (status: AppUpdateStatus) => void,
    private readonly handoffInstaller: (installerPath: string) => void,
  ) {}

  initialize(enabled: boolean) {
    this.enabled = enabled;
    this.schedule();
    if (!this.supported()) {
      this.update({
        state: "unsupported",
        message: app.isPackaged
          ? "Updates are unavailable for this operating system build"
          : "Update checks are available in packaged builds",
      });
      return;
    }
    this.update(
      enabled
        ? { state: "idle", message: "Automatic updates are on" }
        : {
            state: "disabled",
            message:
              "Automatic updates are off; manual checks remain available",
          },
    );
    void this.restoreReadyUpdate().then((restored) => {
      if (restored) {
        this.scheduleAutomaticInstall();
        return;
      }
      if (!this.enabled) return;
      const timeout = setTimeout(() => void this.check(false), 2_500);
      timeout.unref();
    });
  }

  async setEnabled(enabled: boolean) {
    this.enabled = enabled;
    this.schedule();
    if (!this.supported()) return this.getStatus();
    if (!enabled) this.cancelAutomaticInstall();
    if (this.status.state === "installing") return this.getStatus();
    if (this.status.state === "ready") {
      this.scheduleAutomaticInstall();
      return this.getStatus();
    }
    if (!enabled) {
      this.update({
        state: this.available ? "available" : "disabled",
        message: this.available
          ? `osAi ${this.available.version} is available`
          : "Automatic updates are off; manual checks remain available",
        version: this.available?.version,
        channel: this.available?.channel.label,
      });
      return this.getStatus();
    }
    await this.check(false);
    return this.getStatus();
  }

  async check(manual = false) {
    if (!this.enabled && !manual) return this.getStatus();
    if (this.status.state === "ready" || this.status.state === "installing")
      return this.getStatus();
    const channel = this.channel();
    if (!app.isPackaged || !channel) {
      this.update({
        state: "unsupported",
        message: app.isPackaged
          ? "No update channel exists for this system"
          : "Update checks are available in packaged builds",
      });
      return this.getStatus();
    }
    if (this.checking || this.downloading) return this.getStatus();
    this.checking = true;
    this.update({
      state: "checking",
      message: `Checking the ${channel.label} update channel`,
      channel: channel.label,
    });
    try {
      const apiUrl = `${RELEASE_API_ROOT}/${encodeURIComponent(channel.tag)}`;
      if (!isTrustedUpdateUrl(apiUrl))
        throw new Error("The update channel address is not trusted");
      const response = await fetch(apiUrl, {
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "osAi-updater",
          "X-GitHub-Api-Version": "2022-11-28",
        },
        signal: AbortSignal.timeout(30_000),
      });
      if (response.status === 404) {
        this.available = null;
        this.update({
          state: "current",
          message: "osAi is up to date",
          version: app.getVersion(),
          channel: channel.label,
        });
        return this.getStatus();
      }
      if (!response.ok) throw new Error(`GitHub returned ${response.status}`);
      const release = (await response.json()) as {
        tag_name?: unknown;
        draft?: unknown;
        prerelease?: unknown;
        assets?: ReleaseAsset[];
      };
      if (release.draft || release.prerelease)
        throw new Error("The update channel is not public and stable");
      if (release.tag_name !== channel.tag)
        throw new Error("GitHub returned the wrong update channel");
      const assets = Array.isArray(release.assets) ? release.assets : [];
      const selected = selectUpdateAsset(
        assets,
        app.getVersion(),
        process.platform,
        process.arch,
      );
      if (!selected) {
        if (
          !assets.some((asset) =>
            Boolean(
              updateAssetVersion(
                String(asset.name || ""),
                process.platform,
                process.arch,
              ),
            ),
          )
        )
          throw new Error(`No ${channel.label} package has been uploaded yet`);
        this.available = null;
        this.update({
          state: "current",
          message: "osAi is up to date",
          version: app.getVersion(),
          channel: channel.label,
        });
        return this.getStatus();
      }
      const name = String(selected.asset.name || "");
      if (name !== updateAssetName(selected.version))
        throw new Error("The update package name is invalid");
      this.available = {
        version: selected.version,
        name,
        asset: selected.asset,
        channel,
      };
      this.update({
        state: "available",
        message: `osAi ${selected.version} is available`,
        version: selected.version,
        channel: channel.label,
      });
      if (this.enabled) await this.downloadAvailable();
    } catch (error) {
      this.update({
        state: "error",
        message: `Update check failed: ${this.cleanError(error)}`,
        channel: channel.label,
      });
    } finally {
      this.checking = false;
    }
    return this.getStatus();
  }

  async downloadAvailable() {
    if (!this.supported() || this.downloading) return this.getStatus();
    if (!this.available) {
      await this.check(true);
      if (!this.available || this.status.state !== "available")
        return this.getStatus();
    }
    this.downloading = true;
    try {
      await this.download(this.available);
    } catch (error) {
      this.update({
        state: "error",
        message: `Update download failed: ${this.cleanError(error)}`,
        version: this.available.version,
        channel: this.available.channel.label,
      });
    } finally {
      this.downloading = false;
    }
    return this.getStatus();
  }

  getStatus() {
    return { ...this.status };
  }

  async installReadyUpdate() {
    if (this.status.state !== "ready") return this.getStatus();
    this.cancelAutomaticInstall();
    const expectedExtension =
      process.platform === "win32"
        ? ".exe"
        : process.platform === "darwin"
          ? ".dmg"
          : ".deb";
    if (
      !this.downloadedPackage ||
      path.extname(this.downloadedPackage).toLowerCase() !== expectedExtension
    )
      return this.getStatus();
    try {
      if (process.platform !== "darwin") {
        const error = await shell.openPath(this.downloadedPackage);
        if (error) throw new Error(error);
      }
      this.update({
        state: "installing",
        message:
          process.platform === "darwin"
            ? "osAi is closing; the installer will open next"
            : "Installer opened; osAi is closing",
        version: this.status.version,
        channel: this.status.channel,
        percent: 100,
      });
      const timeout = setTimeout(
        () => this.handoffInstaller(this.downloadedPackage),
        150,
      );
      timeout.unref();
    } catch (error) {
      this.update({
        state: "error",
        message: `Installer failed to open: ${this.cleanError(error)}`,
        version: this.status.version,
        channel: this.status.channel,
      });
    }
    return this.getStatus();
  }

  dispose() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.cancelAutomaticInstall();
  }

  private supported() {
    return app.isPackaged && Boolean(this.channel());
  }

  private channel() {
    return updateChannel(process.platform, process.arch, os.release());
  }

  private schedule() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (!this.enabled || !this.supported()) return;
    this.timer = setInterval(() => void this.check(false), CHECK_INTERVAL_MS);
    this.timer.unref();
  }

  private get readyMetadataPath() {
    return path.join(this.updatesRoot, "ready-update.json");
  }

  private async download(update: AvailableUpdate) {
    const url = String(update.asset.browser_download_url || "");
    const digest = String(update.asset.digest || "");
    const expectedBytes = Number(update.asset.size || 0);
    if (!isTrustedUpdateUrl(url))
      throw new Error("The update download address is not trusted");
    if (!/^sha256:[a-f0-9]{64}$/i.test(digest))
      throw new Error("The update has no trusted SHA-256 checksum");
    if (
      !Number.isSafeInteger(expectedBytes) ||
      expectedBytes < 10_000_000 ||
      expectedBytes > MAX_PACKAGE_BYTES
    )
      throw new Error("The update package size is invalid");
    await fs.mkdir(this.updatesRoot, { recursive: true });
    const target = path.join(this.updatesRoot, update.name);
    const partial = `${target}.partial`;
    await fs.rm(partial, { force: true });
    const response = await fetch(url, {
      redirect: "follow",
      signal: AbortSignal.timeout(30 * 60_000),
    });
    if (!response.ok || !response.body || !isTrustedUpdateUrl(response.url))
      throw new Error(`Update download failed (${response.status})`);
    let received = 0;
    let lastPercent = -1;
    const hash = createHash("sha256");
    const body = Readable.fromWeb(response.body as never);
    body.on("data", (chunk: Buffer) => {
      received += chunk.length;
      hash.update(chunk);
      const percent = Math.min(
        99,
        Math.floor((received / expectedBytes) * 100),
      );
      if (percent === lastPercent) return;
      lastPercent = percent;
      this.update({
        state: "downloading",
        message: `Downloading osAi ${update.version}`,
        version: update.version,
        percent,
        channel: update.channel.label,
      });
    });
    try {
      await pipeline(
        body,
        createWriteStream(partial, { flags: "w", mode: 0o600 }),
      );
      if (received !== expectedBytes)
        throw new Error("The update download is incomplete");
      if (`sha256:${hash.digest("hex")}`.toLowerCase() !== digest.toLowerCase())
        throw new Error("The update failed checksum verification");
      await fs.rm(target, { force: true });
      await fs.rename(partial, target);
      const ready: ReadyUpdate = {
        version: update.version,
        name: update.name,
        digest: digest.toLowerCase(),
        bytes: expectedBytes,
        channelTag: update.channel.tag,
      };
      await fs.writeFile(
        this.readyMetadataPath,
        `${JSON.stringify(ready, null, 2)}\n`,
        { mode: 0o600 },
      );
      this.downloadedPackage = target;
      this.update({
        state: "ready",
        message: `osAi ${update.version} is downloaded and ready`,
        version: update.version,
        percent: 100,
        channel: update.channel.label,
      });
      this.scheduleAutomaticInstall();
    } catch (error) {
      await fs.rm(partial, { force: true }).catch(() => undefined);
      throw error;
    }
  }

  private async restoreReadyUpdate() {
    try {
      const raw = JSON.parse(
        await fs.readFile(this.readyMetadataPath, "utf8"),
      ) as Partial<ReadyUpdate>;
      const channel = this.channel();
      const version = String(raw.version || "");
      const name = String(raw.name || "");
      const digest = String(raw.digest || "");
      const bytes = Number(raw.bytes || 0);
      if (
        !channel ||
        raw.channelTag !== channel.tag ||
        !isNewerVersion(version, app.getVersion()) ||
        name !== updateAssetName(version) ||
        !/^sha256:[a-f0-9]{64}$/i.test(digest) ||
        !Number.isSafeInteger(bytes)
      )
        throw new Error("Stale update metadata");
      const file = path.join(this.updatesRoot, name);
      const stat = await fs.lstat(file);
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size !== bytes)
        throw new Error("Stale update package");
      const hash = createHash("sha256");
      await pipeline(createReadStream(file), hash);
      if (`sha256:${hash.digest("hex")}`.toLowerCase() !== digest.toLowerCase())
        throw new Error("Stored update checksum mismatch");
      this.downloadedPackage = file;
      this.update({
        state: "ready",
        message: `osAi ${version} is downloaded and ready`,
        version,
        percent: 100,
        channel: channel.label,
      });
      return true;
    } catch {
      await fs
        .rm(this.readyMetadataPath, { force: true })
        .catch(() => undefined);
      return false;
    }
  }

  private update(next: Omit<AppUpdateStatus, "currentVersion">) {
    this.status = { ...next, currentVersion: app.getVersion() };
    this.emit(this.getStatus());
  }

  private scheduleAutomaticInstall() {
    this.cancelAutomaticInstall();
    if (!this.enabled || this.status.state !== "ready") return;
    this.automaticInstallTimer = setTimeout(
      () => void this.installReadyUpdate(),
      1_200,
    );
    this.automaticInstallTimer.unref();
  }

  private cancelAutomaticInstall() {
    if (this.automaticInstallTimer) clearTimeout(this.automaticInstallTimer);
    this.automaticInstallTimer = null;
  }

  private cleanError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return message.replace(/https?:\/\/\S+/g, "GitHub Releases").slice(0, 220);
  }
}
