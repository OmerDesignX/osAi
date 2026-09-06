import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { Preferences } from "../types.js";

export const defaultSessionsRoot = path.join(os.homedir(), "osAi", "sessions");

export const defaultPreferences: Preferences = {
  version: 1,
  theme: "dark",
  backendExecutable: "",
  autoUpdateEnabled: false,
  sessionsRoot: defaultSessionsRoot,
  sessionRoots: [defaultSessionsRoot],
};

function directory(value: unknown) {
  return typeof value === "string" &&
    value.length <= 4096 &&
    path.isAbsolute(value)
    ? path.resolve(value)
    : "";
}

function validate(value: unknown): Preferences {
  if (!value || typeof value !== "object") return { ...defaultPreferences };
  const input = value as Partial<Preferences>;
  const sessionsRoot = directory(input.sessionsRoot) || defaultSessionsRoot;
  const sessionRoots = Array.isArray(input.sessionRoots)
    ? input.sessionRoots.map(directory).filter(Boolean)
    : [];
  return {
    version: 1,
    theme:
      input.theme === "blue-dark" || input.theme === "blue-light"
        ? input.theme
        : "dark",
    backendExecutable:
      typeof input.backendExecutable === "string"
        ? input.backendExecutable.slice(0, 4096)
        : "",
    autoUpdateEnabled: input.autoUpdateEnabled === true,
    sessionsRoot,
    sessionRoots: [...new Set([sessionsRoot, ...sessionRoots])].slice(0, 32),
  };
}

export async function readPreferences(file: string) {
  try {
    return validate(JSON.parse(await fs.readFile(file, "utf8")));
  } catch {
    return { ...defaultPreferences };
  }
}

export async function writePreferences(file: string, value: unknown) {
  const preferences = validate(value);
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(preferences, null, 2)}\n`, {
    mode: 0o600,
  });
  await fs.rename(temporary, file);
  return preferences;
}
