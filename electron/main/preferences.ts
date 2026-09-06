import fs from "node:fs/promises";
import path from "node:path";
import type { Preferences } from "../types.js";

export const defaultPreferences: Preferences = {
  version: 1,
  theme: "dark",
  backendExecutable: "",
  autoUpdateEnabled: false,
};

function validate(value: unknown): Preferences {
  if (!value || typeof value !== "object") return { ...defaultPreferences };
  const input = value as Partial<Preferences>;
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
