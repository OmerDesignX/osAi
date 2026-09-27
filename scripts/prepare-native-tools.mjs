import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { runtimeExecutable } from "./prepare-python-runtime.mjs";

const root = path.resolve(import.meta.dirname, "..");
const toolsRoot = path.join(root, "build", "native-tools");
const runtimeRoot = path.join(root, "build", "python-runtime");
const cmakeVersion = "4.4.3";
const ninjaVersion = "1.13.2";

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      shell: false,
      windowsHide: true,
      stdio: "inherit",
    });
    child.once("error", reject);
    child.once("exit", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`${command} exited with code ${code}`)),
    );
  });
}

async function clean() {
  const buildRoot = path.join(root, "build");
  if (path.dirname(toolsRoot) !== buildRoot)
    throw new Error("Refusing to clean outside the native build directory");
  await fs.rm(toolsRoot, { recursive: true, force: true });
}

export async function prepareNativeTools(platform, architecture) {
  if (
    !["windows", "macos", "linux"].includes(platform) ||
    !["arm64", "x64"].includes(architecture)
  )
    throw new Error("Unsupported native tools target");
  await clean();
  const packages = path.join(toolsRoot, ".packages");
  await fs.mkdir(packages, { recursive: true });
  const python = runtimeExecutable(runtimeRoot, platform);
  await run(python, [
    "-m",
    "pip",
    "install",
    "--disable-pip-version-check",
    "--no-input",
    "--no-deps",
    "--only-binary=:all:",
    "--target",
    packages,
    `cmake==${cmakeVersion}`,
    `ninja==${ninjaVersion}`,
  ]);
  const suffix = platform === "windows" ? ".exe" : "";
  const cmakeData = path.join(packages, "cmake", "data");
  const cmake = path.join(toolsRoot, "cmake", "bin", `cmake${suffix}`);
  const ninjaSource = path.join(packages, "bin", `ninja${suffix}`);
  const ninja = path.join(toolsRoot, "bin", `ninja${suffix}`);
  if (
    !(await fs
      .stat(path.join(cmakeData, "bin", `cmake${suffix}`))
      .catch(() => null)) ||
    !(await fs.stat(ninjaSource).catch(() => null))
  )
    throw new Error("Pinned CMake or Ninja wheel has an unexpected layout");
  await fs.cp(cmakeData, path.join(toolsRoot, "cmake"), { recursive: true });
  await fs.mkdir(path.dirname(ninja), { recursive: true });
  await fs.copyFile(ninjaSource, ninja);
  if (platform !== "windows") {
    await fs.chmod(cmake, 0o755);
    await fs.chmod(ninja, 0o755);
  }
  await fs.rm(packages, { recursive: true, force: true });
  await run(cmake, ["--version"]);
  await run(ninja, ["--version"]);
  await fs.writeFile(
    path.join(toolsRoot, "OSAI_NATIVE_TOOLS.json"),
    `${JSON.stringify({ platform, architecture, cmakeVersion, ninjaVersion }, null, 2)}\n`,
  );
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(import.meta.filename)
) {
  const [platform, architecture] = process.argv.slice(2);
  if (platform === "clean") {
    await clean();
  } else {
    await prepareNativeTools(platform, architecture);
  }
}
