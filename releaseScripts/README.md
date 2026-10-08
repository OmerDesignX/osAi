# Local release builds

Set the release number in the repository root `VERSION.txt`, then run one command on each target system:

```sh
# macOS 12 or newer — builds Apple Silicon and Intel DMGs
bash releaseScripts/macos/build.sh

# Windows 10 or 11
.\releaseScripts\windows\build-windows.cmd

# Current x64 Debian or Ubuntu
bash releaseScripts/linux/build.sh
```

Each script synchronizes `package.json` from the root version file, installs locked App dependencies, downloads and verifies the pinned CPython runtime, runs checks, builds and verifies the native package, stages it in `release-assets/<platform>`, and removes intermediate output. Release machines need Node.js 22 or newer with npm included; the scripts use the exact pnpm version declared by `package.json` and bootstrap it automatically when it is not already on `PATH`. Windows also uses Git Bash. They do not need a CUDA or Vulkan SDK because the release does not compile or bundle llama.cpp.

On first launch, the installed App downloads the osAi CLI repository from the URL in `backend-source.json`, installs its Python dependencies, and compiles llama.cpp on that computer. Each native release bundles pinned CMake and Ninja tools. The CLI code must be pushed to GitHub before a release can download those changes. CUDA compilation uses a compatible CUDA Toolkit when present. On Windows, setup uses Microsoft C++ Build Tools when present or downloads a verified portable compiler. CUDA installs build CUDA alone to avoid an unnecessary Vulkan SDK download and combined build. When CUDA is unavailable, setup reuses a complete installed Vulkan SDK, including one from an older osAi installation. Only if none exists does it download a verified SDK to a shared cache outside versioned installations. A machine with an available GPU reports a build failure instead of silently installing only a CPU trainer. macOS uses Metal when available; Apple silicon with macOS 14 or newer can also use MLX.

Windows Vulkan backend setup checks that the shader compiler can start before building. If the Microsoft C++ runtime is missing or outdated, it downloads a pinned, SHA-256-verified Microsoft x64 Redistributable and installs it without restarting Windows. This prerequisite may request administrator approval. The App keeps complete setup diagnostics in `last-install.log` and shows the disk-space cause first when a native build runs out of space.

macOS produces:

- `osAi-<version>-mac-arm64.dmg`
- `osAi-<version>-mac-x64.dmg`

The macOS build disables certificate discovery and does not sign or notarize the application. macOS may therefore show a Gatekeeper warning when the downloaded app is first opened.
