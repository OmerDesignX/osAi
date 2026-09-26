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

Each script synchronizes `package.json` from the root version file, installs locked App dependencies, downloads and verifies the pinned CPython runtime, runs checks, builds and verifies the native package, stages it in `release-assets/<platform>`, and removes intermediate output. Release machines need Node.js 22 and pnpm 11; Windows also uses Git Bash. They do not need a CUDA or Vulkan SDK because the release does not compile or bundle llama.cpp.

On first launch, the installed App downloads the osAi CLI repository from the URL in `backend-source.json`, installs its Python dependencies, and compiles llama.cpp on that computer. The CLI code must be pushed to GitHub before a release can download those changes. CUDA compilation uses a compatible CUDA Toolkit when present. On Windows, setup uses Microsoft C++ Build Tools when present or downloads a verified portable compiler, and downloads a verified Vulkan SDK into its private build cache when a Vulkan runtime is present and the SDK is missing. If a combined CUDA and Vulkan build fails, setup tries CUDA, Vulkan, then CPU. macOS uses Metal when available; Apple silicon with macOS 14 or newer can also use MLX.

macOS produces:

- `osAi-<version>-mac-arm64.dmg`
- `osAi-<version>-mac-x64.dmg`

The macOS build disables certificate discovery and does not sign or notarize the application. macOS may therefore show a Gatekeeper warning when the downloaded app is first opened.
