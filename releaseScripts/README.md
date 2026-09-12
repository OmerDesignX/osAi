# Local release builds

Set the release number in `releaseScripts/VERSION.txt`, then run the native script on each target system:

```sh
# macOS 12 or newer — builds Apple Silicon and Intel DMGs
bash releaseScripts/macos/build.sh

# Windows 10 or 11
.\releaseScripts\windows\build-windows.cmd

# Current x64 Debian or Ubuntu
bash releaseScripts/linux/build.sh
```

Each script synchronizes the version, installs locked dependencies, downloads and SHA-256 records the osAi CLI source, compiles the required llama.cpp executables, creates an offline Python wheelhouse, downloads and verifies the pinned CPython runtime, runs the checks, verifies the native package, stages the installer in `release-assets/<platform>`, and removes intermediate output. On Windows, the script downloads a pinned, SHA-256-verified portable CMake, Ninja, and LLVM-MinGW toolchain when CMake is unavailable. macOS and Linux release builders need CMake and their native C/C++ toolchain. People installing the finished App do not need build tools.

macOS produces:

- `osAi-<version>-mac-arm64.dmg`
- `osAi-<version>-mac-x64.dmg`

The macOS build disables certificate discovery and does not sign or notarize the application. macOS may therefore show a Gatekeeper warning when the downloaded app is first opened.
