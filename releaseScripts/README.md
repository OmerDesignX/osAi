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

Each script synchronizes the version, installs locked dependencies, checks formatting, runs the tests, verifies the native package, copies the finished installer into `release-assets/<platform>`, and removes the intermediate `release/` directory.

macOS produces:

- `osAi-<version>-mac-arm64.dmg`
- `osAi-<version>-mac-x64.dmg`

The macOS build disables certificate discovery and does not sign or notarize the application. macOS may therefore show a Gatekeeper warning when the downloaded app is first opened.
