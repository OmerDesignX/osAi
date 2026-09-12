[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $PSCommandPath
$buildScript = Join-Path $scriptDirectory "build-windows.sh"
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDirectory)

function Assert-BuildToolPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $toolRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "build\release-tools\windows"))
  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $prefix = $toolRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
  if (-not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to modify an unexpected build-tool path: $fullPath"
  }
  return $fullPath
}

function Remove-BuildToolItem {
  param([Parameter(Mandatory = $true)][string]$Path)

  $safePath = Assert-BuildToolPath -Path $Path
  $item = Get-Item -LiteralPath $safePath -Force -ErrorAction SilentlyContinue
  if (-not $item) { return }
  if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Refusing to remove a linked build-tool path: $safePath"
  }
  Remove-Item -LiteralPath $safePath -Recurse -Force -ErrorAction Stop
}

function Get-BuildToolSha256 {
  param([Parameter(Mandatory = $true)][string]$Path)

  $stream = [System.IO.File]::OpenRead($Path)
  $hasher = [System.Security.Cryptography.SHA256]::Create()
  try {
    return ([System.BitConverter]::ToString($hasher.ComputeHash($stream))).Replace("-", "")
  }
  finally {
    $hasher.Dispose()
    $stream.Dispose()
  }
}

function Get-VerifiedBuildToolArchive {
  param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][string]$Sha256,
    [Parameter(Mandatory = $true)][long]$MaximumBytes
  )

  $destinationPath = Assert-BuildToolPath -Path $Destination
  $existing = Get-Item -LiteralPath $destinationPath -ErrorAction SilentlyContinue
  if ($existing -and $existing.Length -le $MaximumBytes) {
    $existingHash = Get-BuildToolSha256 -Path $destinationPath
    if ($existingHash -eq $Sha256) { return }
  }
  if ($existing) { Remove-BuildToolItem -Path $destinationPath }

  $temporary = Assert-BuildToolPath -Path ($destinationPath + ".part")
  Remove-BuildToolItem -Path $temporary
  Write-Host "Downloading $(Split-Path -Leaf $destinationPath)"
  $oldProgress = $ProgressPreference
  try {
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $Url -OutFile $temporary -UseBasicParsing -ErrorAction Stop
  }
  finally {
    $ProgressPreference = $oldProgress
  }
  $download = Get-Item -LiteralPath $temporary -ErrorAction Stop
  if ($download.Length -lt 1 -or $download.Length -gt $MaximumBytes) {
    Remove-BuildToolItem -Path $temporary
    throw "The downloaded build-tool archive has an unexpected size"
  }
  $downloadHash = Get-BuildToolSha256 -Path $temporary
  if ($downloadHash -ne $Sha256) {
    Remove-BuildToolItem -Path $temporary
    throw "The downloaded build-tool archive failed SHA-256 verification"
  }
  Move-Item -LiteralPath $temporary -Destination $destinationPath -ErrorAction Stop
}

function Expand-VerifiedBuildToolArchive {
  param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][string]$ExpectedRelativePath
  )

  $destinationPath = Assert-BuildToolPath -Path $Destination
  $expected = Join-Path $destinationPath $ExpectedRelativePath
  if (Test-Path -LiteralPath $expected -PathType Leaf) { return }
  Remove-BuildToolItem -Path $destinationPath
  New-Item -ItemType Directory -Path $destinationPath -ErrorAction Stop | Out-Null
  Expand-Archive -LiteralPath $Archive -DestinationPath $destinationPath -ErrorAction Stop
  if (-not (Test-Path -LiteralPath $expected -PathType Leaf)) {
    throw "The verified build-tool archive did not contain $ExpectedRelativePath"
  }
}

function Enable-WindowsBuildToolchain {
  if ($env:OSAI_CMAKE -and (Test-Path -LiteralPath $env:OSAI_CMAKE -PathType Leaf)) {
    return
  }
  $systemCmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
  if ($systemCmake) { return }

  $toolRoot = Join-Path $projectRoot "build\release-tools\windows"
  New-Item -ItemType Directory -Path $toolRoot -Force -ErrorAction Stop | Out-Null
  $cmakeArchive = Join-Path $toolRoot "cmake-4.4.0-windows-x86_64.zip"
  $llvmArchive = Join-Path $toolRoot "llvm-mingw-20260616-ucrt-x86_64.zip"
  $ninjaArchive = Join-Path $toolRoot "ninja-1.13.2-windows-x86_64.zip"

  Get-VerifiedBuildToolArchive -Url "https://github.com/Kitware/CMake/releases/download/v4.4.0/cmake-4.4.0-windows-x86_64.zip" -Destination $cmakeArchive -Sha256 "156D70EB7625A7B469444DF7D0861D2AF8D5D0A437FCE32C350372B08F5620E8" -MaximumBytes 100MB
  Get-VerifiedBuildToolArchive -Url "https://github.com/mstorsjo/llvm-mingw/releases/download/20260616/llvm-mingw-20260616-ucrt-x86_64.zip" -Destination $llvmArchive -Sha256 "B9B68A4D276E16FA25802AABA458E4638F64B3884C290AACCDC2D87083B6CA35" -MaximumBytes 300MB
  Get-VerifiedBuildToolArchive -Url "https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-win.zip" -Destination $ninjaArchive -Sha256 "07FC8261B42B20E71D1720B39068C2E14FFCEE6396B76FB7A795FB460B78DC65" -MaximumBytes 10MB

  $cmakeRoot = Join-Path $toolRoot "cmake"
  $llvmRoot = Join-Path $toolRoot "llvm-mingw"
  $ninjaRoot = Join-Path $toolRoot "ninja"
  Expand-VerifiedBuildToolArchive -Archive $cmakeArchive -Destination $cmakeRoot -ExpectedRelativePath "cmake-4.4.0-windows-x86_64\bin\cmake.exe"
  Expand-VerifiedBuildToolArchive -Archive $llvmArchive -Destination $llvmRoot -ExpectedRelativePath "llvm-mingw-20260616-ucrt-x86_64\bin\clang++.exe"
  Expand-VerifiedBuildToolArchive -Archive $ninjaArchive -Destination $ninjaRoot -ExpectedRelativePath "ninja.exe"

  $cmakeBin = Join-Path $cmakeRoot "cmake-4.4.0-windows-x86_64\bin"
  $llvmBin = Join-Path $llvmRoot "llvm-mingw-20260616-ucrt-x86_64\bin"
  $env:OSAI_CMAKE = Join-Path $cmakeBin "cmake.exe"
  $env:CC = Join-Path $llvmBin "clang.exe"
  $env:CXX = Join-Path $llvmBin "clang++.exe"
  $env:CMAKE_GENERATOR = "Ninja"
  $env:Path = "$cmakeBin;$llvmBin;$ninjaRoot;$env:Path"
  Write-Host "Using the pinned portable Windows release toolchain."
}

function Find-GitBash {
  $candidates = [System.Collections.Generic.List[string]]::new()

  if ($env:OSAI_GIT_BASH) {
    $candidates.Add($env:OSAI_GIT_BASH)
  }

  $git = Get-Command git.exe -ErrorAction SilentlyContinue
  if ($git) {
    $gitRoot = Split-Path -Parent (Split-Path -Parent $git.Source)
    $candidates.Add((Join-Path $gitRoot "bin\bash.exe"))
  }

  if ($env:ProgramFiles) {
    $candidates.Add((Join-Path $env:ProgramFiles "Git\bin\bash.exe"))
  }
  if (${env:ProgramFiles(x86)}) {
    $candidates.Add((Join-Path ${env:ProgramFiles(x86)} "Git\bin\bash.exe"))
  }
  if ($env:LOCALAPPDATA) {
    $candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Git\bin\bash.exe"))
  }

  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }

  throw @"
Git Bash was not found. Install Git for Windows, reopen PowerShell, and run:
  .\releaseScripts\windows\build-windows.cmd

If Git Bash is installed somewhere unusual, set OSAI_GIT_BASH to the full
path of its bash.exe before running this script.
"@
}

if (-not (Test-Path -LiteralPath $buildScript -PathType Leaf)) {
  throw "The shared Windows build script is missing: $buildScript"
}

$gitBash = Find-GitBash
Enable-WindowsBuildToolchain
Write-Host "Building one osAi x64 installer for Windows 10 and Windows 11."
Write-Host "Using Git Bash: $gitBash"

try {
  Push-Location -LiteralPath $projectRoot
  & $gitBash -lc "bash releaseScripts/windows/build-windows.sh"
  if ($LASTEXITCODE -ne 0) {
    throw "The Windows release build failed with exit code $LASTEXITCODE."
  }
}
finally {
  Pop-Location
}

