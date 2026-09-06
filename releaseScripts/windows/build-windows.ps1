[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $PSCommandPath
$buildScript = Join-Path $scriptDirectory "build-windows.sh"
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDirectory)

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

