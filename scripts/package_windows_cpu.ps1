param(
  [string]$LibTorchDir = $env:LIBTORCH_DIR,
  [string]$SfmlDir = $env:SFML_DIR,
  [string]$SfmlBinDir = $env:SFML_BIN_DIR
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$BuildDir = Join-Path $Root "build"
$StageDir = Join-Path $Root "dist\windows\HexProject"
$InstallerDir = Join-Path $Root "dist"
$CleanBuild = $env:CLEAN_BUILD
if (-not $CleanBuild) { $CleanBuild = "1" }

function Ensure-LibTorchCpu {
  param([string]$TargetDir)
  if (Test-Path $TargetDir) { return }
  $toolsDir = Join-Path $Root "tools"
  New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
  $zipPath = Join-Path $toolsDir "libtorch-cpu.zip"
  $url = "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip"
  Write-Host "Downloading LibTorch CPU..."
  Invoke-WebRequest $url -OutFile $zipPath
  if (Test-Path (Join-Path $toolsDir "libtorch")) {
    Remove-Item (Join-Path $toolsDir "libtorch") -Recurse -Force
  }
  Expand-Archive $zipPath -DestinationPath $toolsDir -Force
  Remove-Item $zipPath -Force
  Move-Item (Join-Path $toolsDir "libtorch") $TargetDir -Force
}

if (-not $LibTorchDir) {
  $LibTorchDir = Join-Path $Root "tools\libtorch-cpu"
}

Ensure-LibTorchCpu -TargetDir $LibTorchDir

if (-not (Test-Path $LibTorchDir)) {
  Write-Error "LibTorch CPU not found at $LibTorchDir"
  exit 1
}

function Clean-BuildDir {
  param([string]$Dir, [string]$RootDir, [string]$CleanFlag)
  if (-not (Test-Path $Dir)) { return }
  $cache = Join-Path $Dir "CMakeCache.txt"
  if (Test-Path $cache) {
    $line = Select-String -Path $cache -Pattern "^CMAKE_HOME_DIRECTORY:INTERNAL=" -SimpleMatch | Select-Object -First 1
    if ($line) {
      $cachedRoot = $line.Line.Substring($line.Line.IndexOf("=") + 1)
      if ($cachedRoot -ne $RootDir) {
        Write-Host "Build dir was generated from a different source:"
        Write-Host "  $cachedRoot"
        Remove-Item $Dir -Recurse -Force
        return
      }
    }
  }
  if ($CleanFlag -eq "1") {
    Remove-Item $Dir -Recurse -Force
  }
}

Clean-BuildDir -Dir $BuildDir -RootDir $Root -CleanFlag $CleanBuild

$cmakeArgs = @("-S", $Root, "-B", $BuildDir, "-DCMAKE_PREFIX_PATH=$LibTorchDir")
if ($SfmlDir) { $cmakeArgs += "-DSFML_DIR=$SfmlDir" }

cmake @cmakeArgs
cmake --build $BuildDir --config Release

if (Test-Path $StageDir) { Remove-Item $StageDir -Recurse -Force }
New-Item -ItemType Directory -Path $StageDir | Out-Null

$binDir = Join-Path $BuildDir "Release"
Copy-Item "$binDir\hex_ui.exe" $StageDir
if (Test-Path "$binDir\hex.exe") { Copy-Item "$binDir\hex.exe" $StageDir }

Copy-Item "$Root\assets" "$StageDir\assets" -Recurse
if (Test-Path "$Root\config") { Copy-Item "$Root\config" "$StageDir\config" -Recurse }
if (Test-Path "$Root\scripts\models") {
  New-Item -ItemType Directory -Path "$StageDir\scripts\models" -Force | Out-Null
  Copy-Item "$Root\scripts\models\*" "$StageDir\scripts\models" -Recurse -Force
}

Copy-Item "$LibTorchDir\lib\*.dll" $StageDir -Force

$sfmlBin = $SfmlBinDir
if (-not $sfmlBin -and $SfmlDir) {
  $candidate = Join-Path $SfmlDir "bin"
  if (Test-Path $candidate) { $sfmlBin = $candidate }
}
if ($sfmlBin -and (Test-Path $sfmlBin)) {
  Copy-Item "$sfmlBin\*.dll" $StageDir -Force
}

$env:HEXPROJECT_APPDIR = $StageDir

$iscc = "iscc.exe"
if (-not (Get-Command $iscc -ErrorAction SilentlyContinue)) {
  Write-Error "Inno Setup not found. Install it and ensure iscc.exe is on PATH."
  exit 1
}

& $iscc (Join-Path $Root "scripts\installer_windows.iss")

Write-Host "Installer generated in $InstallerDir"
