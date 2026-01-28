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
  $url = "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip"
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
$binStage = Join-Path $StageDir "bin"
New-Item -ItemType Directory -Path $binStage | Out-Null

$binDir = Join-Path $BuildDir "Release"
Copy-Item "$binDir\hex_ui.exe" $binStage
if (Test-Path "$binDir\hex.exe") { Copy-Item "$binDir\hex.exe" $binStage }

Copy-Item "$Root\assets" "$StageDir\assets" -Recurse
if (Test-Path "$Root\config") { Copy-Item "$Root\config" "$StageDir\config" -Recurse }
if (Test-Path "$Root\scripts\models") {
  New-Item -ItemType Directory -Path "$StageDir\scripts\models" -Force | Out-Null
  Copy-Item "$Root\scripts\models\*" "$StageDir\scripts\models" -Recurse -Force
}

Copy-Item "$LibTorchDir\lib\*.dll" $binStage -Force

function Find-SfmlBin {
  param([string]$ExplicitBin, [string]$ExplicitDir, [string]$BuildDir)
  if ($ExplicitBin -and (Test-Path $ExplicitBin)) { return $ExplicitBin }

  if ($ExplicitDir) {
    $binFromDir = Join-Path $ExplicitDir "bin"
    if (Test-Path $binFromDir) { return $binFromDir }
    $candidate = Join-Path $ExplicitDir "..\..\..\bin"
    if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
  }

  $cache = Join-Path $BuildDir "CMakeCache.txt"
  if (Test-Path $cache) {
    $line = Select-String -Path $cache -Pattern "^SFML_DIR:PATH=" -SimpleMatch | Select-Object -First 1
    if ($line) {
      $cachedDir = $line.Line.Substring($line.Line.IndexOf("=") + 1)
      if ($cachedDir) {
        $binFromCache = Join-Path $cachedDir "..\..\..\bin"
        if (Test-Path $binFromCache) { return (Resolve-Path $binFromCache).Path }
      }
    }
  }

  $searchRoots = @("C:\SFML", "C:\Program Files\SFML", "C:\Program Files (x86)\SFML")
  foreach ($root in $searchRoots) {
    if (-not (Test-Path $root)) { continue }
    $dll = Get-ChildItem $root -Recurse -Filter "sfml-graphics-2.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($dll) { return $dll.DirectoryName }
  }

  return $null
}

$sfmlBin = Find-SfmlBin -ExplicitBin $SfmlBinDir -ExplicitDir $SfmlDir -BuildDir $BuildDir
if (-not $sfmlBin) {
  Write-Error "SFML DLLs not found. Set SFML_DIR or SFML_BIN_DIR, or install SFML."
  exit 1
}

Copy-Item "$sfmlBin\sfml-*.dll" $binStage -Force
if (Test-Path "$sfmlBin\openal32.dll") {
  Copy-Item "$sfmlBin\openal32.dll" $binStage -Force
}

$env:HEXPROJECT_APPDIR = $StageDir

$iscc = "iscc.exe"
if (-not (Get-Command $iscc -ErrorAction SilentlyContinue)) {
  Write-Error "Inno Setup not found. Install it and ensure iscc.exe is on PATH."
  exit 1
}

& $iscc (Join-Path $Root "scripts\installer_windows.iss")

Write-Host "Installer generated in $InstallerDir"
