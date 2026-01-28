#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBTORCH_DIR="${LIBTORCH_DIR:-$ROOT_DIR/tools/libtorch-cpu}"
BUILD_DIR="$ROOT_DIR/build"
DIST_DIR="$ROOT_DIR/dist"
APPDIR="$DIST_DIR/AppDir"
APPIMAGE_NAME="HexProject-CPU-x86_64.AppImage"
CLEAN_BUILD="${CLEAN_BUILD:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

require_cmd cmake
require_cmd linuxdeploy
require_cmd appimagetool
require_cmd unzip
require_cmd python3

download_libtorch_cpu() {
  local url="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip"
  local tools_dir="$ROOT_DIR/tools"
  local zip_path="$tools_dir/libtorch-cpu.zip"
  mkdir -p "$tools_dir"

  if command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$zip_path"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$zip_path"
  else
    echo "Missing command: wget or curl" >&2
    exit 1
  fi

  rm -rf "$tools_dir/libtorch"
  unzip -q "$zip_path" -d "$tools_dir"
  rm -f "$zip_path"
  if [ -d "$tools_dir/libtorch" ]; then
    rm -rf "$LIBTORCH_DIR"
    mv "$tools_dir/libtorch" "$LIBTORCH_DIR"
  fi
}

make_icon() {
  local src="$ROOT_DIR/assets/HEX.png"
  local dst="$APPDIR/hexproject.png"
  local size="${1:-512}"
  python3 - <<PY
from PIL import Image
import sys
src = "$src"
dst = "$dst"
size = int("$size")
img = Image.open(src).convert("RGBA")
img = img.resize((size, size), Image.LANCZOS)
img.save(dst)
PY
}

clean_build_dir() {
  if [ ! -d "$BUILD_DIR" ]; then
    return
  fi
  local cache="$BUILD_DIR/CMakeCache.txt"
  if [ -f "$cache" ]; then
    local cached_root
    cached_root="$(grep -m1 '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$cache" | cut -d= -f2- || true)"
    if [ -n "$cached_root" ] && [ "$cached_root" != "$ROOT_DIR" ]; then
      echo "Build dir was generated from a different source:"
      echo "  $cached_root"
      rm -rf "$BUILD_DIR"
      return
    fi
  fi
  if [ "$CLEAN_BUILD" = "1" ]; then
    rm -rf "$BUILD_DIR"
  fi
}

if [ ! -d "$LIBTORCH_DIR" ]; then
  echo "LIBTORCH_DIR not found: $LIBTORCH_DIR"
  echo "Downloading LibTorch CPU into: $LIBTORCH_DIR"
  download_libtorch_cpu
fi

if [ ! -d "$LIBTORCH_DIR" ]; then
  echo "LibTorch CPU download failed." >&2
  exit 1
fi

clean_build_dir
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR"
cmake --build "$BUILD_DIR" --config Release

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/assets" "$APPDIR/usr/config" "$APPDIR/usr/scripts/models" "$APPDIR/usr/lib"

cp "$BUILD_DIR/hex_ui" "$APPDIR/usr/bin/"
if [ -f "$BUILD_DIR/hex" ]; then
  cp "$BUILD_DIR/hex" "$APPDIR/usr/bin/"
fi

cp -a "$ROOT_DIR/assets/." "$APPDIR/usr/assets/"
if [ -d "$ROOT_DIR/config" ]; then
  cp -a "$ROOT_DIR/config/." "$APPDIR/usr/config/"
fi
if [ -d "$ROOT_DIR/scripts/models" ]; then
  cp -a "$ROOT_DIR/scripts/models/." "$APPDIR/usr/scripts/models/"
fi

if [ -d "$LIBTORCH_DIR/lib" ]; then
  cp -a "$LIBTORCH_DIR/lib/"*.so* "$APPDIR/usr/lib/" 2>/dev/null || true
fi

cat > "$APPDIR/HexProject.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=HexProject
Exec=hex_ui
Icon=hexproject
Categories=Game;
EOF

make_icon 512

cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$HERE/usr/lib:$LD_LIBRARY_PATH"
cd "$HERE/usr/bin"
exec "$HERE/usr/bin/hex_ui" "$@"
EOF
chmod +x "$APPDIR/AppRun"

linuxdeploy --appdir "$APPDIR" \
  --executable "$APPDIR/usr/bin/hex_ui" \
  --desktop-file "$APPDIR/HexProject.desktop" \
  --icon-file "$APPDIR/hexproject.png"

if [ -f "$APPDIR/usr/bin/hex" ]; then
  linuxdeploy --appdir "$APPDIR" --executable "$APPDIR/usr/bin/hex"
fi

mkdir -p "$DIST_DIR"
appimagetool "$APPDIR" "$DIST_DIR/$APPIMAGE_NAME"

echo "AppImage created: $DIST_DIR/$APPIMAGE_NAME"
