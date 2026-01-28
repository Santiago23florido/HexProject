#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBTORCH_DIR="${LIBTORCH_DIR:-$HOME/libtorch}"
BUILD_DIR="$ROOT_DIR/build"
DIST_DIR="$ROOT_DIR/dist"
PKGROOT="$DIST_DIR/pkgroot"
VERSION="${VERSION:-1.0.0}"
ARCH="amd64"
PKGNAME="hexproject"
CLEAN_BUILD="${CLEAN_BUILD:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

require_cmd cmake
require_cmd dpkg-deb
require_cmd ldd
require_cmd python3

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

make_icon() {
  local src="$ROOT_DIR/assets/HEX.png"
  local dst="$PKGROOT/usr/share/icons/hicolor/256x256/apps/hexproject.png"
  python3 - <<PY
from PIL import Image
src = "$src"
dst = "$dst"
img = Image.open(src).convert("RGBA")
img = img.resize((256, 256), Image.LANCZOS)
img.save(dst)
PY
}

if [ ! -d "$LIBTORCH_DIR" ]; then
  echo "LIBTORCH_DIR not found: $LIBTORCH_DIR" >&2
  exit 1
fi

clean_build_dir
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR"
cmake --build "$BUILD_DIR" --config Release

rm -rf "$PKGROOT"
mkdir -p "$PKGROOT/DEBIAN"
mkdir -p "$PKGROOT/opt/hexproject/bin"
mkdir -p "$PKGROOT/opt/hexproject/assets"
mkdir -p "$PKGROOT/opt/hexproject/config"
mkdir -p "$PKGROOT/opt/hexproject/scripts/models"
mkdir -p "$PKGROOT/opt/hexproject/lib"
mkdir -p "$PKGROOT/usr/bin"
mkdir -p "$PKGROOT/usr/share/applications"
mkdir -p "$PKGROOT/usr/share/icons/hicolor/256x256/apps"

cp "$BUILD_DIR/hex_ui" "$PKGROOT/opt/hexproject/bin/"
if [ -f "$BUILD_DIR/hex" ]; then
  cp "$BUILD_DIR/hex" "$PKGROOT/opt/hexproject/bin/"
fi

cp -a "$ROOT_DIR/assets/." "$PKGROOT/opt/hexproject/assets/"
if [ -d "$ROOT_DIR/config" ]; then
  cp -a "$ROOT_DIR/config/." "$PKGROOT/opt/hexproject/config/"
fi
if [ -d "$ROOT_DIR/scripts/models" ]; then
  cp -a "$ROOT_DIR/scripts/models/." "$PKGROOT/opt/hexproject/scripts/models/"
fi

if [ -d "$LIBTORCH_DIR/lib" ]; then
  cp -a "$LIBTORCH_DIR/lib/"*.so* "$PKGROOT/opt/hexproject/lib/" 2>/dev/null || true
fi

if ldd "$BUILD_DIR/hex_ui" >/dev/null 2>&1; then
  ldd "$BUILD_DIR/hex_ui" | awk '/libsfml|libopenal/ {print $3}' | while read -r lib; do
    if [ -f "$lib" ]; then
      cp -a "$lib" "$PKGROOT/opt/hexproject/lib/"
    fi
  done
fi

cat > "$PKGROOT/usr/bin/hexproject" <<'EOF'
#!/usr/bin/env bash
export LD_LIBRARY_PATH="/opt/hexproject/lib:$LD_LIBRARY_PATH"
cd /opt/hexproject/bin
exec /opt/hexproject/bin/hex_ui "$@"
EOF
chmod +x "$PKGROOT/usr/bin/hexproject"

cat > "$PKGROOT/usr/share/applications/hexproject.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=HexProject
Exec=/usr/bin/hexproject
Icon=hexproject
Categories=Game;
EOF

make_icon

cat > "$PKGROOT/DEBIAN/control" <<EOF
Package: ${PKGNAME}
Version: ${VERSION}
Section: games
Priority: optional
Architecture: ${ARCH}
Maintainer: HexProject
Depends: libc6, libstdc++6
Description: HexProject (CPU build)
 CPU-only build of the HexProject game with bundled assets and libtorch.
EOF

mkdir -p "$DIST_DIR"
dpkg-deb --build "$PKGROOT" "$DIST_DIR/${PKGNAME}_${VERSION}_${ARCH}.deb"

echo "Deb package created: $DIST_DIR/${PKGNAME}_${VERSION}_${ARCH}.deb"
