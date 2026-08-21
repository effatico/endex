#!/usr/bin/env sh
# endex installer — downloads a prebuilt binary from GitHub Releases.
#
#   curl -fsSL https://raw.githubusercontent.com/effatico/endex/main/install.sh | sh
#
# Options (env vars):
#   ENDEX_VERSION   tag to install (default: latest)
#   ENDEX_PREFIX    install dir     (default: /usr/local/bin, or ~/.local/bin without sudo)
set -eu

REPO="effatico/endex"
BIN="endex"

say()  { printf '%s\n' "$*"; }
fail() { say "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "missing required tool: $1"; }

need curl
need uname

OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
  Linux)  os_part="unknown-linux-gnu" ;;
  Darwin) os_part="apple-darwin" ;;
  *)      fail "unsupported OS: $OS (on Windows, download endex-x86_64-pc-windows-msvc.zip from Releases)" ;;
esac

case "$ARCH" in
  x86_64|amd64)  arch_part="x86_64" ;;
  arm64|aarch64) arch_part="aarch64" ;;
  *)             fail "unsupported architecture: $ARCH" ;;
esac

TARGET="${arch_part}-${os_part}"

# Resolve version.
if [ "${ENDEX_VERSION:-}" ]; then
  VERSION="$ENDEX_VERSION"
else
  say "Resolving latest release..."
  VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | head -1 | cut -d'"' -f4) \
    || fail "could not determine latest release"
fi
[ -n "$VERSION" ] || fail "no release found"
say "Installing endex $VERSION ($TARGET)..."

ASSET="endex-${TARGET}.tar.gz"
BASE="https://github.com/$REPO/releases/download/$VERSION"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -fsSL "$BASE/$ASSET"        -o "$TMP/$ASSET"        || fail "download failed: $BASE/$ASSET"
curl -fsSL "$BASE/$ASSET.sha256" -o "$TMP/$ASSET.sha256" || fail "checksum download failed"

# Verify checksum.
cd "$TMP"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c "$ASSET.sha256" >/dev/null || fail "checksum mismatch"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c "$ASSET.sha256" >/dev/null || fail "checksum mismatch"
else
  say "warning: no sha256sum/shasum found; skipping checksum verification"
fi

tar -xzf "$ASSET"

# Pick an install dir.
if [ "${ENDEX_PREFIX:-}" ]; then
  DEST="$ENDEX_PREFIX"
elif [ -w /usr/local/bin ]; then
  DEST="/usr/local/bin"
elif command -v sudo >/dev/null 2>&1; then
  DEST="/usr/local/bin"
else
  DEST="$HOME/.local/bin"
fi
mkdir -p "$DEST" 2>/dev/null || true

if [ -w "$DEST" ]; then
  mv "$BIN" "$DEST/$BIN"
else
  say "Installing to $DEST requires sudo..."
  sudo mv "$BIN" "$DEST/$BIN"
fi
chmod +x "$DEST/$BIN" 2>/dev/null || sudo chmod +x "$DEST/$BIN"

say "endex installed to $DEST/$BIN"
case ":$PATH:" in
  *":$DEST:"*) : ;;
  *) say "note: $DEST is not on your PATH — add it, e.g.:  export PATH=\"$DEST:\$PATH\"" ;;
esac

say ""
say "Register with Claude Code:"
say "  claude mcp add endex -- $DEST/$BIN mcp /path/to/your/repo"
say "  (add -e EMBED_PROVIDER=... -e EMBED_URL=... -e EMBED_MODEL=... for semantic search)"
