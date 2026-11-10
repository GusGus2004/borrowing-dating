#!/usr/bin/env bash
# Install a pinned BEAST2 (with its bundled JRE) into env/beast2/, plus the
# packages needed for lexical analyses, into env/beast2-packages/. Idempotent.
set -euo pipefail

BEAST_VERSION="2.7.7"   # last release with full per-platform bundles (incl. JRE)
BABEL_VERSION="0.4.0"   # SDollo (MutationDeathModel) + lexical models; latest release
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/beast2"
URL_BASE="https://github.com/CompEvol/beast2/releases/download/v${BEAST_VERSION}"

if [[ -x "$DEST/bin/beast" ]]; then
  echo "BEAST2 already installed at $DEST"
else
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  case "$(uname -s)-$(uname -m)" in
    Darwin-*)
      curl -fL "$URL_BASE/BEAST.v${BEAST_VERSION}.Mac.dmg" -o "$tmp/beast.dmg"
      hdiutil attach -nobrowse -readonly -mountpoint "$tmp/mnt" "$tmp/beast.dmg" >/dev/null
      src="$(find "$tmp/mnt" -maxdepth 1 -type d -name 'BEAST*' | head -1)"
      cp -R "$src" "$DEST"
      hdiutil detach "$tmp/mnt" >/dev/null
      ;;
    Linux-x86_64)
      curl -fL "$URL_BASE/BEAST.v${BEAST_VERSION}.Linux.x86.tgz" | tar xz -C "$tmp"
      mv "$tmp/beast" "$DEST"
      ;;
    Linux-aarch64)
      curl -fL "$URL_BASE/BEAST.v${BEAST_VERSION}.Linux.aarch64.tgz" | tar xz -C "$tmp"
      mv "$tmp/beast" "$DEST"
      ;;
    *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
  esac
fi

# Packages go in a project-local directory, not ~/Library or ~/.beast.
PKG="$HERE/beast2-packages"
mkdir -p "$PKG"
"$DEST/bin/packagemanager" -dir "$PKG" -add Babel -version "$BABEL_VERSION"
"$DEST/bin/packagemanager" -dir "$PKG" -list | grep -E '^(BEAST|Babel)' || true
echo "BEAST2 ${BEAST_VERSION} ready: $DEST/bin/beast  (packages: $PKG)"
