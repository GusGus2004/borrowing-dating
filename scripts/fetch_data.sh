#!/usr/bin/env bash
# Download raw CLDF datasets at pinned commits into data/raw/. Never edit them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"

# name  github-repo  commit
DATASETS=(
  "kaufmanpmed lexibank/kaufmanpmed a774237b76081437389148a3dc007309679e38e2"
)

for entry in "${DATASETS[@]}"; do
  read -r name repo sha <<<"$entry"
  dest="$RAW/$name"
  if [[ -f "$dest/.commit" && "$(cat "$dest/.commit")" == "$sha" ]]; then
    echo "$name @ ${sha:0:7} already present"
    continue
  fi
  [[ -d "$dest" ]] && chmod -R u+w "$dest"
  rm -rf "$dest"
  mkdir -p "$dest"
  curl -fsSL "https://codeload.github.com/$repo/tar.gz/$sha" | tar xz -C "$dest" --strip-components=1
  echo "$sha" > "$dest/.commit"
  chmod -R a-w "$dest"/cldf
  echo "fetched $name @ ${sha:0:7}"
done
