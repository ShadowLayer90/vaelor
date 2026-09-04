#!/usr/bin/env bash
set -euo pipefail

repository="https://github.com/b4rtaz/distributed-llama.git"
commit="59af889085c6c0316a4524a92b42f04caa4bcc6d"
output="${1:-dist/runtime-artifacts}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone --filter=blob:none --no-checkout "$repository" "$work/source"
git -C "$work/source" checkout --detach "$commit"
test "$(git -C "$work/source" rev-parse HEAD)" = "$commit"
make -C "$work/source" dllama dllama-api

architecture="$(uname -m)"
case "$architecture" in
  aarch64|arm64) architecture="arm64" ;;
  x86_64|amd64) architecture="amd64" ;;
  *) echo "Unsupported build architecture: $architecture" >&2; exit 1 ;;
esac

mkdir -p "$output"
tar -C "$work/source" -czf \
  "$output/distributed-llama-${commit}-${architecture}.tar.gz" \
  dllama dllama-api launch.py LICENSE
sha256sum "$output/distributed-llama-${commit}-${architecture}.tar.gz" \
  > "$output/distributed-llama-${commit}-${architecture}.tar.gz.sha256"
