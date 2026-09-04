#!/usr/bin/env bash
#
# Fetch and install the Vaelor NPU assistant model (Qwen3.5-4B-NPU2) from the
# GitHub release. The model is ~3.35 GB, so it ships as split <2 GB parts (a
# GitHub release asset is capped at 2 GB). This script downloads the parts,
# reassembles the tar, verifies its checksum, and unpacks it into the
# FastFlowLM (lemonade-server) models directory.
#
# The Vaelor repo is PRIVATE, so downloading release assets needs GitHub auth.
# Provide it one of two ways:
#   - have `gh` installed and authenticated (`gh auth login`), OR
#   - export GH_TOKEN with a token that can read the repo.
#
# Usage:  deploy/fetch-npu-model.sh
# Env:    VAELOR_REPO (default ShadowLayer90/vaelor)
#         VAELOR_RELEASE_TAG (default v1.0b1)
#         FLM_MODELS_DIR (default the lemonade-server flm models dir)
set -Eeuo pipefail

REPO="${VAELOR_REPO:-ShadowLayer90/vaelor}"
TAG="${VAELOR_RELEASE_TAG:-v1.0b1}"
MODELS_DIR="${FLM_MODELS_DIR:-/var/snap/lemonade-server/common/.config/flm/models}"
MODEL_NAME="Qwen3.5-4B-NPU2"
PART_GLOB="qwen35-4b-npu2.tar.part*"
EXPECT_SHA="80401bba2fc26896a7cac8ef23629ee495b20b1b853bfb9ce5202562772ca0e4"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching ${MODEL_NAME} from ${REPO} release ${TAG} ..."
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$PART_GLOB" --dir "$tmp"
elif [[ -n "${GH_TOKEN:-}" ]]; then
  api="https://api.github.com/repos/${REPO}"
  meta="$(curl -fsSL -H "Authorization: Bearer $GH_TOKEN" \
              -H "Accept: application/vnd.github+json" \
              "${api}/releases/tags/${TAG}")"
  # Pair each asset id with its name; download the model parts by id (the id
  # endpoint with the octet-stream Accept is what serves private assets).
  echo "$meta" \
    | grep -oE '"id": [0-9]+|"name": "[^"]+"' \
    | paste - - \
    | while read -r id_field name_field; do
        id="${id_field##*: }"; name="${name_field##*: \"}"; name="${name%\"}"
        case "$name" in
          qwen35-4b-npu2.tar.part*)
            echo "  - $name"
            curl -fsSL -H "Authorization: Bearer $GH_TOKEN" \
                 -H "Accept: application/octet-stream" \
                 "${api}/releases/assets/${id}" -o "${tmp}/${name}"
            ;;
        esac
      done
else
  echo "ERROR: need an authenticated 'gh' or a GH_TOKEN to read the private release." >&2
  exit 1
fi

shopt -s nullglob
parts=("${tmp}"/qwen35-4b-npu2.tar.part*)
if (( ${#parts[@]} == 0 )); then
  echo "ERROR: no model parts were downloaded." >&2
  exit 1
fi

echo "Reassembling ${#parts[@]} part(s) ..."
# Sort so partNN concatenates in order regardless of shell glob ordering.
cat $(printf '%s\n' "${parts[@]}" | sort) > "${tmp}/model.tar"

echo "Verifying checksum ..."
actual="$(sha256sum "${tmp}/model.tar" | cut -d' ' -f1)"
if [[ "$actual" != "$EXPECT_SHA" ]]; then
  echo "ERROR: checksum mismatch (expected ${EXPECT_SHA}, got ${actual})." >&2
  exit 1
fi

echo "Installing into ${MODELS_DIR} ..."
sudo mkdir -p "$MODELS_DIR"
sudo tar -C "$MODELS_DIR" -xf "${tmp}/model.tar"

echo "Done. NPU model installed at ${MODELS_DIR}/${MODEL_NAME}."
