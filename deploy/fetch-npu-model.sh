#!/usr/bin/env bash
#
# Fetch and install the Vaelor NPU assistant model (Qwen3.5-4B-NPU2) from the
# GitHub release. The model is ~3.35 GB, so it ships as split <2 GB parts (a
# GitHub release asset is capped at 2 GB). This script downloads the parts,
# reassembles the tar, verifies its checksum, and unpacks it into the
# FastFlowLM models directory (/var/lib/vaelor/flm/models).
#
# install-vaelor.sh runs this automatically (install_npu_model) so a fresh box is
# turnkey; it is also runnable standalone to (re)fetch the model on its own - e.g.
# after installing with --without-npu-model. Standalone it uses sudo to write the
# models dir; invoked by the root installer it writes directly.
#
# On a PUBLIC repo the release assets download with no auth, which is the default
# path and needs nothing installed. If the repo is PRIVATE, provide GitHub auth
# one of two ways:
#   - have `gh` installed and authenticated (`gh auth login`), OR
#   - export GH_TOKEN with a token that can read the repo.
#
# Usage:  deploy/fetch-npu-model.sh
# Env:    VAELOR_REPO (default ShadowLayer90/vaelor)
#         VAELOR_RELEASE_TAG (default v1.0b2)
#         FLM_MODELS_DIR (default /var/lib/vaelor/flm/models)
set -Eeuo pipefail

REPO="${VAELOR_REPO:-ShadowLayer90/vaelor}"
TAG="${VAELOR_RELEASE_TAG:-v1.0b2}"
MODELS_DIR="${FLM_MODELS_DIR:-/var/lib/vaelor/flm/models}"
MODEL_NAME="Qwen3.5-4B-NPU2"
PART_GLOB="qwen35-4b-npu2.tar.part*"
EXPECT_SHA="80401bba2fc26896a7cac8ef23629ee495b20b1b853bfb9ce5202562772ca0e4"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Install into a root-owned models dir. Run standalone this needs sudo; when the
# main installer calls this script it is already root, and a minimal box may have
# no sudo at all - so use it only when we are not already root.
run_privileged() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "ERROR: writing ${MODELS_DIR} needs root; re-run as root or install sudo." >&2
    exit 1
  fi
}

echo "Fetching ${MODEL_NAME} from ${REPO} release ${TAG} ..."
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$PART_GLOB" --dir "$tmp"
elif [[ -n "${GH_TOKEN:-}" ]]; then
  # Resolve the release-asset ids with a JSON parser, not grep|paste|while: the
  # release JSON carries id/name tokens for the release, its author and each
  # asset's uploader, so pairing "id" with "name" line-by-line mis-pairs them
  # and 404s on the wrong asset id. python3 is a hard install dependency, so a
  # real JSON walk that reads exactly each asset's own id+name is safe to rely
  # on. Each part is streamed by its id (the id endpoint with the octet-stream
  # Accept is what serves private assets).
  REPO="$REPO" TAG="$TAG" DEST="$tmp" python3 - <<'PY'
import json
import os
import sys
import urllib.request

token = os.environ["GH_TOKEN"]
repo = os.environ["REPO"]
tag = os.environ["TAG"]
dest = os.environ["DEST"]
api = "https://api.github.com/repos/{}".format(repo)


def fetch(url, accept):
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer {}".format(token),
            "Accept": accept,
            "User-Agent": "vaelor-fetch-npu-model",
        },
    )
    return urllib.request.urlopen(request)


with fetch("{}/releases/tags/{}".format(api, tag),
           "application/vnd.github+json") as response:
    release = json.load(response)

count = 0
for asset in release.get("assets", []):
    name = str(asset.get("name", ""))
    if not name.startswith("qwen35-4b-npu2.tar.part"):
        continue
    print("  - {}".format(name))
    url = "{}/releases/assets/{}".format(api, asset["id"])
    with fetch(url, "application/octet-stream") as response, \
            open(os.path.join(dest, name), "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    count += 1

if count == 0:
    sys.stderr.write("No model parts found in release {}.\n".format(tag))
    sys.exit(1)
PY
else
  # No gh and no GH_TOKEN: a PUBLIC repo needs neither. List the release assets
  # through the anonymous API and download each part by its public URL. If the
  # repo is private the API returns 404, and the message says how to authenticate
  # - so a private repo still gets a clear instruction rather than a raw error.
  echo "  (no gh / GH_TOKEN - downloading the public release assets)"
  REPO="$REPO" TAG="$TAG" DEST="$tmp" python3 - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

repo = os.environ["REPO"]
tag = os.environ["TAG"]
dest = os.environ["DEST"]
api = "https://api.github.com/repos/{}/releases/tags/{}".format(repo, tag)


def get(url, accept="application/vnd.github+json"):
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": "vaelor-fetch-npu-model"},
    )
    return urllib.request.urlopen(request)


try:
    with get(api) as response:
        release = json.load(response)
except urllib.error.HTTPError as error:
    if error.code in (401, 403, 404):
        sys.stderr.write(
            "Cannot read release {} of {} without auth (HTTP {}). If the repo "
            "is private, install and authenticate 'gh', or export GH_TOKEN.\n"
            .format(tag, repo, error.code)
        )
    else:
        sys.stderr.write("Release lookup failed: HTTP {}.\n".format(error.code))
    sys.exit(1)

count = 0
for asset in release.get("assets", []):
    name = str(asset.get("name", ""))
    if not name.startswith("qwen35-4b-npu2.tar.part"):
        continue
    url = asset.get("browser_download_url")
    if not url:
        continue
    print("  - {}".format(name))
    with get(url, "application/octet-stream") as response, \
            open(os.path.join(dest, name), "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    count += 1

if count == 0:
    sys.stderr.write("No model parts found in release {}.\n".format(tag))
    sys.exit(1)
PY
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
# Create with an explicit 0755 mode rather than `mkdir -p`: under a restrictive
# umask (e.g. 0077) a plain mkdir yields 0700, which the root installer can write
# but `vaelor`/`vaelor-workloads` cannot traverse or list - so the live
# `npu_model_present()` check reads an empty/inaccessible dir and the model looks
# absent. `install -d -m 0755` fixes the mode regardless of umask.
run_privileged install -d -m 0755 "$MODELS_DIR"
run_privileged tar -C "$MODELS_DIR" -xf "${tmp}/model.tar"

echo "Done. NPU model installed at ${MODELS_DIR}/${MODEL_NAME}."
