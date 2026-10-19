#!/usr/bin/env bash
#
# Fetch and install the Vaelor NPU assistant model package from the GitHub
# release. WHICH package - its name, part names, release and SHA-256 - is the
# installed Vaelor package's serving pair (vaelor/npu_runtime_pin.py, VD-219),
# read through `python -I -m vaelor.npu_runtime_pin shell`, so this script and
# the console's Install action can never pin two different packages, and the
# package always matches the FastFlowLM runtime the same pin names. The package is several GB (its exact size is pinned with it, NPU_PACKAGE_BYTES), so it ships as split <2 GB parts (a
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
# Usage:  deploy/fetch-npu-model.sh   (after Vaelor is installed)
# Env:    VAELOR_REPO (default ShadowLayer90/vaelor)
#         VAELOR_RELEASE_TAG (default v1.5.4)
#         FLM_MODELS_DIR (default /var/lib/vaelor/flm/models)
#         VAELOR_PYTHON (default /opt/vaelor/venv/bin/python)
set -Eeuo pipefail

PIN_PYTHON="${VAELOR_PYTHON:-/opt/vaelor/venv/bin/python}"
# The pin is KEY=value lines of plain tokens; each is checked before it is used,
# never eval'd.
if ! pin_lines="$("${PIN_PYTHON}" -I -m vaelor.npu_runtime_pin shell)"; then
  echo "ERROR: the model package is named by the installed Vaelor package, and"     "${PIN_PYTHON} could not say which; install Vaelor first." >&2
  exit 1
fi
NPU_MODEL_NAME="" NPU_PART_PREFIX="" NPU_PACKAGE_SHA="" NPU_PACKAGE_BYTES=""
while IFS="=" read -r key value; do
  value="${value%$'\r'}"
  [[ "${value}" =~ ^[A-Za-z0-9._:+-]+$ ]] || continue
  case "${key}" in
    NPU_MODEL_NAME | NPU_PART_PREFIX | NPU_PACKAGE_SHA | NPU_PACKAGE_BYTES)
      printf -v "${key}" '%s' "${value}" ;;
  esac
done <<<"${pin_lines}"
if [[ -z "${NPU_MODEL_NAME}" || -z "${NPU_PART_PREFIX}" ||
  ! "${NPU_PACKAGE_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: the installed Vaelor package names no complete NPU model package." >&2
  exit 1
fi

REPO="${VAELOR_REPO:-ShadowLayer90/vaelor}"
TAG="${VAELOR_RELEASE_TAG:-v1.5.4}"
MODELS_DIR="${FLM_MODELS_DIR:-/var/lib/vaelor/flm/models}"
MODEL_NAME="${NPU_MODEL_NAME}"
PART_PREFIX="${NPU_PART_PREFIX}"
PART_GLOB="${PART_PREFIX}*"
EXPECT_SHA="${NPU_PACKAGE_SHA}"

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

# Give one installed model entry the modes the console needs to read it: every
# directory 0755 and every file 0644. A symlink is skipped, and find does not
# follow one below it, so nothing outside the entry changes. The installer's
# normalise_npu_model_modes applies the same two modes to a model that is
# already installed; tests/test_npu_model_modes.py holds the two together.
normalise_model_modes() {
  local entry="$1"
  [[ -L "${entry}" ]] && return 0
  if [[ -d "${entry}" ]]; then
    run_privileged find "${entry}" -type d -exec chmod 0755 {} +
    run_privileged find "${entry}" -type f -exec chmod 0644 {} +
  elif [[ -f "${entry}" ]]; then
    run_privileged chmod 0644 "${entry}"
  fi
}

# The size printed is the pinned package's own (LESSONS 5: a typed "~3.35 GB" outlived its package).
if [[ "${NPU_PACKAGE_BYTES}" =~ ^[0-9]+$ ]]; then
  npu_size="$(awk -v b="${NPU_PACKAGE_BYTES}" 'BEGIN { printf "%.2f GiB", b / 1073741824 }')"
else
  npu_size="size not stated by the pin"
fi
echo "Fetching ${MODEL_NAME} (${npu_size}) from ${REPO} release ${TAG} ..."
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
  REPO="$REPO" TAG="$TAG" DEST="$tmp" PART_PREFIX="$PART_PREFIX" python3 - <<'PY'
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
    if not name.startswith(os.environ["PART_PREFIX"]):
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
  REPO="$REPO" TAG="$TAG" DEST="$tmp" PART_PREFIX="$PART_PREFIX" python3 - <<'PY'
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
    if not name.startswith(os.environ["PART_PREFIX"]):
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
parts=("${tmp}/${PART_PREFIX}"*)
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
# The archive's own modes are not the install's: the v1.5 release tar stores the
# model folder 0770, and `tar -xf` as root keeps archived modes (and owners), so
# the folder landed drwxrwx--- root:root and the console - user vaelor - could
# not measure it ("Size unknown"). Extract without the archive's owners and
# permissions, then set every entry the archive placed (read from tar's own
# verbose list, so nothing else under ${MODELS_DIR} is touched) to 0755 for a
# directory and 0644 for a file, whatever the umask was.
# The list goes to a file, not a process substitution, so a failed extraction
# still stops this script under `set -e`.
run_privileged tar --no-same-owner --no-same-permissions \
  -C "$MODELS_DIR" -xvf "${tmp}/model.tar" >"${tmp}/extracted.list"
mapfile -t extracted < <(sed -e 's#^\./##' -e 's#/.*##' "${tmp}/extracted.list" | sort -u)
for entry in "${extracted[@]}"; do
  [[ -n "${entry}" && "${entry}" != "." && "${entry}" != ".." ]] || continue
  normalise_model_modes "${MODELS_DIR}/${entry}"
done

echo "Done. NPU model installed at ${MODELS_DIR}/${MODEL_NAME}."
