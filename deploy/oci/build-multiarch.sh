#!/usr/bin/env sh
set -eu

version="${1:-}"
[ -n "${version}" ] || {
  echo "Usage: build-multiarch.sh VERSION [OUTPUT]" >&2
  exit 2
}
output="${2:-dist/release/vaelor-control-plane-${version}.oci.tar}"

docker buildx build \
  --platform linux/arm64,linux/amd64 \
  --file deploy/oci/Dockerfile \
  --build-arg "VAELOR_VERSION=${version}" \
  --output "type=oci,dest=${output}" \
  .

echo "${output}"
