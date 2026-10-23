#!/bin/sh
set -eu
umask 077

mkdir -p \
  /var/lib/vaelor/credentials \
  /var/lib/vaelor/logs \
  /run/vaelor

if [ "${1:-}" = "vaelor-credential-broker" ] \
  && [ ! -s "${VAELOR_CREDENTIAL_MASTER_KEY_FILE}" ]; then
  python -c 'import os; path=os.environ["VAELOR_CREDENTIAL_MASTER_KEY_FILE"]; stream=open(path, "xb"); stream.write(os.urandom(32)); stream.close(); os.chmod(path, 0o600)'
fi

exec "$@"
