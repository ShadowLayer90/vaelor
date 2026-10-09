"""The weights-pull program shipped to nodes, and its progress-file contract.

The payload half of the model-library pull: `gpu_pool_runtime` emits the
``docker run`` and systemd unit that RUN this program, and this module is the
program. Two responsibilities, so two modules - one is the node command
surface, the other a self-contained Python file executed by the image's own
interpreter, whose text is data to everything here.

**The progress-file contract.** The program MUST periodically write, atomically
(temp file in the same directory + ``os.replace``), a JSON object to the
progress path::

    {"phase": "starting"|"fetching"|"verifying"|"done"|"error",
     "percent": <int 0-100>, "downloaded_bytes": <int>,
     "total_bytes": <int>, "message": <str>}

It exits 0 only after writing phase ``done`` (percent 100), and exits non-zero
after writing phase ``error`` with a human message - including when it is
*stopped*: a ``SIGTERM`` (what ``docker stop`` and ``systemctl stop`` send) and
a keyboard interrupt both write the ``error`` record before exiting, so a
stopped pull leaves the documented terminal state rather than a stale
``fetching`` row a poll would follow to its own deadline. Every terminal record
is written only after the progress ticker has been stopped AND joined (under a
bounded wait, so a wedged tick cannot make a stop unresponsive), because one
last tick landing after it would overwrite the terminal record with
``fetching``. ``percent`` appears ONLY
when the total is known - the hub API supplies that denominator, and when it
cannot the total is written as 0 with no percent rather than a fabricated
fraction. Both figures are BYTES, summed from the files actually on disk under
the repo's blob directory (partial ``*.incomplete`` blobs included), never a
count of files: a ``tqdm_class`` hook receives the per-FILE bar and the
per-file byte bars are internal, which is how a 16 GB model once reported nine
bytes cached.
"""

from __future__ import annotations

import hashlib

#: The self-contained pull program, run by the image's own Python inside a
#: container. Shipped as a constant rather than baked into an artifact, because
#: there is no install tree any more.
PULL_SCRIPT = '''"""Pull one Hugging Face repo into the mounted cache, reporting progress."""
import argparse
import glob
import json
import os
import signal
import sys
import tempfile
import threading

#: What the terminal record says when the pull was stopped from outside rather
#: than failing on its own. One spelling, so the SIGTERM and the keyboard
#: interrupt cannot describe the same event two ways.
STOP_MESSAGE = "The pull was stopped."


def write_progress(path, payload):
    """Write the progress object atomically, so a poll never reads half a file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(payload, stream)
    os.replace(temporary, path)


def blob_directory(cache_dir, repo):
    """Where a repo's file contents land: <cache>/hub/models--org--name/blobs."""
    return os.path.join(
        cache_dir, "hub", "models--" + repo.replace("/", "--"), "blobs"
    )


def bytes_on_disk(directory):
    """The BYTES written under a blob directory, in-flight partials included.

    A downloading file is a ``*.incomplete`` blob that is renamed onto its digest
    when it completes, so counting both spellings makes the sum continuous across
    the rename. This is measured, not inferred from a file counter.
    """
    total = 0
    for entry in glob.glob(os.path.join(directory, "*")):
        try:
            total += os.path.getsize(entry)
        except OSError:
            continue
    return total


def total_from_api(repo, revision):
    """The repo's summed file sizes from the hub API, or 0 when unknown.

    ``files_metadata=True`` is what populates ``siblings[*].size``; a sibling
    without one is skipped. 0 means UNKNOWN - the caller then writes no percent
    rather than a fraction of a denominator nobody has.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    except Exception:
        return 0
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        size = getattr(sibling, "size", None)
        if size:
            total += int(size)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--revision", default=None)
    options = parser.parse_args()
    blobs = blob_directory(options.cache_dir, options.model)

    def report(phase, message, downloaded, total):
        payload = {
            "phase": phase,
            "downloaded_bytes": int(downloaded),
            "total_bytes": int(total),
            "message": message,
        }
        if total > 0:
            payload["percent"] = min(100, int(downloaded) * 100 // int(total))
        write_progress(options.progress_file, payload)

    report("starting", "Starting the pull of " + options.model, 0, 0)
    total = 0
    finished = threading.Event()

    def tick():
        while not finished.wait(1.0):
            report(
                "fetching", "Fetching " + options.model,
                bytes_on_disk(blobs), total,
            )

    ticker = threading.Thread(target=tick)
    ticker.daemon = True
    ticker.start()

    def settle():
        """Stop the ticker and WAIT for it, so the next write is the last one.

        The ticker writes a "fetching" row every second. Setting the event only
        asks it to stop; a tick already inside ``report`` still lands, and it
        would overwrite the terminal record - leaving a finished pull reading as
        still fetching for as long as anything polls it.

        The wait is BOUNDED: a tick blocked on a wedged filesystem must not make
        SIGTERM unresponsive and turn a stop into a kill. The terminal record is
        written after the timeout either way - an unbounded join is the worse
        failure, and a stuck ticker cannot outlive the process.
        """
        finished.set()
        ticker.join(timeout=5.0)

    def stopped(_signum, _frame):
        """Record the terminal error when we are told to stop, then exit 1.

        ``docker stop`` and ``systemctl stop`` send SIGTERM, whose default
        action would kill the program silently and leave the last "fetching"
        row behind as the file's final word.
        """
        settle()
        report("error", STOP_MESSAGE, bytes_on_disk(blobs), total)
        sys.exit(1)

    signal.signal(signal.SIGTERM, stopped)
    # The hub call is made only once the handler above is armed. It is a
    # network round trip that can hang for as long as the hub is slow, and a
    # stop landing inside it must still leave the documented terminal record
    # rather than killing the program on SIGTERM's default action. The ticker
    # reads `total` every second, so it picks the denominator up here.
    total = total_from_api(options.model, options.revision)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=options.model,
            revision=options.revision,
            cache_dir=os.path.join(options.cache_dir, "hub"),
        )
    except KeyboardInterrupt:
        settle()
        report("error", STOP_MESSAGE, bytes_on_disk(blobs), total)
        return 1
    except Exception as error:
        settle()
        report(
            "error", str(error)[:500] or "The model pull failed.",
            bytes_on_disk(blobs), total,
        )
        return 1
    settle()
    cached = bytes_on_disk(blobs)
    write_progress(options.progress_file, {
        "phase": "done",
        "percent": 100,
        "downloaded_bytes": cached,
        "total_bytes": cached,
        "message": "Cached " + options.model,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

#: The digest of the program above, and the file name that carries it. The name
#: is content-hashed so shipping a new version is safe while a pull is in
#: flight: a changed program is a different path, so it can never truncate the
#: file a running container is executing.
PULL_SCRIPT_DIGEST = hashlib.sha256(PULL_SCRIPT.encode("utf-8")).hexdigest()
PULL_SCRIPT_NAME = f"vaelor_pull-{PULL_SCRIPT_DIGEST[:12]}.py"


if __name__ == "__main__":
    # The controller's pull container runs THIS file, mounted read-only out of
    # the installed package by the root bridge (VD-143), so the program a root
    # container executes there is never one another account could have
    # written into the model store. The file carries the program as text, so
    # running the file runs exactly that text, as ``__main__``.
    exec(compile(PULL_SCRIPT, PULL_SCRIPT_NAME, "exec"), {"__name__": "__main__"})
