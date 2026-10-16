"""The 4-bit guesser's host-side file work: one stdlib program, here and on a worker (VD-216 item 2).

Split out of `vllm_guesser_build` (the 1,000-line ceiling): prepare, settle,
check, note, clean and sweep, each on a JSON plan (`vllm_guesser_build.host_plan`).
It runs in-process on the controller (the installer, the root bridge) and as
``python3 -c`` on a worker, which has no Vaelor package - the
`vllm_overlay_prune` arrangement. Every folder is opened from ``/`` with no
link followed and must be root's and writable by nobody else; every file a
single-name regular file of root's (LESSONS 18, #178).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

#: prepare / settle / check / note / clean, each on a JSON plan; stdlib only.
#: Every folder is opened from ``/`` with no link followed and must be root's
#: and writable by nobody else; every file a single-name regular file of root's.
HOST_PROGRAM = r'''
import hashlib, json, os, re, stat, sys, time

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
OTHERS = stat.S_IWGRP | stat.S_IWOTH


class Refused(Exception):
    pass


def sealed(path, owners):
    handle = os.open("/", DIR_FLAGS)
    try:
        status = os.fstat(handle)
        for name in [part for part in path.split("/") if part]:
            child = os.open(name, DIR_FLAGS, dir_fd=handle)
            child_status = os.fstat(child)
            os.close(handle)
            handle = child
            if status.st_uid not in owners or (status.st_mode & OTHERS and not (
                    status.st_mode & stat.S_ISVTX and child_status.st_uid in owners)):
                raise Refused("not-roots")
            status = child_status
        if status.st_uid not in owners or status.st_mode & OTHERS:
            raise Refused("not-roots")
        return handle
    except BaseException:
        os.close(handle)
        raise


def child(parent, name, owners, private=False):
    handle = os.open(name, DIR_FLAGS, dir_fd=parent)
    status = os.fstat(handle)
    if status.st_uid not in owners or status.st_mode & (0o077 if private else OTHERS):
        os.close(handle)
        raise Refused("not-roots")
    return handle


def digest(parent, name, owners, limit, mode=None):
    handle = os.open(name, FILE_FLAGS, dir_fd=parent)
    try:
        status = os.fstat(handle)
        if not stat.S_ISREG(status.st_mode) or status.st_uid not in owners or status.st_nlink != 1:
            raise Refused("not-roots")
        if mode is not None:
            os.fchmod(handle, mode)
        elif status.st_mode & OTHERS:
            raise Refused("not-roots")
        if status.st_size > limit:
            raise Refused("unreadable:larger than any file Vaelor builds")
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(handle, 1 << 22)
            if not chunk:
                break
            hasher.update(chunk)
        return hasher.hexdigest()
    finally:
        os.close(handle)


def folder(parent, name, plan, private=False, mode=None):
    owners = set(plan["owners"])
    handle = child(parent, name, owners, private)
    try:
        names, wanted = sorted(os.listdir(handle)), sorted(plan["outputs"])
        stray = [entry for entry in names if entry not in wanted]
        if stray:
            raise Refused("stray:" + ", ".join(stray))
        missing = [entry for entry in wanted if entry not in names]
        if missing:
            raise Refused("missing:" + ", ".join(missing))
        return {entry: digest(handle, entry, owners, plan["limit"], mode) for entry in wanted}
    finally:
        os.close(handle)


def remove(parent, name):
    status = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if stat.S_ISDIR(status.st_mode):
        handle = os.open(name, DIR_FLAGS, dir_fd=parent)
        try:
            for entry in os.listdir(handle):
                remove(handle, entry)
        finally:
            os.close(handle)
        os.rmdir(name, dir_fd=parent)
    else:
        os.unlink(name, dir_fd=parent)


def write(parent, name, data, mode):
    temporary = ".{}.{}".format(name, os.urandom(6).hex())
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                     | os.O_CLOEXEC, mode, dir_fd=parent)
    try:
        os.fchmod(handle, mode)
        view = memoryview(data)
        while view:
            view = view[os.write(handle, view):]
        os.fsync(handle)
    finally:
        os.close(handle)
    os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)


def keyed(plan, make=False):
    owners = set(plan["owners"])
    if make:
        root = sealed(plan["root"], owners)
        try:
            try:
                os.mkdir(plan["key"], 0o755, dir_fd=root)
            except FileExistsError:
                pass
        finally:
            os.close(root)
    return sealed(plan["root"] + "/" + plan["key"], owners)


def record(handle, plan, reason, found):
    body = {"key": plan["key"], "ready": not reason, "reason": reason, "outputs": found}
    write(handle, "record.json", (json.dumps(body, sort_keys=True) + "\n").encode(), 0o644)


def read_record(handle):
    try:
        file = os.open("record.json", FILE_FLAGS, dir_fd=handle)
    except FileNotFoundError:
        return None
    try:
        status = os.fstat(file)
        if not stat.S_ISREG(status.st_mode) or status.st_mode & OTHERS:
            return None
        value = json.loads(os.read(file, 65536).decode("utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        os.close(file)
    return value if isinstance(value, dict) else None


def check(plan):
    result = {"ready": False, "reason": "", "record": None}
    try:
        handle = keyed(plan)
    except FileNotFoundError:
        result["reason"] = "not-built"
        return result
    except Refused as reason:
        result["reason"] = str(reason)
        return result
    try:
        result["record"] = read_record(handle)
        found = folder(handle, plan["folder"], plan)
        if found != plan["outputs"]:
            result["reason"] = "not-pinned"
        else:
            result["ready"] = True
    except FileNotFoundError:
        result["reason"] = "not-built"
    except Refused as reason:
        result["reason"] = str(reason)
    except OSError as error:
        result["reason"] = "unreadable:" + str(error.strerror or error)
    finally:
        os.close(handle)
    return result


def prepare(plan):
    data = plan["program"].encode("utf-8")
    if hashlib.sha256(data).hexdigest() != plan["program_sha256"]:
        return {"ok": False, "reason": "program-not-pinned"}
    handle = keyed(plan, make=True)
    try:
        try:
            remove(handle, plan["build"])
        except FileNotFoundError:
            pass
        os.mkdir(plan["build"], 0o700, dir_fd=handle)
        build = child(handle, plan["build"], set(plan["owners"]), private=True)
        try:
            write(build, plan["program_name"], data, 0o444)
            os.mkdir("out", 0o700, dir_fd=build)
        finally:
            os.close(build)
    finally:
        os.close(handle)
    return {"ok": True, "reason": ""}


def settle(plan):
    handle = keyed(plan)
    reason, found = "", {}
    try:
        build = child(handle, plan["build"], set(plan["owners"]), private=True)
        try:
            found = folder(build, "out", plan, private=True, mode=0o644)
            if found != plan["outputs"]:
                reason = "not-pinned"
            else:
                out = child(build, "out", set(plan["owners"]), private=True)
                try:
                    os.fchmod(out, 0o755)
                finally:
                    os.close(out)
                try:
                    remove(handle, plan["folder"])
                except FileNotFoundError:
                    pass
                os.rename("out", plan["folder"], src_dir_fd=build, dst_dir_fd=handle)
                os.fsync(handle)
        except Refused as refused:
            reason = str(refused)
        finally:
            os.close(build)
        record(handle, plan, reason, found)
    finally:
        os.close(handle)
    return {"ready": not reason, "reason": reason}


def note(plan):
    handle = keyed(plan, make=True)
    try:
        record(handle, plan, plan["reason"], {})
    finally:
        os.close(handle)
    return {"ok": True}


def clean(plan):
    handle = keyed(plan)
    try:
        remove(handle, plan["build"])
    except FileNotFoundError:
        pass
    finally:
        os.close(handle)
    return {"ok": True}


def sweep(plan):
    handle = keyed(plan)
    removed = []
    try:
        now = time.time()
        for name in sorted(os.listdir(handle)):
            if not re.fullmatch(r"build-[0-9a-f]{12}", name):
                continue
            status = os.stat(name, dir_fd=handle, follow_symlinks=False)
            if now - status.st_mtime > plan["older_than"]:
                remove(handle, name)
                removed.append(name)
    finally:
        os.close(handle)
    return {"ok": True, "removed": removed}


VERBS = {"check": check, "prepare": prepare, "settle": settle, "note": note, "clean": clean,
         "sweep": sweep}

if __name__ == "__main__":
    PLAN = json.loads(sys.argv[1])
    print(json.dumps(VERBS[PLAN["verb"]](PLAN)))
'''


def run_host(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """:data:`HOST_PROGRAM` run here, on this machine."""
    namespace: Dict[str, Any] = {"__name__": "vaelor_guesser_host"}
    exec(compile(HOST_PROGRAM, "vaelor-guesser-host", "exec"), namespace)  # noqa: S102 - our own text
    return namespace["VERBS"][plan["verb"]](dict(plan))


def run_host_on_worker(transport: Any, plan: Mapping[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """:data:`HOST_PROGRAM` run on a worker, as root, over SSH."""
    reply = transport.run(["python3", "-c", HOST_PROGRAM, json.dumps(dict(plan))],
                          sudo=True, timeout=timeout)
    value = json.loads(str(reply))
    if not isinstance(value, dict):
        raise ValueError("the worker's guesser check answered nothing readable")
    return value
