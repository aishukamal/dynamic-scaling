# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""In-process libtpu suspend / resume through libtpu's control pipes. Standard library only.

When a TPU process runs with ``LIBTPU_CHECKPOINTING_ENABLED=true``, libtpu starts a control thread
named ``libtpu{XXXXYYYY}`` where the 8 hex digits encode two pipe file descriptors (request-write,
response-read). The thread services length-delimited protobuf control messages; the schema is
``pkg/sentry/control/tpu_control.proto`` in the gVisor tree (https://github.com/google/gvisor).
External tools (gVisor, the tpucheckpoint CLI) reach these pipes through ``/proc/<pid>/fd``. This
module uses the fact that the descriptors belong to *this* process: a worker writes CHECKPOINT to
its own pipe and later RESTORE, with no external service, no PIDs and no elevated privileges.

``suspend()`` parks the process's TPU state (HBM copied to host memory, device descriptors closed)
while the process stays alive on CPU; ``resume()`` re-attaches the devices and hydrates the state.
Compiled programs survive inside the image, so a process compiles once.

Contract: the process issues no TPU operation between ``suspend()`` and ``resume()``. The protocol's
LOCK action is the device-side quiesce; ``suspend()`` attempts it and tolerates runtimes without it,
in which case the caller must be idle on the device before suspending.

This module is the interim home of the client while torch_tpu reviews ``torch.tpu.suspend()`` /
``torch.tpu.resume()`` / ``torch.tpu.runtime_state()``. The API matches so the swap is an import
change in ``platform_tpu.suspend_device`` / ``resume_device``, the only callers.
"""

import logging
import os
import re
import select
import struct
import threading
import time

logger = logging.getLogger(__name__)

# tpu_control.proto: ControlAction
ACTION_LOCK = 1
ACTION_CHECKPOINT = 2
ACTION_RESTORE = 3
ACTION_UNLOCK = 4
ACTION_GETSTATE = 5

# tpu_control.proto: RuntimeState
STATE_NAMES = {
    0: "UNSPECIFIED",
    1: "RUNNING",
    2: "LOCKED",
    3: "DETACHED",
    4: "RESTORING",
    5: "FAULTED",
}

_CONTROL_THREAD_RE = re.compile(r"^libtpu([0-9a-fA-F]{8})$")

# Defaults measured on v5e / v6e slices up to 64 chips; every public function takes them as arguments.
DEFAULT_SUSPEND_TIMEOUT_S = 300
DEFAULT_RESUME_TIMEOUT_S = 600
PROC_SELF_TASK = "/proc/self/task"


# ---------------------------------------------------------------------------
# Protobuf wire codec (request: action=1 varint, timeout_secs=2 varint;
# response: success=1 varint, current_state=2 varint, error_message=3 bytes).
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def encode_request(action: int, timeout_secs: int) -> bytes:
    """One length-framed ControlRequest: 4-byte big-endian length, then the protobuf body."""
    body = b"\x08" + _encode_varint(action)  # field 1, varint
    body += b"\x10" + _encode_varint(timeout_secs)  # field 2, varint
    return struct.pack(">I", len(body)) + body


def encode_response(success: bool, state: int = 0, error: str = "") -> bytes:
    """One length-framed ControlResponse (used by tests and tools that fake the control thread)."""
    body = b"\x08" + _encode_varint(1 if success else 0)
    body += b"\x10" + _encode_varint(state)
    if error:
        data = error.encode("utf-8")
        body += b"\x1a" + _encode_varint(len(data)) + data
    return struct.pack(">I", len(body)) + body


def decode_response(payload: bytes) -> dict:
    """Decode a ControlResponse body; unknown fields are skipped."""
    resp = {"success": False, "state": 0, "error": ""}
    pos = 0
    while pos < len(payload):
        tag, pos = _decode_varint(payload, pos)
        field, wire = tag >> 3, tag & 0x07
        if wire == 0:  # varint
            value, pos = _decode_varint(payload, pos)
            if field == 1:
                resp["success"] = bool(value)
            elif field == 2:
                resp["state"] = value
        elif wire == 2:  # length-delimited
            length, pos = _decode_varint(payload, pos)
            data = payload[pos : pos + length]
            pos += length
            if field == 3:
                resp["error"] = data.decode("utf-8", errors="replace")
        elif wire == 5:  # fixed32
            pos += 4
        elif wire == 1:  # fixed64
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
    resp["state_name"] = STATE_NAMES.get(resp["state"], str(resp["state"]))
    return resp


# ---------------------------------------------------------------------------
# Channel to this process's own libtpu control thread.
# ---------------------------------------------------------------------------


def find_self_channel(task_dir: str = PROC_SELF_TASK) -> tuple[int, int] | None:
    """Locate this process's libtpu control pipes.

    Returns ``(request_write_fd, response_read_fd)``, or ``None`` when libtpu checkpointing is not
    active in this process.
    """
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return None
    for tid in tids:
        try:
            with open(os.path.join(task_dir, tid, "comm")) as f:
                comm = f.read().strip()
        except OSError:
            continue
        m = _CONTROL_THREAD_RE.match(comm)
        if m:
            digits = m.group(1)
            return int(digits[:4], 16), int(digits[4:], 16)
    return None


def _rss_mib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def _sync_device() -> None:
    """Drain pending device work from this process before suspending (best effort)."""
    try:
        import torch

        if hasattr(torch, "tpu") and torch.tpu.is_available():
            torch.tpu.synchronize()
    except Exception as e:  # noqa: BLE001 - JAX processes have no torch.tpu
        logger.debug("device synchronize before suspend skipped: %s", e)


class LibtpuSelfControl:
    """Serialized request/response exchange on this process's control pipes, plus the suspend state."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, req_fd: int, rsp_fd: int):
        self._req_fd = req_fd
        self._rsp_fd = rsp_fd
        self._lock = threading.Lock()
        self._device_locked = False
        self.suspended = False
        self.faulted = False

    @classmethod
    def get(cls) -> "LibtpuSelfControl":
        with cls._instance_lock:
            if cls._instance is None:
                channel = find_self_channel()
                if channel is None:
                    raise RuntimeError(
                        "libtpu control thread not found in this process; is "
                        "LIBTPU_CHECKPOINTING_ENABLED=true set and the TPU initialized?"
                    )
                cls._instance = cls(*channel)
            return cls._instance

    def request(self, action: int, timeout_secs: int) -> dict:
        deadline = time.monotonic() + timeout_secs + 30  # slack over libtpu's own timeout
        with self._lock:
            frame = encode_request(action, timeout_secs)
            written = 0
            while written < len(frame):
                written += os.write(self._req_fd, frame[written:])
            header = self._read_exact(4, deadline)
            (length,) = struct.unpack(">I", header)
            if length > 1 << 20:
                raise RuntimeError(f"implausible control response length {length}")
            return decode_response(self._read_exact(length, deadline))

    def _read_exact(self, n: int, deadline: float) -> bytes:
        buf = b""
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out reading libtpu control response ({len(buf)}/{n} bytes)")
            ready, _, _ = select.select([self._rsp_fd], [], [], min(remaining, 1.0))
            if not ready:
                continue
            chunk = os.read(self._rsp_fd, n - len(buf))
            if not chunk:
                raise RuntimeError("libtpu control response pipe closed")
            buf += chunk
        return buf


# ---------------------------------------------------------------------------
# Public API: suspend / resume / runtime_state.
# ---------------------------------------------------------------------------


def is_suspended() -> bool:
    """True between a successful ``suspend()`` and the next successful ``resume()`` in this process."""
    ctl = LibtpuSelfControl._instance
    return bool(ctl is not None and ctl.suspended)


def suspend(timeout_s: int = DEFAULT_SUSPEND_TIMEOUT_S) -> dict:
    """Park this process's TPU state (HBM to host memory, devices released). Idempotent.

    Returns the control response plus ``suspend_s`` and ``rss_delta_mib`` (host memory the image took);
    ``{"skipped": True}`` when the process is already suspended.
    """
    ctl = LibtpuSelfControl.get()
    if ctl.suspended:
        return {"skipped": True, "reason": "already suspended"}
    _sync_device()

    lock_resp = None
    try:
        lock_resp = ctl.request(ACTION_LOCK, min(timeout_s, 60))
        ctl._device_locked = bool(lock_resp.get("success"))
    except Exception as e:  # noqa: BLE001
        logger.debug("LOCK attempt failed (%s); proceeding, caller must be idle on the device", e)
    if lock_resp is not None and not lock_resp.get("success"):
        logger.debug("LOCK not supported/failed: %s; caller must be idle on the device", lock_resp.get("error", ""))

    rss0 = _rss_mib()
    t0 = time.perf_counter()
    resp = ctl.request(ACTION_CHECKPOINT, timeout_s)
    if not resp["success"]:
        raise RuntimeError(f"TPU suspend failed in state {resp['state_name']}: {resp['error']}")
    resp["suspend_s"] = time.perf_counter() - t0
    resp["rss_delta_mib"] = _rss_mib() - rss0
    ctl.suspended = True
    logger.info("TPU suspend done in %.2fs (state %s)", resp["suspend_s"], resp["state_name"])
    return resp


def resume(timeout_s: int = DEFAULT_RESUME_TIMEOUT_S) -> dict:
    """Re-attach the TPU devices and hydrate this process's parked state. Idempotent.

    Resume is a slice-wide rendezvous inside libtpu: every mesh member's RESTORE must be pending at the
    same time (callers fan out concurrently). A failed RESTORE is never re-sent: the runtime is marked
    faulted and every later call raises at once, because a re-sent RESTORE wedges the runtime.
    """
    ctl = LibtpuSelfControl.get()
    if ctl.faulted:
        raise RuntimeError("TPU resume refused: an earlier RESTORE failed in this process; do not retry")
    if not ctl.suspended:
        return {"skipped": True, "reason": "not suspended"}

    t0 = time.perf_counter()
    resp = ctl.request(ACTION_RESTORE, timeout_s)
    if not resp["success"]:
        ctl.faulted = True
        raise RuntimeError(
            f"TPU resume failed in state {resp['state_name']}: {resp['error']}. "
            "Not retried: a pending RESTORE that is re-sent wedges the runtime."
        )
    resp["resume_s"] = time.perf_counter() - t0
    ctl.suspended = False
    logger.info("TPU resume done in %.2fs (state %s)", resp["resume_s"], resp["state_name"])

    if ctl._device_locked:
        try:
            ctl.request(ACTION_UNLOCK, 60)
        except Exception as e:  # noqa: BLE001
            logger.debug("UNLOCK attempt failed (%s)", e)
        ctl._device_locked = False
    return resp


def runtime_state(timeout_s: int = 30) -> dict:
    """Query the runtime state; tolerant of runtimes without GETSTATE."""
    try:
        return LibtpuSelfControl.get().request(ACTION_GETSTATE, timeout_s)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "state": 0, "state_name": "UNSPECIFIED", "error": str(e)}
