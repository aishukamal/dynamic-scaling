# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the libtpu control-pipe client and PlatformTPU.suspend_device / resume_device.

No libtpu: the control thread is faked by a thread servicing a pipe pair with the same framing.
"""

import os
import struct
import threading
from unittest import mock

import pytest

from verl_hardware_plugin.accelerators.tpu import platform_tpu, tpu_control

# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def test_request_framing_matches_the_protocol():
    # length 5, field 1 (action) = CHECKPOINT, field 2 (timeout) = 300 as a two-byte varint.
    assert tpu_control.encode_request(tpu_control.ACTION_CHECKPOINT, 300).hex() == "00000005" + "0802" + "10ac02"
    assert tpu_control.encode_request(tpu_control.ACTION_RESTORE, 600).hex() == "00000005" + "0803" + "10d804"


def test_response_decoding_and_unknown_fields():
    frame = tpu_control.encode_response(True, state=3, error="")
    (length,) = struct.unpack(">I", frame[:4])
    assert tpu_control.decode_response(frame[4 : 4 + length]) == {
        "success": True,
        "state": 3,
        "error": "",
        "state_name": "DETACHED",
    }
    frame = tpu_control.encode_response(False, state=5, error="boom")
    resp = tpu_control.decode_response(frame[4:])
    assert (resp["success"], resp["state_name"], resp["error"]) == (False, "FAULTED", "boom")
    # Unknown fixed32 (field 9) and fixed64 (field 10) fields are skipped.
    payload = b"\x08\x01" + b"\x4d" + b"\x00\x00\x00\x00" + b"\x51" + b"\x00" * 8 + b"\x10\x01"
    assert tpu_control.decode_response(payload)["state_name"] == "RUNNING"


# ---------------------------------------------------------------------------
# Fake control thread on a pipe pair
# ---------------------------------------------------------------------------


class _FakeControlThread:
    """Services the request pipe like libtpu's control thread; ``fail`` lists actions to reject."""

    def __init__(self, fail=()):
        self.req_r, self.req_w = os.pipe()
        self.rsp_r, self.rsp_w = os.pipe()
        self.actions = []
        self.fail = set(fail)
        self.state = 1
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        self.control = tpu_control.LibtpuSelfControl(self.req_w, self.rsp_r)

    def _serve(self):
        while True:
            header = os.read(self.req_r, 4)
            if len(header) < 4:
                return
            (length,) = struct.unpack(">I", header)
            body = os.read(self.req_r, length)
            action, pos = tpu_control._decode_varint(body, 1)
            self.actions.append(action)
            ok = action not in self.fail
            if ok and action == tpu_control.ACTION_CHECKPOINT:
                self.state = 3
            elif ok and action == tpu_control.ACTION_RESTORE:
                self.state = 1
            os.write(self.rsp_w, tpu_control.encode_response(ok, state=self.state, error="" if ok else "nope"))

    def close(self):
        for fd in (self.req_w, self.req_r, self.rsp_w, self.rsp_r):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.fixture
def fake_control(monkeypatch):
    def _install(fail=()):
        fake = _FakeControlThread(fail=fail)
        monkeypatch.setattr(tpu_control.LibtpuSelfControl, "_instance", fake.control)
        return fake

    fakes = []

    def install(fail=()):
        fakes.append(_install(fail))
        return fakes[-1]

    yield install
    for fake in fakes:
        fake.close()
    tpu_control.LibtpuSelfControl._instance = None


def test_suspend_then_resume_round_trip(fake_control):
    fake = fake_control()
    info = tpu_control.suspend(timeout_s=7)
    assert info["success"] and info["state_name"] == "DETACHED" and "suspend_s" in info
    assert tpu_control.is_suspended()
    info = tpu_control.resume(timeout_s=9)
    assert info["success"] and info["state_name"] == "RUNNING" and "resume_s" in info
    assert not tpu_control.is_suspended()
    A = tpu_control
    assert fake.actions == [A.ACTION_LOCK, A.ACTION_CHECKPOINT, A.ACTION_RESTORE, A.ACTION_UNLOCK]
    assert tpu_control.runtime_state()["state_name"] == "RUNNING"


def test_suspend_and_resume_are_idempotent(fake_control):
    fake = fake_control()
    assert tpu_control.resume() == {"skipped": True, "reason": "not suspended"}
    tpu_control.suspend()
    assert tpu_control.suspend() == {"skipped": True, "reason": "already suspended"}
    tpu_control.resume()
    assert tpu_control.resume() == {"skipped": True, "reason": "not suspended"}
    A = tpu_control
    assert fake.actions == [A.ACTION_LOCK, A.ACTION_CHECKPOINT, A.ACTION_RESTORE, A.ACTION_UNLOCK]


def test_lock_failure_is_tolerated_and_unlock_skipped(fake_control):
    fake = fake_control(fail={tpu_control.ACTION_LOCK})
    tpu_control.suspend()
    tpu_control.resume()
    A = tpu_control
    assert fake.actions == [A.ACTION_LOCK, A.ACTION_CHECKPOINT, A.ACTION_RESTORE]


def test_failed_restore_is_never_retried(fake_control):
    fake = fake_control(fail={tpu_control.ACTION_RESTORE})
    tpu_control.suspend()
    with pytest.raises(RuntimeError, match="Not retried"):
        tpu_control.resume()
    with pytest.raises(RuntimeError, match="do not retry"):
        tpu_control.resume()
    assert fake.actions.count(tpu_control.ACTION_RESTORE) == 1


def test_failed_checkpoint_raises_and_keeps_the_process_resident(fake_control):
    fake_control(fail={tpu_control.ACTION_CHECKPOINT})
    with pytest.raises(RuntimeError, match="suspend failed"):
        tpu_control.suspend()
    assert not tpu_control.is_suspended()


# ---------------------------------------------------------------------------
# channel discovery on a temp dir
# ---------------------------------------------------------------------------


def test_find_self_channel_parses_the_thread_name(tmp_path):
    for tid, comm in (("11", "python"), ("12", "libtpu00070009"), ("13", "grpc")):
        (tmp_path / tid).mkdir()
        (tmp_path / tid / "comm").write_text(comm + "\n")
    assert tpu_control.find_self_channel(task_dir=str(tmp_path)) == (7, 9)
    assert tpu_control.find_self_channel(task_dir=str(tmp_path / "none")) is None


# ---------------------------------------------------------------------------
# PlatformTPU.suspend_device / resume_device
# ---------------------------------------------------------------------------


def test_platform_functions_use_the_plugin_client_and_are_idempotent(fake_control):
    fake = fake_control()
    platform = platform_tpu.PlatformTPU()
    assert "suspend_s" in platform.suspend_device()
    assert platform.suspend_device() == {"skipped": True, "reason": "already suspended"}
    assert "resume_s" in platform.resume_device()
    assert platform_tpu.resume_device() == {"skipped": True, "reason": "not suspended"}
    A = tpu_control
    assert fake.actions == [A.ACTION_LOCK, A.ACTION_CHECKPOINT, A.ACTION_RESTORE, A.ACTION_UNLOCK]


def test_platform_functions_prefer_torch_tpu_when_it_has_suspend():
    fake_tpu = mock.Mock()
    fake_tpu.suspend.return_value = {"suspend_s": 1.0}
    fake_tpu.resume.return_value = {"resume_s": 2.0}
    with mock.patch.object(platform_tpu.torch, "tpu", fake_tpu, create=True):
        assert platform_tpu.suspend_device() == {"suspend_s": 1.0}
        assert platform_tpu.resume_device(timeout_s=5) == {"resume_s": 2.0}
    fake_tpu.suspend.assert_called_once_with()
    fake_tpu.resume.assert_called_once_with(5)
