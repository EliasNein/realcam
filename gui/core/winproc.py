"""Windows process control with ctypes: job object (kills the whole process tree), keep-awake, process lookup. No Qt, no torch."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Mapping, Sequence

CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _k32.SetThreadExecutionState.restype = wintypes.DWORD
    _k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]


class JobObject:
    """A Windows job object that kills every process in it when terminated or when the last handle closes (GUI crash)."""

    def __init__(self) -> None:
        self._handle = _k32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(self._handle, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(err)

    def assign(self, proc: subprocess.Popen) -> None:
        if not _k32.AssignProcessToJobObject(self._handle, wintypes.HANDLE(int(proc._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self, exit_code: int = 1) -> None:
        if self._handle:
            _k32.TerminateJobObject(self._handle, exit_code)

    def close(self) -> None:
        if self._handle:
            _k32.CloseHandle(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def spawn_in_job(argv: Sequence[str], cwd: Path, log_path: Path, env: Mapping[str, str] | None = None) -> tuple[subprocess.Popen, JobObject]:
    """Start `argv` suspended, put it into a new job object, then let it run: no child can escape the job."""
    job = JobObject()
    log = open(log_path, "ab")
    try:
        proc = subprocess.Popen(list(argv), cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=CREATE_SUSPENDED | CREATE_NO_WINDOW, env=dict(env) if env else None)
    except BaseException:
        job.close()
        raise
    finally:
        log.close()
    try:
        job.assign(proc)
        _ntdll.NtResumeProcess(wintypes.HANDLE(int(proc._handle)))
    except BaseException:
        proc.kill()
        proc.wait()
        job.close()
        raise
    return proc, job


def process_start_time(pid: int) -> int | None:
    """Creation time (FILETIME ticks) of a running process, None if there is no such process."""
    handle = _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != STILL_ACTIVE:
            return None
        created, a, b, c = (wintypes.FILETIME() for _ in range(4))
        if not _k32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(a), ctypes.byref(b), ctypes.byref(c)):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        _k32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    return process_start_time(pid) is not None


def set_keep_awake(on: bool) -> bool:
    """Keep the system awake (the display may still sleep). Must be called from the same thread for on and off."""
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS
    return bool(_k32.SetThreadExecutionState(flags))


def external_enhance_processes(timeout: float = 30.0) -> tuple[list[tuple[int, str]], bool]:
    """Running python processes whose command line mentions enhance.py: ([(pid, command line)], lookup_worked)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'enhance[.]py' } "
          "| ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=timeout,
                             creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return [], False
    if res.returncode != 0:
        return [], False
    found = []
    for line in res.stdout.splitlines():
        pid, _, cmd = line.partition("\t")
        if pid.strip().isdigit():
            found.append((int(pid), cmd.strip()))
    return found, True


def own_pid() -> int:
    return os.getpid()
