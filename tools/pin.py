"""Run a command pinned to this program's CPU set (Windows).

Usage: python tools/pin.py <command> [args...]
Child processes inherit the affinity, so everything the command spawns stays pinned.
Thread-pool env vars default to the CPU count; set them lower per job if needed.
"""
import ctypes
import os
import shutil
import subprocess
import sys

CPUS = [4, 5, 6, 7, 8]  # logical CPUs allotted to this program; 0-3 are reserved for the user


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mask = sum(1 << c for c in CPUS)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
        print(f"SetProcessAffinityMask failed: {ctypes.get_last_error()}", file=sys.stderr)
        return 1
    n = str(len(CPUS))
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS", "RAYON_NUM_THREADS"):
        env.setdefault(k, n)
    cmd = sys.argv[1:]
    # CreateProcess doesn't apply PATHEXT to relative paths like .venv/Scripts/python
    cmd[0] = shutil.which(cmd[0]) or cmd[0]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
