"""SHA-256 of the README "## Pre-registration" section (up to "## Deviations"). `log` appends it to run.log."""
import hashlib
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
txt = (HERE / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
sec = txt[txt.index("## Pre-registration\n"):txt.index("## Deviations\n")]
h = hashlib.sha256(sec.encode("utf-8")).hexdigest()
print(h, len(sec), "chars")
if sys.argv[1:] == ["log"]:
    with open(HERE / "run.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pre-registration section sha256 {h}\n")
