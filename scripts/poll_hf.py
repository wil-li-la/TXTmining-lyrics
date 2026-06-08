"""Poll the HF Space build/runtime stage until it settles. Prints one line per change."""
import os
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

REPO = "wil-li-la/pop-lyrics-taste-profiler"
TERMINAL_OK = {"RUNNING"}
TERMINAL_BAD = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "DELETING", "PAUSED"}


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    last = None
    seen_building = False
    deadline = 600  # seconds
    waited = 0
    while waited < deadline:
        stage = api.get_space_runtime(REPO).stage
        if stage != last:
            print(f"stage: {stage}", flush=True)
            last = stage
        # Compound stages like RUNNING_BUILDING / RUNNING_APP_STARTING mean a
        # rebuild is mid-flight while the old app still serves.
        if "BUILD" in stage or "STARTING" in stage:
            seen_building = True
        if stage in TERMINAL_BAD:
            print(f"FAILED: {stage}", flush=True)
            sys.exit(2)
        if stage == "RUNNING" and seen_building:
            print("DONE: RUNNING", flush=True)
            return
        time.sleep(10)
        waited += 10
    print("TIMEOUT waiting for rebuild", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
