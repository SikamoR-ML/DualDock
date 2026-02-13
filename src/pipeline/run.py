import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return ""

def main() -> None:
    p = argparse.ArgumentParser("DualDock pipeline entrypoint")
    p.add_argument("--config", help="Path to YAML config (optional for now)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--stage", default="analyze", choices=["generate", "dock", "analyze"])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "config": args.config,
        "cwd": str(Path.cwd()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "git_commit": sh("git rev-parse HEAD"),
        "git_status": sh("git status --porcelain"),
    }

    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "README_RUN.txt").write_text(
        "This run folder was created by src.pipeline.run\n",
        encoding="utf-8",
    )

    print(f"[OK] Wrote: {out/'metadata.json'}")

if __name__ == "__main__":
    main()
