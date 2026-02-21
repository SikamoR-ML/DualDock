#!/usr/bin/env python3
"""Compatibility entrypoint for the REINVENT dual-target wrapper."""

from integrations.reinvent4.external_process.boltz_dual_wrapper import *  # noqa: F401,F403
from integrations.reinvent4.external_process.boltz_dual_wrapper import main


if __name__ == "__main__":
    raise SystemExit(main())
