#!/usr/bin/env python3
"""Compatibility entrypoint for the REINVENT single-target wrapper."""

from integrations.reinvent4.external_process.boltz_single_wrapper import *  # noqa: F401,F403
from integrations.reinvent4.external_process.boltz_single_wrapper import main


if __name__ == "__main__":
    raise SystemExit(main())
