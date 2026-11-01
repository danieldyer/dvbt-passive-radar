#!/usr/bin/env python3
"""Compatibility entry point: real terminal validation with Ctrl-C shutdown."""

import sys

from validate_tui import main

if __name__ == "__main__":
    sys.argv.append("--ctrl-c")
    main()
