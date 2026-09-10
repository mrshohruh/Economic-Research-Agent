#!/usr/bin/env python
"""Entry point: python run.py --data <file>"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uzhousing.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
