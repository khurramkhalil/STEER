"""Pytest configuration: make the repo root importable for `import steer`, etc."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
