"""User-pricing pipeline package.

Adds the project root to sys.path so modules can `import config`.
"""
import os
import sys

__version__ = "0.1.0"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
