"""
Root conftest.py — onboarding/packaging helper (NOT a behaviour change).

Its only job is to guarantee the project root is on ``sys.path`` so that
``import rag`` and ``from tests.rag.conftest import ...`` resolve no matter how
the suite is launched (``pytest`` vs ``python -m pytest``, from the project root
or a subdirectory). The package layout is unchanged.

pytest already auto-discovers this file at the project root and treats its
directory as the rootdir; the explicit ``sys.path`` insert below is belt-and-
suspenders for fresh clones.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
