"""
Explainable Machine Learning for CKD Classification Pipeline Package
"""

import os
import sys

# Ensure local dependencies in deps/ are discoverable
deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)
