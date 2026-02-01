"""
Tangent: Python bindings for manifold-based nonlinear least squares optimization.

Uses cppyy to allow JIT compilation of tangent's header files.
"""

from .core import init, define_error_term, get_tangent_root, is_initialized
from .optimizer import Optimizer, OptimizationResult
from .templates import error_term_template


class _TangentNamespace:
    """Lazy proxy to cppyy.gbl.Tangent with auto-initialization."""

    def __getattr__(self, name):
        init()
        import cppyy
        return getattr(cppyy.gbl.Tangent, name)


Tangent = _TangentNamespace()

__version__ = "0.1.0"
__all__ = [
    # Core
    "init",
    "define_error_term",
    "get_tangent_root",
    "is_initialized",
    # C++ namespace proxy
    "Tangent",
    # Optimizer
    "Optimizer",
    "OptimizationResult",
    # Templates
    "error_term_template",
]
