"""
DEPRECATED: Use 'python -m meridian.cli' instead.
This module will be removed in v1.0.0.
"""
import warnings

warnings.warn(
    "The 'core.cli' module is deprecated. Use 'python -m meridian.cli' instead.",
    DeprecationWarning,
    stacklevel=2
)

from meridian.cli import main

# Alias for compatibility
cli = main

if __name__ == "__main__":
    main()
