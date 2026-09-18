"""Entry point for the Avis rental support agent TUI."""
import sys
import os

# Ensure src/ is on the path so modules can import each other
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terminal import main  # noqa: E402

if __name__ == "__main__":
    main()
