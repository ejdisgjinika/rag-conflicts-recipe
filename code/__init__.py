"""Experiment modules. NLTK assets are installed explicitly by scripts/setup_nltk.py."""
from pathlib import Path
import nltk

nltk.data.path.insert(0, str(Path(__file__).resolve().parent.parent / ".cache" / "nltk_data"))
