"""Download required NLTK assets into this project only."""
from pathlib import Path
import nltk

if __name__ == "__main__":
    destination = Path(__file__).resolve().parent.parent / ".cache" / "nltk_data"
    destination.mkdir(parents=True, exist_ok=True)
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        if not nltk.download(resource, download_dir=str(destination), raise_on_error=True):
            raise RuntimeError(f"Failed to download {resource}")
