import os
import re
from pathlib import Path

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")

def read_txt_byteslike(file) -> str:
    return file.read().decode("utf-8", errors="ignore")
