"""Bounded local text preparation for the small educational models."""
import hashlib
from pathlib import Path
import unicodedata

MAX_BYTES = 4_000_000
MAX_CHARACTERS = 20_000


def read_text(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Input exceeds 4 MB")
    text = raw.decode("utf-8-sig")
    return text, hashlib.sha256(raw).hexdigest()


def clean_paragraphs(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if any(unicodedata.category(c).startswith("C") and c not in "\n\t" for c in text):
        raise ValueError("Control/format characters are not supported")
    paragraphs, current, seen = [], [], set()
    for line in [*text.splitlines(), ""]:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraph = "\n".join(current)
            if paragraph not in seen:
                paragraphs.append(paragraph)
                seen.add(paragraph)
            current = []
    if len(paragraphs) < 2:
        raise ValueError("Need at least two distinct paragraphs for train/validation")
    clean = "\n\n".join(paragraphs)
    if len(clean) > MAX_CHARACTERS:
        raise ValueError("Educational corpus exceeds 20000 characters; select a smaller sample")
    return paragraphs


def split_dataset(paragraphs):
    training, validation = "\n\n".join(paragraphs[:-1]), paragraphs[-1]
    alphabet = "".join(sorted(set(training)))
    if not 2 <= len(alphabet) <= 256 or len(training) < 4 or len(validation) < 4:
        raise ValueError("Need sufficient training/validation text and 2–256 training characters")
    if set(validation) - set(alphabet):
        raise ValueError("Validation has out-of-vocabulary characters; revise training data explicitly")
    return training, validation, alphabet


def read_dataset(path):
    text, digest = read_text(path)
    return (*split_dataset(clean_paragraphs(text)), digest)
