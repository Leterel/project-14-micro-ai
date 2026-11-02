"""Prepare a small local corpus without scraping, uploading or changing inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from dataset import read_text, clean_paragraphs, split_dataset


def prepare(input_path, output):
    text, digest = read_text(input_path)
    paragraphs = clean_paragraphs(text)
    # Validate coverage before creating the new output directory.
    training, validation, alphabet = split_dataset(paragraphs)
    normalized = "\n\n".join(paragraphs) + "\n"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "corpus.txt").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)
    manifest = {"format": "micro-ai-local-corpus", "version": 1,
                "source_sha256": digest, "corpus_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                "normalization": "UTF-8, NFC, LF, trimmed line edges; exact duplicate paragraphs removed",
                "paragraphs": len(paragraphs), "training_characters": len(training),
                "validation_characters": len(validation), "alphabet_size": len(alphabet),
                "split": "last distinct paragraph held out; vocabulary from training only",
                "rights": "Use only local text you are permitted to process/share; no rights inferred."}
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.input, args.output), indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
