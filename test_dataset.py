import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dataset import clean_paragraphs, read_dataset
from prepare_corpus import prepare


class DatasetTests(unittest.TestCase):
    def test_normalizes_unicode_newlines_and_deduplicates_before_split(self):
        self.assertEqual(clean_paragraphs(" cafe\u0301  \r\n\r\ncafé\r\n\r\n  café café "), ["café", "café café"])

    def test_provenance_and_training_vocabulary_roundtrip_without_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/"input.txt", Path(folder)/"prepared"
            raw = b"aabb abba\r\n\r\naabb abba\r\n\r\nabba aabb\r\n"
            source.write_bytes(raw)
            manifest = prepare(source, output)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(manifest["source_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest["paragraphs"], 2)
            train, validation, alphabet, _ = read_dataset(output/"corpus.txt")
            self.assertEqual(train, "aabb abba")
            self.assertEqual(validation, "abba aabb")
            self.assertEqual(alphabet, " ab")
            self.assertEqual(json.loads((output/"manifest.json").read_text()), manifest)
            with self.assertRaises(FileExistsError): prepare(source, output)

    def test_rejects_bad_data_before_creating_output(self):
        for text in ["one paragraph", "abcd\n\nzzzz", "aabb\x00\n\naabb", "aabb\n\n"+"a"*20001]:
            with self.subTest(text=text[:20]), tempfile.TemporaryDirectory() as folder:
                source, output = Path(folder)/"input.txt", Path(folder)/"prepared"
                source.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError): prepare(source, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
