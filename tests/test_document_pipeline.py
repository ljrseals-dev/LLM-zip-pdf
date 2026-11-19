"""Regression tests for document ingestion and retrieval."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from src import llm
from src.rag import build_context, chunk_text, simple_retrieve
from src.zip_utils import extract_supported_files, list_zip_contents


class ZipExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "input.zip"

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_paths_and_prevents_duplicate_basename_overwrite(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("one/readme.txt", "first")
            archive.writestr("two/readme.txt", "second")
        extracted = extract_supported_files(self.archive)
        try:
            self.assertEqual(len(extracted), 2)
            self.assertEqual((Path(extracted[0]).read_text(), Path(extracted[1]).read_text()),
                             ("first", "second"))
            self.assertNotEqual(extracted[0], extracted[1])
        finally:
            from src.zip_utils import cleanup_dir
            cleanup_dir(extracted[0].parents[1])

    def test_skips_path_traversal_and_unsupported_files(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("../escape.txt", "bad")
            archive.writestr("/absolute.txt", "bad")
            archive.writestr("notes.bin", "ignored")
            archive.writestr("docs/good.md", "safe")
        destination = self.root / "out"
        extracted = extract_supported_files(self.archive, destination)
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].read_text(), "safe")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_enforces_file_count_limit(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("a.txt", "a")
            archive.writestr("b.txt", "b")
        with self.assertRaises(ValueError):
            extract_supported_files(self.archive, max_files=1)

    def test_enforces_uncompressed_size_limit(self):
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("large.txt", "x" * 100)
        with self.assertRaises(ValueError):
            extract_supported_files(self.archive, max_file_bytes=50)

    def test_rejects_preexisting_symlink_in_destination(self):
        outside = self.root / "outside"
        outside.mkdir()
        destination = self.root / "out"
        destination.mkdir()
        try:
            (destination / "linked").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this platform")
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("linked/escape.txt", "must not escape")
        with self.assertRaises(ValueError):
            extract_supported_files(self.archive, destination)
        self.assertFalse((outside / "escape.txt").exists())

    def test_caps_archive_listing_metadata(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("a.txt", "a")
            archive.writestr("b.txt", "b")
        with self.assertRaises(ValueError):
            list_zip_contents(self.archive, max_entries=1)


class RetrievalTests(unittest.TestCase):
    def test_long_paragraph_chunks_are_bounded_and_overlap_validated(self):
        chunks = chunk_text("word " * 1000, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)

    def test_empty_query_returns_document_order(self):
        chunks = ["first", "second", "third"]
        self.assertEqual([item[0] for item in simple_retrieve("", chunks, 2)],
                         ["first", "second"])

    def test_context_obeys_character_limit(self):
        result = build_context("topic", "topic " * 1000, max_context_chars=120)
        self.assertLessEqual(len(result), 120)


if __name__ == "__main__":
    unittest.main()


class InferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.previous_model = llm._llm
        llm._llm = lambda *args, **kwargs: {"choices": [{"text": "ok"}]}

    def tearDown(self):
        llm._llm = self.previous_model

    def test_rejects_nonfinite_temperature(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    llm.generate("hello", temperature=value)

    def test_rejects_noninteger_token_count(self):
        with self.assertRaises(ValueError):
            llm.generate("hello", max_tokens=1.5)

    def test_rejects_invalid_stop_sequence(self):
        with self.assertRaises(ValueError):
            llm.generate("hello", stop="STOP")
