"""Security and failure-path regression tests for ZIP, RAG and PDF helpers."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src import pdf_utils
from src.rag import build_context, chunk_text, simple_retrieve
from src.zip_utils import (
    create_results_zip, extract_supported_files, list_zip_contents,
)


class ZipHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "input.zip"

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_invalid_limit_types(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("a.txt", "a")
        with self.assertRaises(TypeError):
            list_zip_contents(self.archive, max_entries=True)
        with self.assertRaises(TypeError):
            extract_supported_files(self.archive, max_files=1.5)

    def test_rejects_symlink_member_even_when_named_like_directory(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            info = zipfile.ZipInfo("linked/")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, "outside")
        with self.assertRaises(ValueError):
            list_zip_contents(self.archive)

    def test_rolls_back_new_files_in_caller_destination_on_conflict(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("new.txt", "new")
            archive.writestr("existing.txt", "archive")
        destination = self.root / "destination"
        destination.mkdir()
        (destination / "existing.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError):
            extract_supported_files(self.archive, destination)
        self.assertFalse((destination / "new.txt").exists())
        self.assertEqual((destination / "existing.txt").read_text(encoding="utf-8"), "keep")

    def test_rejects_result_symlink_and_preserves_existing_output_on_bad_input(self):
        output = self.root / "results.zip"
        output.write_bytes(b"previous")
        with self.assertRaises(TypeError):
            create_results_zip({"answer.txt": object()}, output)
        self.assertEqual(output.read_bytes(), b"previous")
        target = self.root / "target.zip"
        target.write_bytes(b"target")
        link = self.root / "link.zip"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaises(ValueError):
            create_results_zip({"answer.txt": "ok"}, link)
        self.assertEqual(target.read_bytes(), b"target")

    def test_result_zip_is_readable_and_flattened_names_are_unique(self):
        output = create_results_zip({"folder/a.txt": "hello"}, self.root / "out.zip")
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.namelist(), ["a.txt"])
            self.assertEqual(archive.read("a.txt"), b"hello")


class PdfAndRetrievalHardeningTests(unittest.TestCase):
    def test_pdf_page_limit_rejects_bool_and_float(self):
        for value in (True, 1.5, 0):
            with self.subTest(value=value):
                expected = TypeError if isinstance(value, bool) or isinstance(value, float) else ValueError
                with self.assertRaises(expected):
                    pdf_utils.extract_text_from_pdf("unused.pdf", max_pages=value)

    def test_retrieval_rejects_invalid_query_and_chunk_collection(self):
        with self.assertRaises(TypeError):
            simple_retrieve(None, ["text"])
        with self.assertRaises(TypeError):
            build_context(None, "document")

    def test_chunking_enforces_document_and_chunk_budgets(self):
        with self.assertRaisesRegex(ValueError, "character chunking limit"):
            chunk_text("x" * 20, max_chars=10)
        with self.assertRaisesRegex(ValueError, "chunk limit"):
            chunk_text("a b c", chunk_size=1, overlap=0, max_chunks=2)
        with self.assertRaises(TypeError):
            chunk_text("text", max_chunks=True)

    def test_zero_context_is_empty_and_chunks_are_bounded(self):
        self.assertEqual(build_context("x", "document", max_context_chars=0), "")
        chunks = chunk_text("ab " * 500, chunk_size=31, overlap=7)
        self.assertTrue(all(len(item) <= 31 for item in chunks))


if __name__ == "__main__":
    unittest.main()
