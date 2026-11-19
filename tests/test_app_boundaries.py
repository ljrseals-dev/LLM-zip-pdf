"""Regression tests for Gradio upload handling without a Gradio runtime."""
import importlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src import llm


class AppBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The upload handler is deliberately importable without installing Gradio.
        cls.app = importlib.import_module("src.app")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"test model placeholder")
        self.pdf = self.root / "input.pdf"
        self.pdf.write_bytes(b"placeholder")

    def tearDown(self):
        self.temp.cleanup()

    def test_upload_path_accepts_string_and_file_wrapper(self):
        class Upload:
            name = str(self.pdf)
        self.assertEqual(self.app._upload_path(str(self.pdf)), self.pdf)
        self.assertEqual(self.app._upload_path(Upload()), self.pdf)

    def test_missing_or_unsupported_upload_is_rejected_before_model_load(self):
        with patch.object(llm, "load_model") as load:
            self.assertIsNone(self.app.process_upload(None, "", str(self.model), 128, 0.2)[1])
            self.assertIn("Please upload", self.app.process_upload(None, "", str(self.model), 128, 0.2)[0])
            unsupported = self.root / "notes.bin"
            unsupported.write_bytes(b"x")
            self.assertIn("Unsupported", self.app.process_upload(str(unsupported), "", str(self.model), 128, 0.2)[0])
            load.assert_not_called()

    def test_successful_pdf_upload_returns_readable_unique_result_archives(self):
        with patch.object(llm, "is_loaded", return_value=True), \
             patch.object(llm, "loaded_model_path", return_value=str(self.model.resolve())), \
             patch.object(llm, "generate", return_value="answer"), \
             patch.object(self.app, "extract_text_from_pdf", return_value="document text"):
            first = self.app.process_upload(str(self.pdf), "question", str(self.model), 128, 0.2)
            second = self.app.process_upload(str(self.pdf), "question", str(self.model), 128, 0.2)
        self.assertEqual(first[0], "answer")
        self.assertTrue(Path(first[1]).is_file())
        self.assertTrue(Path(second[1]).is_file())
        self.assertNotEqual(first[1], second[1])
        with zipfile.ZipFile(first[1]) as archive:
            self.assertEqual(set(archive.namelist()),
                             {"extracted_text.txt", "answer.txt", "question.txt"})
            self.assertEqual(archive.read("answer.txt"), b"answer")

    def test_zip_with_no_supported_files_returns_clear_message_and_cleans_temp_dir(self):
        archive_path = self.root / "unsupported.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("notes.bin", b"binary")
        with patch.object(llm, "is_loaded", return_value=True), \
             patch.object(llm, "loaded_model_path", return_value=str(self.model.resolve())), \
             patch.object(self.app, "extract_supported_files", return_value=[]), \
             patch.object(self.app, "cleanup_dir", wraps=self.app.cleanup_dir) as cleanup:
            answer, result_path = self.app.process_upload(str(archive_path), "question",
                                                          str(self.model), 128, 0.2)
        self.assertEqual(answer, "ZIP contains no supported document files.")
        self.assertIsNone(result_path)
        cleanup.assert_called_once()

    def test_failed_inference_removes_partial_result_and_cleans_extraction(self):
        archive_path = self.root / "input.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("folder/document.txt", "document text")
        with patch.object(llm, "is_loaded", return_value=True), \
             patch.object(llm, "loaded_model_path", return_value=str(self.model.resolve())), \
             patch.object(llm, "generate", side_effect=RuntimeError("backend failed")), \
             patch.object(self.app, "cleanup_dir", wraps=self.app.cleanup_dir) as cleanup:
            answer, result_path = self.app.process_upload(str(archive_path), "question",
                                                          str(self.model), 128, 0.2)
        self.assertIn("backend failed", answer)
        self.assertIsNone(result_path)
        cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
