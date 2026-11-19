"""End-to-end and failure-path regression tests for the local document pipeline."""
import argparse
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import app, cli, llm, pdf_utils
from src.zip_utils import extract_supported_files


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, _path, strict=False):
        self.is_encrypted = False
        self.pages = [FakePage("alpha"), FakePage("beta")]


class PdfResourceLimitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pdf = Path(self.temp.name) / "fixture.pdf"
        self.pdf.write_bytes(b"%PDF-fake")

    def tearDown(self):
        self.temp.cleanup()

    def test_extracts_page_markers_and_obeys_char_budget(self):
        with patch.object(pdf_utils, "PdfReader", FakeReader):
            text = pdf_utils.extract_text_from_pdf(self.pdf, max_pages=2, max_chars=20)
            self.assertIn("--- Page 1 ---", text)
            with self.assertRaisesRegex(ValueError, "character limit"):
                pdf_utils.extract_text_from_pdf(self.pdf, max_pages=2, max_chars=5)

    def test_rejects_page_count_above_limit_and_file_over_limit(self):
        with patch.object(pdf_utils, "PdfReader", FakeReader):
            limited = pdf_utils.extract_text_from_pdf(self.pdf, max_pages=1)
            self.assertIn("Page 1", limited)
            self.assertNotIn("Page 2", limited)
            self.assertIn("1 of 2 pages", limited)
            with self.assertRaisesRegex(ValueError, "byte"):
                pdf_utils.extract_text_from_pdf(self.pdf, max_file_bytes=2)

    def test_rejects_encrypted_pdf(self):
        class EncryptedReader(FakeReader):
            def __init__(self, path, strict=False):
                super().__init__(path, strict)
                self.is_encrypted = True
        with patch.object(pdf_utils, "PdfReader", EncryptedReader):
            with self.assertRaisesRegex(ValueError, "Encrypted"):
                pdf_utils.extract_text_from_pdf(self.pdf)


class ModelLifecycleTests(unittest.TestCase):
    def test_failed_model_replacement_keeps_previous_instance(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.gguf"
            model.write_bytes(b"model")
            previous = object()
            old_llm, old_path = llm._llm, llm._model_path
            llm._llm, llm._model_path = previous, "previous.gguf"
            class BrokenLlama:
                def __init__(self, **kwargs):
                    raise RuntimeError("backend init failed")
            try:
                with patch.dict("sys.modules", {"llama_cpp": SimpleNamespace(Llama=BrokenLlama)}):
                    with self.assertRaisesRegex(RuntimeError, "backend init"):
                        llm.load_model(model)
                self.assertIs(llm._llm, previous)
                self.assertEqual(llm._model_path, "previous.gguf")
            finally:
                llm._llm, llm._model_path = old_llm, old_path

    def test_model_path_and_stop_sequences_are_validated(self):
        with self.assertRaises(TypeError):
            llm.load_model(None)
        previous = llm._llm
        class Completion:
            def __init__(self):
                self.kwargs = None
            def __call__(self, prompt, **kwargs):
                self.kwargs = kwargs
                return {"choices": [{"text": "ok"}]}
        model = Completion()
        llm._llm = model
        try:
            with self.assertRaises(ValueError):
                llm.generate("hello", stop=[""])
            self.assertEqual(llm.generate("hello", stop=[]), "ok")
            self.assertEqual(model.kwargs["stop"], [])
        finally:
            llm._llm = previous


class ZipRollbackTests(unittest.TestCase):
    def test_failed_nested_extraction_removes_new_parent_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "bad-order.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/item.txt", "first")
                archive.writestr("nested/item.txt/child.txt", "conflict")
            destination = root / "created" / "deep" / "destination"
            with self.assertRaises((ValueError, NotADirectoryError)):
                extract_supported_files(archive_path, destination)
            self.assertFalse(destination.exists())
            self.assertFalse((root / "created").exists())


class CliAndUiTests(unittest.TestCase):
    def test_process_zip_requires_model_for_question_or_summary(self):
        args = argparse.Namespace(
            zip="missing.zip", output="results.zip", question=None,
            summarize=True, model=None
        )
        with self.assertRaisesRegex(ValueError, "--model is required"):
            cli.cmd_process_zip(args)

    def test_extract_only_rejects_archive_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.pdf"
            source.write_bytes(b"placeholder")
            args = argparse.Namespace(pdf=str(source), output=str(Path(temp) / "output.zip"),
                                      max_pages=10, extract_only=True)
            with self.assertRaisesRegex(ValueError, ".txt extension"):
                cli.cmd_process_pdf(args)

    def test_cli_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.pdf"
            source.write_bytes(b"not a real pdf")
            args = argparse.Namespace(pdf=str(source), output=str(source), max_pages=10)
            with self.assertRaisesRegex(ValueError, "different from the input"):
                cli.cmd_process_pdf(args)
            self.assertEqual(source.read_bytes(), b"not a real pdf")

    def test_process_zip_extracts_files_and_writes_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "documents.zip"
            output = root / "results.zip"
            with zipfile.ZipFile(source, "w") as zf:
                zf.writestr("nested/readme.txt", "CLI extraction works.")
            args = argparse.Namespace(
                zip=str(source), output=str(output), question=None,
                summarize=False, model=None
            )
            cli.cmd_process_zip(args)
            self.assertTrue(output.is_file())
            with zipfile.ZipFile(output) as result:
                content = result.read("combined_extracted.txt").decode("utf-8")
            self.assertIn("CLI extraction works.", content)

    def test_process_zip_rejects_non_zip_output_extension(self):
        args = argparse.Namespace(
            zip="input.zip", output="results.txt", question=None,
            summarize=False, model=None
        )
        with self.assertRaisesRegex(ValueError, "must use a .zip extension"):
            cli.cmd_process_zip(args)

    def test_upload_path_supports_gradio_filedata_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "upload.pdf"
            path.write_bytes(b"fixture")
            self.assertEqual(app._upload_path({"path": str(path)}), path)
            self.assertEqual(app._upload_path(path), path)

    def test_ui_rejects_bad_model_path_before_loading_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "input.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("doc.txt", "hello")
            with patch.object(app.llm, "load_model") as load:
                answer, output = app.process_upload(
                    str(archive), "", str(Path(temp) / "missing.gguf"), 128, 0.5
                )
            self.assertIn("Model path is invalid", answer)
            self.assertIsNone(output)
            load.assert_not_called()

    def test_ui_processes_zip_and_returns_downloadable_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "input.zip"
            model = root / "model.gguf"
            model.write_bytes(b"model")
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("docs/notes.txt", "A useful note about testing.")
            with patch.object(app.llm, "is_loaded", return_value=True), \
                 patch.object(app.llm, "loaded_model_path", return_value=str(model.resolve())), \
                 patch.object(app.llm, "loaded_model_config", return_value={"n_ctx": 4096, "n_threads": None, "n_gpu_layers": 0, "verbose": False}), \\
                 patch.object(app.llm, "generate", return_value="A concise answer."):
                answer, output = app.process_upload(str(archive), "What matters?",
                                                    str(model), 128, 0.5)
            try:
                self.assertEqual(answer, "A concise answer.")
                self.assertTrue(Path(output).is_file())
                with zipfile.ZipFile(output) as result:
                    self.assertIn("answer.txt", result.namelist())
                    self.assertIn("combined_extracted.txt", result.namelist())
            finally:
                if output:
                    Path(output).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
