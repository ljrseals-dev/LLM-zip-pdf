"""Model and CLI failure-path regression tests."""
import argparse
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src import llm


class ModelBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.previous_model = llm._llm
        self.previous_path = llm._model_path
        self.previous_config = llm._model_config

    def tearDown(self):
        llm._llm = self.previous_model
        llm._model_path = self.previous_path
        llm._model_config = self.previous_config

    def test_loaded_model_config_tracks_settings_without_exposing_mutable_state(self):
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "model.gguf"
            model_path.write_bytes(b"placeholder")
            class FakeLlama:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
            with patch.dict(sys.modules, {"llama_cpp": types.SimpleNamespace(Llama=FakeLlama)}):
                llm.load_model(model_path, n_ctx=2048, n_threads=2, n_gpu_layers=1, verbose=True)
            config = llm.loaded_model_config()
            self.assertEqual(config["n_ctx"], 2048)
            self.assertEqual(config["n_threads"], 2)
            self.assertEqual(config["effective_n_threads"], 2)
            config["n_ctx"] = 1
            self.assertEqual(llm.loaded_model_config()["n_ctx"], 2048)

    def test_chat_api_failure_is_not_silently_retried_as_completion(self):
        class ChatModel:
            def __init__(self):
                self.completion_calls = 0
            def create_chat_completion(self, **kwargs):
                raise RuntimeError("chat format failure")
            def __call__(self, *args, **kwargs):
                self.completion_calls += 1
                return {"choices": [{"text": "unexpected fallback"}]}

        model = ChatModel()
        llm._llm = model
        with self.assertRaisesRegex(RuntimeError, "chat format failure"):
            llm.generate("hello")
        self.assertEqual(model.completion_calls, 0)

    def test_invalid_chat_response_is_reported(self):
        class ChatModel:
            def create_chat_completion(self, **kwargs):
                return {"choices": [{}]}
        llm._llm = ChatModel()
        with self.assertRaisesRegex(RuntimeError, "invalid chat-completion response"):
            llm.generate("hello")

    def test_failed_model_replacement_preserves_current_model(self):
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "model.gguf"
            model_path.write_bytes(b"placeholder")
            existing = object()
            llm._llm = existing
            llm._model_path = "previous-model"
            llm._model_config = {"n_ctx": 1024, "n_threads": 1, "effective_n_threads": 1,
                                 "n_gpu_layers": 0, "verbose": False}
            previous_config = llm.loaded_model_config()
            class BrokenLlama:
                def __init__(self, **kwargs):
                    raise RuntimeError("backend construction failed")
            fake_module = types.SimpleNamespace(Llama=BrokenLlama)
            with patch.dict(sys.modules, {"llama_cpp": fake_module}):
                with self.assertRaisesRegex(RuntimeError, "backend construction failed"):
                    llm.load_model(model_path)
            self.assertIs(llm._llm, existing)
            self.assertEqual(llm._model_path, "previous-model")
            self.assertEqual(llm.loaded_model_config(), previous_config)


class CliFailurePathTests(unittest.TestCase):
    def test_zip_temp_directory_is_cleaned_when_inference_fails(self):
        cli = importlib.import_module("src.cli")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "document.txt"
            source.write_text("document")
            args = argparse.Namespace(
                zip="archive.zip", model="model.gguf", ctx=128, threads=1,
                gpu_layers=0, question="summarize", summarize=False,
                max_tokens=64, temp=0.2, output=str(Path(temp) / "results.zip"),
            )
            with patch.object(cli, "list_zip_contents", return_value=[]), \
                 patch.object(cli, "extract_supported_files", return_value=[source]), \
                 patch.object(cli, "cleanup_dir") as cleanup, \
                 patch.object(cli.llm, "load_model"), \
                 patch.object(cli.llm, "generate", side_effect=RuntimeError("inference failed")):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    cli.cmd_process_zip(args)
            cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
