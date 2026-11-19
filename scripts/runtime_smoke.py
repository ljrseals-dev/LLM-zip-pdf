"""Real GGUF inference and live Gradio upload smoke test.

Run with:
  LLM_ZIP_PDF_MODEL=/path/to/model.gguf python scripts/runtime_smoke.py
"""
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from gradio_client import Client, handle_file

from src import llm


def wait_for_server(process, url, timeout=120):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Gradio server exited before becoming ready")
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Gradio server did not become ready: {last_error}")


def main():
    model = Path(os.environ.get("LLM_ZIP_PDF_MODEL", "")).expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"Set LLM_ZIP_PDF_MODEL to an existing GGUF file; got: {model}")

    print(f"[1/3] Loading real GGUF model: {model.name}", flush=True)
    llm.load_model(model, n_ctx=1024, n_threads=2, n_gpu_layers=0)
    answer = llm.generate(
        "Reply with the single word GREEN.",
        max_tokens=16,
        temperature=0.0,
    )
    if not answer.strip():
        raise AssertionError("Real GGUF inference returned an empty response")
    print(f"Real inference returned: {answer[:160]!r}", flush=True)
    llm.unload()

    with tempfile.TemporaryDirectory(prefix="llm_zip_pdf_smoke_") as temp:
        root = Path(temp)
        model_path = str(model)
        archive = root / "upload.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(
                "validation-note.txt",
                "Integration validation note. The sentinel word is ORCHID. "
                "This text must be extracted from the uploaded ZIP.",
            )

        log_path = root / "gradio.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
        process = subprocess.Popen(
            [sys.executable, "-m", "src.app"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            print("[2/3] Waiting for the live Gradio server", flush=True)
            wait_for_server(process, "http://127.0.0.1:7860/config")

            print("[3/3] Uploading a real ZIP through the Gradio client", flush=True)
            client = Client("http://127.0.0.1:7860", verbose=False)
            result = client.predict(
                handle_file(str(archive)),
                "What is the sentinel word in the uploaded note?",
                model_path,
                96,
                0.0,
                api_name="/process_upload",
            )
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise AssertionError(f"Unexpected Gradio response shape: {type(result)!r}")
            generated_answer, result_file = result
            if not isinstance(generated_answer, str) or not generated_answer.strip():
                raise AssertionError(f"Gradio returned no answer: {generated_answer!r}")

            if isinstance(result_file, dict):
                result_path = result_file.get("path") or result_file.get("name")
            else:
                result_path = getattr(result_file, "path", None) or result_file
            if not result_path or not Path(result_path).is_file():
                raise AssertionError(f"Gradio did not return a downloadable result ZIP: {result_file!r}")
            if not zipfile.is_zipfile(result_path):
                raise AssertionError("Gradio result is not a valid ZIP archive")

            with zipfile.ZipFile(result_path) as output:
                names = set(output.namelist())
                if "combined_extracted.txt" not in names or "answer.txt" not in names:
                    raise AssertionError(f"Result ZIP is missing expected outputs: {sorted(names)}")
                extracted = output.read("combined_extracted.txt").decode("utf-8")
                saved_answer = output.read("answer.txt").decode("utf-8")
                if "ORCHID" not in extracted:
                    raise AssertionError("Uploaded document text was not preserved in result ZIP")
                if not saved_answer.strip():
                    raise AssertionError("Result ZIP contains an empty answer")

            print("PASS: real GGUF inference succeeded.", flush=True)
            print("PASS: live Gradio server launched and accepted a ZIP upload.", flush=True)
            print("PASS: model-generated answer and extracted source were returned in a ZIP.", flush=True)
            print(f"Gradio answer: {generated_answer[:240]!r}", flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            log_handle.close()
            if process.returncode not in (0, -15, 143):
                print(f"Gradio server exit code: {process.returncode}", file=sys.stderr)
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                if "Traceback" in log_text:
                    print("--- Gradio server log ---", file=sys.stderr)
                    print(log_text[-12000:], file=sys.stderr)


if __name__ == "__main__":
    main()
