"""Local Gradio UI for PDF / ZIP + local LLM."""
import math
import os
import tempfile
from pathlib import Path


from . import llm
from .pdf_utils import extract_text_from_pdf
from .rag import build_context
from .zip_utils import cleanup_dir, create_results_zip, extract_supported_files, list_zip_contents


def _upload_path(file_obj) -> Path:
    """Accept Gradio filepath values and common FileData wrappers."""
    raw_path = file_obj
    if isinstance(file_obj, (str, os.PathLike)):
        raw_path = file_obj
    elif isinstance(file_obj, dict):
        raw_path = file_obj.get("path") or file_obj.get("name")
    else:
        raw_path = (getattr(file_obj, "path", None) or
                    getattr(file_obj, "name", None) or file_obj)
    if not isinstance(raw_path, (str, os.PathLike)):
        raise TypeError("Upload must provide a filesystem path")
    path = Path(raw_path).expanduser()
    if path.is_symlink():
        raise ValueError("Symbolic-link uploads are not supported")
    return path


def process_upload(file_obj, question: str, model_path: str,
                   max_tokens: int, temperature: float):
    if file_obj is None:
        return "Please upload a PDF or ZIP file.", None
    extract_dir = None
    output_path = None
    try:
        path = _upload_path(file_obj)
        if not path.is_file():
            return "Uploaded file is unavailable.", None
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".zip"}:
            return "Unsupported file type. Upload .pdf or .zip.", None
        if not isinstance(model_path, str) or not model_path.strip():
            return "Model path is missing. Provide a valid GGUF path.", None
        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file():
            return "Model path is invalid. Provide a valid GGUF file.", None
        if (isinstance(max_tokens, bool) or
                not isinstance(max_tokens, (int, float)) or
                not math.isfinite(max_tokens) or max_tokens < 1 or
                not float(max_tokens).is_integer()):
            return "Max tokens must be a positive integer.", None
        validated_max_tokens = int(max_tokens)
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or temperature < 0):
            return "Temperature must be a finite non-negative number.", None

        results = {}
        if suffix == ".pdf":
            document = extract_text_from_pdf(path)
            results["extracted_text.txt"] = document
            default_question = "Summarize this document with key points."
            prompt_label = "Document context"
        else:
            contents = list_zip_contents(path)
            results["zip_listing.txt"] = "\n".join(
                f"- {item['name']} ({item['size']} B)" for item in contents[:100]
            )
            extract_dir = Path(tempfile.mkdtemp(prefix="llm_upload_"))
            extracted_files = extract_supported_files(path, dest_dir=extract_dir)
            if not extracted_files:
                return "ZIP contains no supported document files.", None
            parts = []
            for file_path in extracted_files:
                if file_path.suffix.lower() == ".pdf":
                    text = extract_text_from_pdf(file_path)
                else:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                parts.append(f"=== {file_path.as_posix()} ===\n{text}")
            document = "\n\n".join(parts)
            results["combined_extracted.txt"] = document
            default_question = "Summarize the contents of these documents."
            prompt_label = "Documents"

        q = (question or "").strip() or default_question
        context = build_context(q, document)

        # Do not allocate/load a potentially large model until upload parsing and
        # all inexpensive argument checks have succeeded.
        model_key = str(resolved_model)
        expected_config = {
            "n_ctx": 4096, "n_threads": None,
            "n_gpu_layers": 0, "verbose": False
        }
        current_config = llm.loaded_model_config() or {}
        if (not llm.is_loaded() or llm.loaded_model_path() != model_key or
                any(current_config.get(key) != value for key, value in expected_config.items()
                    if key != "effective_n_threads")):
            llm.load_model(model_key, n_ctx=4096)
        answer = llm.generate(
            f"{prompt_label}:\n\n{context}\n\nQuestion: {q}",
            max_tokens=validated_max_tokens, temperature=float(temperature)
        )
        results["answer.txt"] = answer
        results["question.txt"] = q

        with tempfile.NamedTemporaryFile(prefix="llm_results_", suffix=".zip", delete=False) as handle:
            output_path = Path(handle.name)
        create_results_zip(results, output_path)
        return answer, str(output_path)
    except Exception as exc:
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
        return f"Processing failed ({type(exc).__name__}): {exc}", None
    finally:
        if extract_dir:
            cleanup_dir(extract_dir)


def build_ui():
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("gradio is required to run the web UI; install requirements.txt") from exc
    with gr.Blocks(title="Standalone LLM – PDF & ZIP") as demo:
        gr.Markdown(
            "# Standalone Local LLM\nUpload a **PDF** or **ZIP** of documents, ask a question "
            "(or leave blank for a summary), and download extracted text plus the answer."
        )
        with gr.Row():
            with gr.Column():
                model_in = gr.Textbox(label="Path to GGUF model", placeholder="/path/to/model.gguf")
                file_in = gr.File(label="Upload PDF or ZIP", file_types=[".pdf", ".zip"], type="filepath")
                question = gr.Textbox(label="Question (optional – leave empty for summary)",
                                      lines=2, placeholder="What are the main conclusions?")
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                btn = gr.Button("Process", variant="primary")
            with gr.Column():
                answer_out = gr.Textbox(label="Answer / Summary", lines=16)
                zip_out = gr.File(label="Download results ZIP")
        btn.click(fn=process_upload, inputs=[file_in, question, model_in, max_tokens, temperature],
                  outputs=[answer_out, zip_out])
    return demo


def main():
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
