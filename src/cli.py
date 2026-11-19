"""Command-line interface for standalone LLM + document tools."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from .pdf_utils import extract_text_from_pdf, get_pdf_page_count
from .zip_utils import list_zip_contents, extract_supported_files, create_results_zip, cleanup_dir
from .rag import build_context
from . import llm


def _reject_input_overwrite(source: str | Path, output: str | Path) -> None:
    """Prevent an output path from replacing its source document/archive."""
    if Path(source).expanduser().resolve() == Path(output).expanduser().resolve():
        raise ValueError("Output path must be different from the input file")


def _write_text_atomic(text: str, output: str | Path) -> Path:
    """Write extracted text without leaving a partial or symlink-following output."""
    target = Path(output)
    if target.is_symlink():
        raise ValueError("Text output path must not be a symbolic link")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if target.is_symlink():
            raise ValueError("Text output path became a symbolic link")
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return target


def cmd_chat(args: argparse.Namespace) -> None:
    llm.load_model(args.model, n_ctx=args.ctx, n_threads=args.threads, n_gpu_layers=args.gpu_layers)
    print("Model loaded. Type 'quit' or Ctrl+C to exit.\n")
    system = args.system or "You are a helpful assistant."
    try:
        while True:
            user = input("You: ").strip()
            if not user or user.lower() in {"quit", "exit", "q"}:
                break
            reply = llm.generate(user, system=system, max_tokens=args.max_tokens, temperature=args.temp)
            print(f"\nAssistant: {reply}\n")
    except KeyboardInterrupt:
        print("\nBye.")


def cmd_process_pdf(args: argparse.Namespace) -> None:
    if args.output:
        _reject_input_overwrite(args.pdf, args.output)
        output_suffix = Path(args.output).suffix.lower()
        if args.extract_only and output_suffix != ".txt":
            raise ValueError("--extract-only output must use a .txt extension")
        if not args.extract_only and output_suffix != ".zip":
            raise ValueError("Q&A results output must use a .zip extension")
    text = extract_text_from_pdf(args.pdf, max_pages=args.max_pages)
    pages = get_pdf_page_count(args.pdf)
    page_limit = args.max_pages if args.max_pages is not None else 500
    extracted_pages = min(pages, page_limit)
    print(
        f"PDF has {pages} page(s); extracted {extracted_pages} page(s). "
        f"Length: {len(text)} chars.\n"
    )

    if args.extract_only:
        out = Path(args.output) if args.output else Path(args.pdf).with_suffix(".txt")
        _write_text_atomic(text, out)
        print(f"Saved raw text to {out}")
        return

    if not args.model:
        print("Provide --model to ask questions / summarize.", file=sys.stderr)
        sys.exit(1)

    llm.load_model(args.model, n_ctx=args.ctx, n_threads=args.threads, n_gpu_layers=args.gpu_layers)

    question = args.question or "Provide a concise summary of the document, including key points."
    context = build_context(question, text)
    prompt = f"Document context:\n\n{context}\n\nQuestion: {question}"
    answer = llm.generate(prompt, max_tokens=args.max_tokens, temperature=args.temp)
    print(answer)

    if args.output:
        results = {
            "extracted_text.txt": text,
            "answer.txt": answer,
            "question.txt": question,
        }
        zip_path = create_results_zip(results, args.output)
        print(f"\nResults written to {zip_path}")


def cmd_process_zip(args: argparse.Namespace) -> None:
    _reject_input_overwrite(args.zip, args.output or "results.zip")
    if (args.output and Path(args.output).suffix.lower() != ".zip"):
        raise ValueError("ZIP processing output must use a .zip extension")
    if (args.question or args.summarize) and not args.model:
        raise ValueError("--model is required when --question or --summarize is used")
    contents = list_zip_contents(args.zip)
    print(f"ZIP contains {len(contents)} file(s):")
    for item in contents[:20]:
        print(f"  - {item['name']} ({item['size']} bytes)")
    if len(contents) > 20:
        print(f"  ... and {len(contents) - 20} more")

    extract_dir = None
    try:
        extract_dir = Path(tempfile.mkdtemp(prefix="llm_zip_cli_"))
        extracted_files = extract_supported_files(args.zip, dest_dir=extract_dir)
        if not extracted_files:
            print("ZIP contains no supported document files.", file=sys.stderr)
            return
        print(f"\nExtracted {len(extracted_files)} supported file(s) to temporary directory.")
        all_text_parts = []
        for fpath in extracted_files:
            try:
                if fpath.suffix.lower() == ".pdf":
                    text = extract_text_from_pdf(fpath)
                else:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                all_text_parts.append(f"=== {fpath.as_posix()} ===\n{text}")
            except Exception as exc:
                all_text_parts.append(
                    f"=== {fpath.name} (error: {type(exc).__name__}: {exc}) ==="
                )

        combined = "\n\n".join(all_text_parts)
        results = {"combined_extracted.txt": combined}
        if args.model and (args.question or args.summarize):
            llm.load_model(args.model, n_ctx=args.ctx, n_threads=args.threads,
                           n_gpu_layers=args.gpu_layers)
            question = args.question or "Summarize the contents of these documents."
            context = build_context(question, combined)
            answer = llm.generate(f"Documents:\n\n{context}\n\nQuestion: {question}",
                                  max_tokens=args.max_tokens, temperature=args.temp)
            results["answer.txt"] = answer
            results["question.txt"] = question
            print("\n--- Answer ---\n")
            print(answer)

        zip_path = create_results_zip(results, args.output or "results.zip")
        print(f"\nResults ZIP written to {zip_path}")
    finally:
        if extract_dir:
            cleanup_dir(extract_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Local LLM with PDF & ZIP support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared model args
    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", "-m", type=str, help="Path to GGUF model")
        p.add_argument("--ctx", type=int, default=4096, help="Context size")
        p.add_argument("--threads", "-t", type=int, default=None, help="CPU threads")
        p.add_argument("--gpu-layers", type=int, default=0, help="GPU layers (0=CPU)")
        p.add_argument("--max-tokens", type=int, default=512)
        p.add_argument("--temp", type=float, default=0.7)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat")
    add_model_args(p_chat)
    p_chat.add_argument("--system", type=str, default=None)
    p_chat.set_defaults(func=cmd_chat)

    # process-pdf
    p_pdf = sub.add_parser("process-pdf", help="Extract / Q&A a PDF")
    add_model_args(p_pdf)
    p_pdf.add_argument("pdf", type=str, help="PDF file path")
    p_pdf.add_argument("--question", "-q", type=str, default=None)
    p_pdf.add_argument("--max-pages", type=int, default=500,
                       help="Maximum PDF pages to extract (hard safety limit)")
    p_pdf.add_argument("--extract-only", action="store_true")
    p_pdf.add_argument("--output", "-o", type=str, default=None, help="Output ZIP or TXT path")
    p_pdf.set_defaults(func=cmd_process_pdf)

    # process-zip
    p_zip = sub.add_parser("process-zip", help="Process a ZIP of documents")
    add_model_args(p_zip)
    p_zip.add_argument("zip", type=str, help="ZIP file path")
    p_zip.add_argument("--question", "-q", type=str, default=None)
    p_zip.add_argument("--summarize", action="store_true")
    p_zip.add_argument("--output", "-o", type=str, default="results.zip")
    p_zip.set_defaults(func=cmd_process_zip)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
