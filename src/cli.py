"""Command-line interface for standalone LLM + document tools."""

import argparse
import sys
from pathlib import Path

from .pdf_utils import extract_text_from_pdf, get_pdf_page_count
from .zip_utils import list_zip_contents, extract_supported_files, create_results_zip, cleanup_dir
from .rag import build_context
from . import llm


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
    text = extract_text_from_pdf(args.pdf, max_pages=args.max_pages)
    pages = get_pdf_page_count(args.pdf)
    print(f"Extracted text from {pages} page(s). Length: {len(text)} chars.\n")

    if args.extract_only:
        out = Path(args.output) if args.output else Path(args.pdf).with_suffix(".txt")
        out.write_text(text, encoding="utf-8")
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
    contents = list_zip_contents(args.zip)
    print(f"ZIP contains {len(contents)} file(s):")
    for item in contents[:20]:
        print(f"  - {item['name']} ({item['size']} bytes)")
    if len(contents) > 20:
        print(f"  ... and {len(contents) - 20} more")

    extract_dir = None
    try:
        extract_dir = extract_supported_files(args.zip)
        print(f"\nExtracted {len(extract_dir)} supported file(s) to temporary directory.")
        all_text_parts = []
        for fpath in extract_dir:
            try:
                if fpath.suffix.lower() == ".pdf":
                    text = extract_text_from_pdf(fpath)
                else:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                all_text_parts.append(f"=== {fpath.as_posix()} ===\n{text}")
            except Exception as exc:
                all_text_parts.append(f"=== {fpath.name} (error: {type(exc).__name__}) ===")

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
    p_pdf.add_argument("--max-pages", type=int, default=None)
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
