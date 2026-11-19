"""CLI output-path regression tests."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import cli


class CliPdfOutputTests(unittest.TestCase):
    def test_extract_only_creates_missing_output_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.pdf"
            source.write_bytes(b"pdf placeholder")
            output = Path(temp) / "nested" / "reports" / "output.txt"
            args = argparse.Namespace(pdf=str(source), max_pages=1, extract_only=True,
                                      output=str(output))
            with patch.object(cli, "extract_text_from_pdf", return_value="extracted text"), \
                 patch.object(cli, "get_pdf_page_count", return_value=1):
                cli.cmd_process_pdf(args)
            self.assertEqual(output.read_text(encoding="utf-8"), "extracted text")


if __name__ == "__main__":
    unittest.main()
