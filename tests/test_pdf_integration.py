"""Small real-file integration tests for the pypdf boundary."""
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from src.pdf_utils import extract_text_from_pdf, get_pdf_page_count


class PdfFileIntegrationTests(unittest.TestCase):
    def test_blank_pdf_page_count_and_empty_text_are_handled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            with path.open("wb") as output:
                writer.write(output)
            self.assertEqual(get_pdf_page_count(path), 1)
            self.assertEqual(extract_text_from_pdf(path), "(No extractable text found)")

    def test_invalid_pdf_is_reported_as_read_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.pdf"
            path.write_bytes(b"not a PDF")
            with self.assertRaisesRegex(ValueError, "Unable to read PDF"):
                extract_text_from_pdf(path)


if __name__ == "__main__":
    unittest.main()
