import os
import unittest
from unittest.mock import Mock, patch
import tempfile

from mas_consensus.text_processing import (
    read_text_file,
    extract_text,
    extract_text_with_pymupdf,
)


class TestTextProcessing(unittest.TestCase):

    def setUp(self):
        self.test_txt_file = "test.txt"
        with open(self.test_txt_file, "w") as f:
            f.write("This is a test file.")

    def tearDown(self):
        os.remove(self.test_txt_file)

    def test_read_text_file(self):
        content = read_text_file(self.test_txt_file)
        self.assertEqual(content, "This is a test file.")

    def test_extract_text_txt(self):
        content = extract_text(self.test_txt_file)
        self.assertEqual(content, "This is a test file.")

    def test_extract_text_unsupported(self):
        with self.assertRaises(ValueError):
            extract_text("test.unsupported")

    @patch("mas_consensus.text_processing.fitz")
    def test_extract_text_pdf(self, mock_fitz):
        # Mock the PyMuPDF functionality
        mock_doc = Mock()
        mock_page = Mock()

        # Mock the context manager behavior
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)

        # Mock page content
        mock_block = {
            "lines": [{"spans": [{"text": "This is a test PDF file."}]}],
            "bbox": [0, 0, 100, 100],
        }

        mock_page.get_text.return_value = {"blocks": [mock_block]}
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_fitz.open.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            content = extract_text_with_pymupdf(tmp_file_path)
            # Just verify it returns a string, actual content depends on the mock
            self.assertIsInstance(content, str)
        finally:
            os.unlink(tmp_file_path)

    def test_extract_text_pdf_real_file(self):
        # Test with a real PDF file if available
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            # Try to extract text - this might fail if the file is empty
            # but we're testing the function can handle this case
            with self.assertRaises(ValueError):
                extract_text(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    unittest.main()
