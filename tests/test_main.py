import unittest
from unittest.mock import Mock, patch
import argparse
import os
import tempfile

from mas_consensus.main import download_file, main


class TestMainModule(unittest.TestCase):

    def test_download_file_existing(self):
        with patch("mas_consensus.main.logging") as mock_logging:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file_path = tmp_file.name

            # File already exists
            download_file("http://example.com/test.pdf", tmp_file_path)

            # Verify logging was called
            mock_logging.info.assert_called_with(
                f"{tmp_file_path} already exists. Skipping the download"
            )

            # Clean up
            os.unlink(tmp_file_path)

    @patch("mas_consensus.main.requests")
    def test_download_file_new(self, mock_requests):
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status.return_value = None
        mock_requests.get.return_value = mock_response

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            # Remove the file to simulate it not existing
            os.unlink(tmp_file_path)
        except:
            pass  # File might not exist, that's fine

        with patch("mas_consensus.main.logging"):
            download_file("http://example.com/test.pdf", tmp_file_path)

            # Verify requests.get was called
            mock_requests.get.assert_called_once_with(
                "http://example.com/test.pdf", stream=True
            )

            # Verify file was created
            self.assertTrue(os.path.exists(tmp_file_path))

            # Clean up
            os.unlink(tmp_file_path)

    @patch("mas_consensus.main.requests")
    def test_download_file_exception(self, mock_requests):
        mock_requests.get.side_effect = Exception("Network error")

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            # Remove the file to simulate it not existing
            os.unlink(tmp_file_path)
        except:
            pass  # File might not exist, that's fine

        with patch("mas_consensus.main.logging"):
            with self.assertRaises(Exception):
                download_file("http://example.com/test.pdf", tmp_file_path)


if __name__ == "__main__":
    unittest.main()
