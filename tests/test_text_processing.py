import os
import unittest
from mas_consensus.text_processing import read_text_file, extract_text


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


if __name__ == "__main__":
    unittest.main()
