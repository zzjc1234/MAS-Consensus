import re
import logging
from typing import List

import fitz

SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def read_text_file(file_path: str) -> str:
    """Reads a text file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_text_with_pymupdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF, attempting to remove headers, footers, and citation markers.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting text from PDF: {pdf_path}")
    references_pattern = re.compile(r'^references?$|^bibliography$')
    citation_pattern = re.compile(r'^\\[\\d+\\]|\\d+\\.^|^[A-Z][a-z]+, *[A-Z]\\.|^\\s*[A-Z][a-z]+\\s+[A-Z][a-z]+')
    number_pattern = re.compile(r'^\\[\\d+\\]$|^\\d+\\.')
    year_pattern = re.compile(r'^.*?\\(\\d{4}\\)[.,].*?$')
    whitespace_pattern = re.compile(r'\\s+')

    full_text: List[str] = []
    in_references = False

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))

                for block in blocks:
                    text = block[4].strip() if block[4] else ''
                    if not text:
                        continue

                    if references_pattern.match(text.lower()):
                        in_references = True
                        continue

                    if in_references and citation_pattern.match(text):
                        continue

                    if number_pattern.match(text):
                        continue

                    if year_pattern.match(text) and len(text.split()) < 20:
                        continue

                    text = whitespace_pattern.sub(' ', text)
                    if text.isupper() or len(text) <= 3:
                        full_text.append(f"\n## {text}\n")
                    else:
                        full_text.append(text + "\n")
    except Exception as e:
        logger.error(f"Error processing PDF file: {str(e)}")
        raise ValueError(f"Error processing PDF file: {str(e)}")

    logger.info(f"Finished extracting text from PDF. Total lines: {len(full_text)}")
    return "\n".join(full_text)


def extract_text(file_path: str) -> str:
    """
    Extracts text from a file, supporting .txt and .pdf formats.
    """
    if file_path.lower().endswith('.txt'):
        return read_text_file(file_path)
    elif file_path.lower().endswith('.pdf'):
        return extract_text_with_pymupdf(file_path)
    else:
        raise ValueError(f"Unsupported file format. File must be either .pdf or .txt")