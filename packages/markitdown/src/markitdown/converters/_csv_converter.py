import csv
import io
from typing import BinaryIO, Any
from charset_normalizer import from_bytes
from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo

ACCEPTED_MIME_TYPE_PREFIXES = [
    "text/csv",
    "application/csv",
]
ACCEPTED_FILE_EXTENSIONS = [".csv"]


class CsvConverter(DocumentConverter):
    """
    Converts CSV files to Markdown tables.
    """

    def __init__(self):
        super().__init__()

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()
        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True
        for prefix in ACCEPTED_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True
        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        # Read the file content
        if stream_info.charset:
            content = file_stream.read().decode(stream_info.charset)
        else:
            content = str(from_bytes(file_stream.read()).best())

        # Parse CSV content
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            return DocumentConverterResult(markdown="")

        # Pre-compute header column count once (O(1) per row instead of O(n))
        header = rows[0]
        num_cols = len(header)

        # Create markdown table with pre-allocated list capacity hint
        markdown_table = []

        # Add header row
        markdown_table.append("| " + " | ".join(header) + " |")

        # Add separator row (pre-compute separator string)
        separator = "| " + " | ".join(["---"] * num_cols) + " |"
        markdown_table.append(separator)

        # Add data rows - optimized to avoid repeated len() calls and in-place mutations
        for row in rows[1:]:
            row_len = len(row)
            if row_len < num_cols:
                # Pad with empty strings - single operation instead of while loop
                normalized_row = row + [""] * (num_cols - row_len)
            elif row_len > num_cols:
                # Truncate
                normalized_row = row[:num_cols]
            else:
                normalized_row = row
            markdown_table.append("| " + " | ".join(normalized_row) + " |")

        result = "\n".join(markdown_table)

        return DocumentConverterResult(markdown=result)
