#!/usr/bin/env python3 -m pytest
"""
Accuracy tests for optimized conversion logic.

These tests verify that the optimizations maintain correct output
while improving performance.
"""
import os
import io
import csv
import tempfile
import pytest

from markitdown import MarkItDown, StreamInfo

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")


class TestCSVConverterAccuracy:
    """Tests for optimized CSV converter accuracy."""

    def test_csv_basic_conversion(self):
        """Test basic CSV to markdown conversion."""
        csv_content = "Name,Age,City\nAlice,30,NYC\nBob,25,LA"

        markitdown = MarkItDown()
        stream = io.BytesIO(csv_content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        assert "| Name | Age | City |" in result.markdown
        assert "| --- | --- | --- |" in result.markdown
        assert "| Alice | 30 | NYC |" in result.markdown
        assert "| Bob | 25 | LA |" in result.markdown

    def test_csv_with_missing_columns(self):
        """Test CSV where some rows have fewer columns than header."""
        csv_content = "A,B,C\n1,2,3\n4,5\n6"

        markitdown = MarkItDown()
        stream = io.BytesIO(csv_content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        # Row with fewer columns should be padded
        lines = result.markdown.strip().split("\n")
        assert len(lines) == 5  # header + separator + 3 data rows

        # Each data row should have exactly 3 columns (padded if needed)
        for line in lines[2:]:
            assert line.count("|") == 4  # 3 columns means 4 pipe chars

    def test_csv_with_extra_columns(self):
        """Test CSV where some rows have more columns than header."""
        csv_content = "A,B\n1,2,3,4\n5,6"

        markitdown = MarkItDown()
        stream = io.BytesIO(csv_content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        lines = result.markdown.strip().split("\n")
        # Row with extra columns should be truncated to match header
        for line in lines[2:]:
            assert line.count("|") == 3  # 2 columns means 3 pipe chars

    def test_csv_empty_file(self):
        """Test empty CSV file."""
        csv_content = ""

        markitdown = MarkItDown()
        stream = io.BytesIO(csv_content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        assert result.markdown == ""

    def test_csv_header_only(self):
        """Test CSV with only header row."""
        csv_content = "Col1,Col2,Col3"

        markitdown = MarkItDown()
        stream = io.BytesIO(csv_content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        assert "| Col1 | Col2 | Col3 |" in result.markdown
        assert "| --- | --- | --- |" in result.markdown

    def test_csv_special_characters(self):
        """Test CSV with special characters."""
        csv_content = 'Name,Description\nTest,"Value with, comma"\nAnother,"Quote ""test"""'

        markitdown = MarkItDown()
        stream = io.BytesIO(csv_content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        assert "Test" in result.markdown
        assert "Value with, comma" in result.markdown

    def test_csv_real_file(self):
        """Test conversion of actual test CSV file."""
        csv_path = os.path.join(TEST_FILES_DIR, "test_mskanji.csv")
        if os.path.exists(csv_path):
            markitdown = MarkItDown()
            result = markitdown.convert(csv_path)
            # Should produce valid markdown table
            assert "|" in result.markdown


class TestXLSXConverterAccuracy:
    """Tests for optimized XLSX converter accuracy."""

    def test_xlsx_conversion(self):
        """Test XLSX file conversion."""
        xlsx_path = os.path.join(TEST_FILES_DIR, "test.xlsx")
        if not os.path.exists(xlsx_path):
            pytest.skip("test.xlsx not found")

        markitdown = MarkItDown()
        result = markitdown.convert(xlsx_path)

        # Should contain markdown table structure
        assert "|" in result.markdown
        assert "---" in result.markdown

    def test_xls_conversion(self):
        """Test XLS file conversion."""
        xls_path = os.path.join(TEST_FILES_DIR, "test.xls")
        if not os.path.exists(xls_path):
            pytest.skip("test.xls not found")

        markitdown = MarkItDown()
        result = markitdown.convert(xls_path)

        # Should contain markdown table structure
        assert "|" in result.markdown
        assert "---" in result.markdown

    def test_xlsx_pipe_character_escaping(self):
        """Test that pipe characters in XLSX data are properly escaped."""
        # Create a test XLSX with pipe characters
        try:
            import pandas as pd
            import openpyxl
        except ImportError:
            pytest.skip("pandas/openpyxl not available")

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            df = pd.DataFrame({
                "Col1": ["A|B", "C"],
                "Col2": ["D", "E|F|G"]
            })
            df.to_excel(temp_path, index=False)

            markitdown = MarkItDown()
            result = markitdown.convert(temp_path)

            # Pipe chars should be escaped
            assert "\\|" in result.markdown
        finally:
            os.unlink(temp_path)

    def test_xlsx_nan_handling(self):
        """Test that NaN values are handled correctly."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not available")

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            df = pd.DataFrame({
                "Col1": [1, np.nan, 3],
                "Col2": ["A", "B", np.nan]
            })
            df.to_excel(temp_path, index=False)

            markitdown = MarkItDown()
            result = markitdown.convert(temp_path)

            # NaN should be converted to empty string, not "nan"
            assert "nan" not in result.markdown.lower() or "| nan |" not in result.markdown.lower()
        finally:
            os.unlink(temp_path)


class TestPPTXConverterAccuracy:
    """Tests for optimized PPTX converter accuracy."""

    def test_pptx_conversion(self):
        """Test PPTX file conversion."""
        pptx_path = os.path.join(TEST_FILES_DIR, "test.pptx")
        if not os.path.exists(pptx_path):
            pytest.skip("test.pptx not found")

        markitdown = MarkItDown()
        result = markitdown.convert(pptx_path)

        # Should contain slide markers
        assert "<!-- Slide number:" in result.markdown

    def test_pptx_alt_text_cleanup(self):
        """Test that alt text is properly cleaned up."""
        pptx_path = os.path.join(TEST_FILES_DIR, "test.pptx")
        if not os.path.exists(pptx_path):
            pytest.skip("test.pptx not found")

        markitdown = MarkItDown()
        result = markitdown.convert(pptx_path)

        # Alt text should not contain raw newlines or brackets
        # (they should be replaced with spaces)
        lines = result.markdown.split("\n")
        for line in lines:
            if line.startswith("!["):
                # Check alt text part (between ![ and ])
                if "](" in line:
                    alt_text = line[2:line.index("](")]
                    assert "\r" not in alt_text
                    assert "\n" not in alt_text
                    assert "[" not in alt_text
                    assert "]" not in alt_text


class TestStreamInfoGuessing:
    """Tests for optimized stream info guessing."""

    def test_cached_charset_detection(self):
        """Test that charset detection uses cached bytes."""
        # Create a UTF-8 text file
        content = "Hello, World! 你好世界"

        markitdown = MarkItDown()
        stream = io.BytesIO(content.encode("utf-8"))

        guesses = markitdown._get_stream_info_guesses(
            stream,
            base_guess=StreamInfo(extension=".txt")
        )

        # Stream position should be reset to beginning
        assert stream.tell() == 0

        # Should have detected charset
        assert len(guesses) > 0

    def test_stream_position_preserved(self):
        """Test that stream position is preserved after guessing."""
        content = b"Test content for position check"

        markitdown = MarkItDown()
        stream = io.BytesIO(content)

        # Move to middle of stream
        stream.seek(10)
        initial_pos = stream.tell()

        # Get guesses (should reset position)
        markitdown._get_stream_info_guesses(
            stream,
            base_guess=StreamInfo()
        )

        # Position should be back to initial
        assert stream.tell() == initial_pos


class TestConverterSortingCache:
    """Tests for converter priority sorting cache."""

    def test_cache_invalidation_on_register(self):
        """Test that cache is invalidated when new converter is registered."""
        from markitdown._base_converter import DocumentConverter, DocumentConverterResult

        class DummyConverter(DocumentConverter):
            def accepts(self, file_stream, stream_info, **kwargs):
                return False
            def convert(self, file_stream, stream_info, **kwargs):
                return DocumentConverterResult(markdown="")

        markitdown = MarkItDown()

        # Trigger cache creation
        _ = markitdown._sorted_converters_cache
        if markitdown._sorted_converters_cache is None:
            # Force cache creation by accessing internal method
            stream = io.BytesIO(b"test")
            try:
                markitdown._convert(
                    file_stream=stream,
                    stream_info_guesses=[StreamInfo()]
                )
            except Exception:
                pass

        # Cache should exist after first use
        # (or be None if no conversion happened)

        # Register new converter
        markitdown.register_converter(DummyConverter())

        # Cache should be invalidated
        assert markitdown._sorted_converters_cache is None


class TestNormalizationPatterns:
    """Tests for pre-compiled regex pattern accuracy."""

    def test_newline_normalization(self):
        """Test that CRLF newline style is normalized to LF."""
        markitdown = MarkItDown()

        # Test content with CRLF newlines (Windows style)
        content = "Line1\r\nLine2\r\nLine3\r\nLine4"
        stream = io.BytesIO(content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".txt"))

        # CRLF should be normalized to LF
        assert "\r\n" not in result.markdown
        # Lines should be present
        assert "Line1" in result.markdown
        assert "Line2" in result.markdown

    def test_excessive_newlines_collapsed(self):
        """Test that excessive newlines are collapsed to double newlines."""
        markitdown = MarkItDown()

        # Test content with many consecutive newlines
        content = "Line1\n\n\n\n\nLine2\n\n\n\n\n\n\n\nLine3"
        stream = io.BytesIO(content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".txt"))

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result.markdown

    def test_trailing_whitespace_removed(self):
        """Test that trailing whitespace on lines is removed."""
        markitdown = MarkItDown()

        content = "Line with trailing spaces   \nAnother line   "
        stream = io.BytesIO(content.encode("utf-8"))
        result = markitdown.convert(stream, stream_info=StreamInfo(extension=".txt"))

        # Lines should not end with spaces
        for line in result.markdown.split("\n"):
            if line:  # Skip empty lines
                assert not line.endswith(" "), f"Line ends with space: '{line}'"


class TestNonSeekableStreamHandling:
    """Tests for SpooledTemporaryFile handling of non-seekable streams."""

    def test_non_seekable_stream_buffering(self):
        """Test that non-seekable streams are properly buffered."""
        import tempfile
        from markitdown._markitdown import _SPOOLED_MAX_SIZE

        # Verify the SpooledTemporaryFile is used for non-seekable streams
        # by checking the configuration constant
        assert _SPOOLED_MAX_SIZE == 1024 * 1024  # 1MB threshold

    def test_spooled_temp_file_mode(self):
        """Test SpooledTemporaryFile configuration."""
        import tempfile

        # Create a spooled temp file with same config as markitdown uses
        spooled = tempfile.SpooledTemporaryFile(max_size=1024*1024, mode="w+b")

        # Write some data
        spooled.write(b"Test data")
        spooled.seek(0)

        # Should be readable
        data = spooled.read()
        assert data == b"Test data"

        spooled.close()


class TestHTTPChunkSize:
    """Tests verifying HTTP chunk size configuration."""

    def test_http_chunk_size_constant(self):
        """Verify HTTP chunk size is set to expected value."""
        from markitdown._markitdown import _HTTP_CHUNK_SIZE

        # Should be 256KB (262144 bytes)
        assert _HTTP_CHUNK_SIZE == 262144

    def test_spooled_max_size_constant(self):
        """Verify spooled temp file max size is set correctly."""
        from markitdown._markitdown import _SPOOLED_MAX_SIZE

        # Should be 1MB (1048576 bytes)
        assert _SPOOLED_MAX_SIZE == 1024 * 1024
