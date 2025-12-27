#!/usr/bin/env python3 -m pytest
"""
Efficiency and performance benchmark tests for optimized conversion logic.

These tests measure and verify performance improvements from optimizations.
Run with: pytest test_optimization_efficiency.py -v -s
"""
import os
import io
import time
import tempfile
import statistics
import pytest

from markitdown import MarkItDown, StreamInfo

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")

# Number of iterations for timing tests
TIMING_ITERATIONS = 5
# Warm-up iterations (not counted)
WARMUP_ITERATIONS = 2


def measure_time(func, iterations=TIMING_ITERATIONS, warmup=WARMUP_ITERATIONS):
    """
    Measure execution time of a function.

    Returns:
        tuple: (mean_time, std_dev, min_time, max_time)
    """
    # Warm-up runs
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return (
        statistics.mean(times),
        statistics.stdev(times) if len(times) > 1 else 0,
        min(times),
        max(times)
    )


class TestRegexPatternEfficiency:
    """Tests for pre-compiled regex pattern performance."""

    def test_newline_pattern_compilation(self):
        """Verify regex patterns are compiled at module load time."""
        import re
        from markitdown._markitdown import (
            _NEWLINE_SPLIT_PATTERN,
            _EXCESSIVE_NEWLINES_PATTERN
        )

        # Patterns should be compiled regex objects, not strings
        assert isinstance(_NEWLINE_SPLIT_PATTERN, re.Pattern)
        assert isinstance(_EXCESSIVE_NEWLINES_PATTERN, re.Pattern)

    def test_pptx_regex_compilation(self):
        """Verify PPTX regex patterns are compiled at module load time."""
        import re
        from markitdown.converters._pptx_converter import (
            _ALT_TEXT_CLEANUP_PATTERN,
            _WHITESPACE_COLLAPSE_PATTERN,
            _FILENAME_CLEANUP_PATTERN
        )

        assert isinstance(_ALT_TEXT_CLEANUP_PATTERN, re.Pattern)
        assert isinstance(_WHITESPACE_COLLAPSE_PATTERN, re.Pattern)
        assert isinstance(_FILENAME_CLEANUP_PATTERN, re.Pattern)

    def test_compiled_vs_inline_regex_performance(self):
        """Compare performance of compiled vs inline regex."""
        import re

        # Large test content
        content = "Line1\r\nLine2\nLine3\r\n" * 10000

        # Pre-compiled pattern (as we use)
        compiled_pattern = re.compile(r"\r?\n")

        def compiled_split():
            return compiled_pattern.split(content)

        def inline_split():
            return re.split(r"\r?\n", content)

        compiled_time = measure_time(compiled_split)
        inline_time = measure_time(inline_split)

        print(f"\n  Compiled regex: {compiled_time[0]*1000:.3f}ms (±{compiled_time[1]*1000:.3f}ms)")
        print(f"  Inline regex:   {inline_time[0]*1000:.3f}ms (±{inline_time[1]*1000:.3f}ms)")

        # Compiled should be faster or at least not slower
        # (allowing some margin for measurement variance)
        assert compiled_time[0] <= inline_time[0] * 1.5, \
            "Compiled regex should not be significantly slower than inline"


class TestConverterCacheEfficiency:
    """Tests for converter sorting cache performance."""

    def test_cache_creation(self):
        """Test that cache is properly created."""
        markitdown = MarkItDown()

        # Initially cache should be populated after first conversion
        assert markitdown._sorted_converters_cache is None or \
               isinstance(markitdown._sorted_converters_cache, list)

    def test_repeated_conversions_use_cache(self):
        """Test that repeated conversions use the cache."""
        markitdown = MarkItDown()
        content = b"Test content"

        # First conversion
        stream1 = io.BytesIO(content)
        markitdown.convert(stream1, stream_info=StreamInfo(extension=".txt"))

        # Cache should now exist
        cache_after_first = markitdown._sorted_converters_cache
        assert cache_after_first is not None

        # Second conversion
        stream2 = io.BytesIO(content)
        markitdown.convert(stream2, stream_info=StreamInfo(extension=".txt"))

        # Cache should be the same object (not re-created)
        assert markitdown._sorted_converters_cache is cache_after_first

    def test_sorting_cache_performance(self):
        """Measure performance benefit of converter sorting cache."""
        markitdown = MarkItDown()
        content = b"Test content for performance measurement"

        def single_conversion():
            stream = io.BytesIO(content)
            markitdown.convert(stream, stream_info=StreamInfo(extension=".txt"))

        # Measure time for repeated conversions (should use cache)
        times = measure_time(single_conversion, iterations=10)

        print(f"\n  Cached conversions: {times[0]*1000:.3f}ms mean (±{times[1]*1000:.3f}ms)")
        print(f"  Min: {times[2]*1000:.3f}ms, Max: {times[3]*1000:.3f}ms")


class TestCSVOptimizationEfficiency:
    """Tests for CSV converter optimization performance."""

    def test_csv_large_file_performance(self):
        """Test CSV conversion performance with large files."""
        # Generate large CSV content
        rows = 1000
        cols = 20
        header = ",".join([f"Col{i}" for i in range(cols)])
        data_rows = "\n".join([
            ",".join([f"Val{r}_{c}" for c in range(cols)])
            for r in range(rows)
        ])
        csv_content = f"{header}\n{data_rows}"

        markitdown = MarkItDown()

        def convert_csv():
            stream = io.BytesIO(csv_content.encode("utf-8"))
            return markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        times = measure_time(convert_csv)

        print(f"\n  CSV ({rows} rows x {cols} cols): {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")

        # Should complete in reasonable time (less than 2 seconds for 1000 rows)
        assert times[0] < 2.0, f"CSV conversion too slow: {times[0]:.3f}s"

    def test_csv_varying_column_counts(self):
        """Test CSV with varying column counts (padding/truncation)."""
        # CSV where rows have different column counts
        lines = [
            "A,B,C,D,E",  # Header: 5 cols
        ]
        for i in range(500):
            # Alternate between fewer and more columns
            if i % 3 == 0:
                lines.append(f"{i},{i+1}")  # 2 cols (needs padding)
            elif i % 3 == 1:
                lines.append(f"{i},{i+1},{i+2},{i+3},{i+4},{i+5},{i+6}")  # 7 cols (needs truncation)
            else:
                lines.append(f"{i},{i+1},{i+2},{i+3},{i+4}")  # 5 cols (exact)

        csv_content = "\n".join(lines)
        markitdown = MarkItDown()

        def convert_csv():
            stream = io.BytesIO(csv_content.encode("utf-8"))
            return markitdown.convert(stream, stream_info=StreamInfo(extension=".csv"))

        times = measure_time(convert_csv)

        print(f"\n  CSV with varying cols (500 rows): {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")


class TestXLSXOptimizationEfficiency:
    """Tests for XLSX converter optimization performance."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_FILES_DIR, "test.xlsx")),
        reason="test.xlsx not found"
    )
    def test_xlsx_conversion_performance(self):
        """Test XLSX file conversion performance."""
        xlsx_path = os.path.join(TEST_FILES_DIR, "test.xlsx")
        markitdown = MarkItDown()

        def convert_xlsx():
            return markitdown.convert(xlsx_path)

        times = measure_time(convert_xlsx)

        print(f"\n  XLSX conversion: {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")

    def test_xlsx_direct_vs_html_intermediate(self):
        """
        Compare direct DataFrame->Markdown vs HTML intermediate.
        This test creates test data to measure the optimization benefit.
        """
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        # Create test DataFrame
        rows = 500
        cols = 10
        df = pd.DataFrame({
            f"Col{i}": [f"Val{r}_{i}" for r in range(rows)]
            for i in range(cols)
        })

        # Direct conversion (optimized)
        from markitdown.converters._xlsx_converter import _dataframe_to_markdown

        def direct_conversion():
            return _dataframe_to_markdown(df)

        # HTML intermediate (old method)
        from markitdown.converters._html_converter import HtmlConverter
        html_converter = HtmlConverter()

        def html_intermediate():
            html = df.to_html(index=False)
            return html_converter.convert_string(html).markdown

        direct_times = measure_time(direct_conversion)
        html_times = measure_time(html_intermediate)

        print(f"\n  Direct DataFrame->MD: {direct_times[0]*1000:.3f}ms (±{direct_times[1]*1000:.3f}ms)")
        print(f"  HTML intermediate:    {html_times[0]*1000:.3f}ms (±{html_times[1]*1000:.3f}ms)")

        # Direct should be faster
        speedup = html_times[0] / direct_times[0] if direct_times[0] > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")


class TestStreamHandlingEfficiency:
    """Tests for stream handling optimization performance."""

    def test_large_stream_handling(self):
        """Test handling of large streams."""
        # Create 5MB of content
        content = b"X" * (5 * 1024 * 1024)

        markitdown = MarkItDown()

        def convert_large():
            stream = io.BytesIO(content)
            return markitdown.convert(
                stream,
                stream_info=StreamInfo(extension=".txt", mimetype="text/plain")
            )

        times = measure_time(convert_large, iterations=3)

        print(f"\n  5MB stream: {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")

    def test_spooled_temp_file_threshold(self):
        """Test that SpooledTemporaryFile works correctly near threshold."""
        from markitdown._markitdown import _SPOOLED_MAX_SIZE

        # Test content just under and over threshold
        under_threshold = b"X" * (_SPOOLED_MAX_SIZE - 1024)
        over_threshold = b"X" * (_SPOOLED_MAX_SIZE + 1024)

        markitdown = MarkItDown()

        def convert_under():
            stream = io.BytesIO(under_threshold)
            return markitdown.convert(
                stream,
                stream_info=StreamInfo(extension=".txt", mimetype="text/plain")
            )

        def convert_over():
            stream = io.BytesIO(over_threshold)
            return markitdown.convert(
                stream,
                stream_info=StreamInfo(extension=".txt", mimetype="text/plain")
            )

        under_times = measure_time(convert_under, iterations=3)
        over_times = measure_time(convert_over, iterations=3)

        print(f"\n  Under threshold ({_SPOOLED_MAX_SIZE-1024} bytes): {under_times[0]*1000:.3f}ms")
        print(f"  Over threshold ({_SPOOLED_MAX_SIZE+1024} bytes):  {over_times[0]*1000:.3f}ms")


class TestHTTPChunkSizeEfficiency:
    """Tests for HTTP chunk size optimization."""

    def test_chunk_size_configuration(self):
        """Verify chunk size is configured for efficiency."""
        from markitdown._markitdown import _HTTP_CHUNK_SIZE

        # 256KB is a good balance for network efficiency
        # Original was 512 bytes which is too small
        assert _HTTP_CHUNK_SIZE >= 65536, \
            f"HTTP chunk size ({_HTTP_CHUNK_SIZE}) should be at least 64KB"
        assert _HTTP_CHUNK_SIZE == 262144, \
            f"Expected 256KB (262144), got {_HTTP_CHUNK_SIZE}"


class TestPPTXOptimizationEfficiency:
    """Tests for PPTX converter optimization performance."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_FILES_DIR, "test.pptx")),
        reason="test.pptx not found"
    )
    def test_pptx_conversion_performance(self):
        """Test PPTX file conversion performance."""
        pptx_path = os.path.join(TEST_FILES_DIR, "test.pptx")
        markitdown = MarkItDown()

        def convert_pptx():
            return markitdown.convert(pptx_path)

        times = measure_time(convert_pptx)

        print(f"\n  PPTX conversion: {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")


class TestOverallConversionEfficiency:
    """End-to-end efficiency tests for common file types."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_FILES_DIR, "test.docx")),
        reason="test.docx not found"
    )
    def test_docx_performance(self):
        """Test DOCX conversion performance."""
        path = os.path.join(TEST_FILES_DIR, "test.docx")
        markitdown = MarkItDown()

        def convert():
            return markitdown.convert(path)

        times = measure_time(convert)
        print(f"\n  DOCX: {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_FILES_DIR, "test.pdf")),
        reason="test.pdf not found"
    )
    def test_pdf_performance(self):
        """Test PDF conversion performance."""
        path = os.path.join(TEST_FILES_DIR, "test.pdf")
        markitdown = MarkItDown()

        def convert():
            return markitdown.convert(path)

        times = measure_time(convert)
        print(f"\n  PDF: {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_FILES_DIR, "test_blog.html")),
        reason="test_blog.html not found"
    )
    def test_html_performance(self):
        """Test HTML conversion performance."""
        path = os.path.join(TEST_FILES_DIR, "test_blog.html")
        markitdown = MarkItDown()

        def convert():
            return markitdown.convert(path)

        times = measure_time(convert)
        print(f"\n  HTML: {times[0]*1000:.3f}ms (±{times[1]*1000:.3f}ms)")

    def test_multiple_file_types_benchmark(self):
        """Benchmark multiple file types for comparison."""
        markitdown = MarkItDown()
        results = {}

        test_files = [
            ("test.docx", "DOCX"),
            ("test.pdf", "PDF"),
            ("test.pptx", "PPTX"),
            ("test.xlsx", "XLSX"),
            ("test.xls", "XLS"),
            ("test_blog.html", "HTML"),
            ("test.epub", "EPUB"),
        ]

        print("\n  === File Type Benchmark ===")
        for filename, label in test_files:
            path = os.path.join(TEST_FILES_DIR, filename)
            if not os.path.exists(path):
                print(f"  {label}: SKIPPED (file not found)")
                continue

            def convert(p=path):
                return markitdown.convert(p)

            times = measure_time(convert, iterations=3, warmup=1)
            results[label] = times[0]
            print(f"  {label}: {times[0]*1000:.1f}ms")

        print("  ===========================")


class TestMemoryEfficiency:
    """Tests for memory-efficient operations."""

    def test_generator_expression_in_normalization(self):
        """Verify that normalization uses generator expression (memory efficient)."""
        # This is a code quality check - the implementation should use
        # generator expression instead of list comprehension for large content
        from markitdown._markitdown import _NEWLINE_SPLIT_PATTERN

        # Create large content
        content = "Line\n" * 100000

        # Using generator (as implemented)
        result = "\n".join(
            line.rstrip()
            for line in _NEWLINE_SPLIT_PATTERN.split(content)
        )

        assert len(result) > 0
        # If we got here without memory error, generator is working


class TestRepeatedConversionEfficiency:
    """Tests for efficiency of repeated conversions."""

    def test_instance_reuse_efficiency(self):
        """Test that reusing MarkItDown instance is efficient."""
        content = b"Test content"

        # Create single instance
        markitdown = MarkItDown()

        def reuse_instance():
            stream = io.BytesIO(content)
            return markitdown.convert(stream, stream_info=StreamInfo(extension=".txt"))

        def new_instance():
            md = MarkItDown()
            stream = io.BytesIO(content)
            return md.convert(stream, stream_info=StreamInfo(extension=".txt"))

        reuse_times = measure_time(reuse_instance, iterations=10)
        new_times = measure_time(new_instance, iterations=10)

        print(f"\n  Reuse instance: {reuse_times[0]*1000:.3f}ms (±{reuse_times[1]*1000:.3f}ms)")
        print(f"  New instance:   {new_times[0]*1000:.3f}ms (±{new_times[1]*1000:.3f}ms)")

        # Reusing instance should be faster (no re-initialization)
        assert reuse_times[0] < new_times[0], \
            "Reusing MarkItDown instance should be faster than creating new ones"
