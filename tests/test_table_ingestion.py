"""Tests for table ingestion utilities."""

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

from src.table_ingestion import ingest_table, load_excel_tables, load_pdf_tables


class FakeRow:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class FakeDataFrame:
    def __init__(self):
        self.columns = ["product", "price", "unit"]
        self._rows = [
            {"product": "Widget", "price": 9.99, "unit": "kg"},
            {"product": "Gadget", "price": 12.5, "unit": "lb"},
        ]

    def iterrows(self):
        for index, row in enumerate(self._rows):
            yield index, FakeRow(row)


@pytest.fixture
def fake_pandas(monkeypatch):
    module = SimpleNamespace()

    def fake_read_excel(path, sheet_name=None):
        return FakeDataFrame()

    module.read_excel = fake_read_excel
    monkeypatch.setitem(sys.modules, "pandas", module)
    return module


def test_load_excel_tables_creates_standard_rows(tmp_path, fake_pandas):
    workbook = tmp_path / "sample.xlsx"
    workbook.write_bytes(b"dummy")

    rows = load_excel_tables(workbook)
    assert len(rows) == 2
    assert rows[0].values["product"] == "Widget"
    assert rows[0].metadata["sheet"] == "Sheet1"


def test_load_pdf_requires_dependency(tmp_path, monkeypatch):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"pdf")

    if 'pdfplumber' in sys.modules:
        monkeypatch.delitem(sys.modules, 'pdfplumber', raising=False)
    monkeypatch.setitem(sys.modules, 'pdfplumber', None)

    with pytest.raises(RuntimeError):
        load_pdf_tables(pdf_file)


def test_load_pdf_tables_with_stub(tmp_path, monkeypatch):
    pdf_file = tmp_path / "tables.pdf"
    pdf_file.write_bytes(b"pdf")

    class FakePage:
        def extract_tables(self):
            return [[
                ["Product", "Price"],
                ["Widget", "10"],
            ]]

    class FakePDF:
        def __init__(self, *args, **kwargs):
            self.pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    module = SimpleNamespace(open=lambda path: FakePDF())
    monkeypatch.setitem(sys.modules, "pdfplumber", module)

    rows = load_pdf_tables(pdf_file)
    assert len(rows) == 1
    assert rows[0].values["product"] == "Widget"
    assert rows[0].metadata["page"] == 1


def test_ingest_dispatches_by_extension(tmp_path, fake_pandas):
    workbook = tmp_path / "dispatch.xlsx"
    workbook.write_bytes(b"dummy")
    rows = ingest_table(workbook)
    assert len(rows) == 2

    with pytest.raises(ValueError):
        ingest_table(tmp_path / "unknown.txt")
