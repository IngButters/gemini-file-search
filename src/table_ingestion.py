"""Utilities for ingesting tabular data from PDF and Excel files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Any


@dataclass
class TableRow:
    """Standardized representation of a row extracted from a table."""

    source: str
    row_index: int
    values: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _stringify(value: Any) -> str:
    """Convert a cell value to a consistent string representation."""

    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return ("%f" % value).rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    return str(value).strip()


def _normalize_header(header: Any, index: int) -> str:
    """Create normalized header names when missing or duplicated."""

    base = _stringify(header).lower().replace(" ", "_")
    base = base or f"column_{index}"
    return base


def _create_rows(
    headers: Sequence[str],
    records: Iterable[Sequence[Any]],
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    starting_index: int = 0,
) -> List[TableRow]:
    """Convert raw records into :class:`TableRow` objects."""

    rows: List[TableRow] = []
    metadata = metadata or {}
    for offset, record in enumerate(records):
        values = {}
        for idx, header in enumerate(headers):
            cell = record[idx] if idx < len(record) else ""
            values[header] = _stringify(cell)
        rows.append(
            TableRow(
                source=source,
                row_index=starting_index + offset,
                values=values,
                metadata=dict(metadata),
            )
        )
    return rows


def load_pdf_tables(file_path: Path) -> List[TableRow]:
    """Load tables from a PDF file using ``pdfplumber`` when available.

    Args:
        file_path: Path to a PDF document.

    Returns:
        A list of :class:`TableRow` instances extracted from all tables in the
        document.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ``pdfplumber`` is not installed.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in tests via monkeypatch
        raise RuntimeError(
            "pdfplumber is required for PDF ingestion. Install pdfplumber or "
            "provide a custom ingestion strategy."
        ) from exc

    rows: List[TableRow] = []
    with pdfplumber.open(str(file_path)) as pdf:  # pragma: no cover - patched in tests
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            for table_index, table in enumerate(tables):
                if not table:
                    continue
                headers = [_normalize_header(header, idx) for idx, header in enumerate(table[0])]
                records = table[1:]
                rows.extend(
                    _create_rows(
                        headers,
                        records,
                        source=str(file_path),
                        metadata={"page": page_number, "table": table_index},
                    )
                )
    return rows


def load_excel_tables(file_path: Path, sheet_name: Optional[str] = None) -> List[TableRow]:
    """Load tables from an Excel workbook using :func:`pandas.read_excel`.

    Args:
        file_path: Path to the Excel workbook.
        sheet_name: Optional sheet name to limit ingestion.

    Returns:
        A list of :class:`TableRow` objects extracted from the workbook.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If pandas is not installed.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in tests via monkeypatch
        raise RuntimeError(
            "pandas is required for Excel ingestion. Install pandas or provide "
            "a custom ingestion strategy."
        ) from exc

    dataframe = pd.read_excel(file_path, sheet_name=sheet_name)
    frames = dataframe.items() if isinstance(dataframe, dict) else [(sheet_name or "Sheet1", dataframe)]

    rows: List[TableRow] = []
    for sheet, frame in frames:
        headers = [_normalize_header(col, idx) for idx, col in enumerate(frame.columns)]
        for row_index, row in frame.iterrows():
            record = [row[col] for col in frame.columns]
            rows.extend(
                _create_rows(
                    headers,
                    [record],
                    source=str(file_path),
                    metadata={"sheet": sheet},
                    starting_index=row_index,
                )
            )
    return rows


def ingest_table(file_path: Path) -> List[TableRow]:
    """Ingest a table file, dispatching based on file extension."""

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return load_excel_tables(file_path)
    if suffix == ".pdf":
        return load_pdf_tables(file_path)
    raise ValueError(f"Unsupported table format: {file_path}")


__all__ = ["TableRow", "ingest_table", "load_excel_tables", "load_pdf_tables"]
