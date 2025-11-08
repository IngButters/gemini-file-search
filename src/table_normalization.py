"""Normalization utilities for tabular comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from .table_ingestion import TableRow


DEFAULT_UNIT_ALIASES: Dict[str, Set[str]] = {
    "kg": {"kg", "kilogram", "kilograms"},
    "lb": {"lb", "lbs", "pound", "pounds"},
    "item": {"ea", "each", "item", "items"},
}

DEFAULT_CURRENCY_ALIASES: Dict[str, Set[str]] = {
    "usd": {"$", "usd", "us$", "dollar", "dollars"},
    "eur": {"€", "eur", "euro", "euros"},
}


@dataclass
class NormalizedRow:
    """Normalized representation of a table row."""

    source: str
    row_index: int
    values: Dict[str, str]
    unit: Optional[str] = None
    currency: Optional[str] = None
    descriptor: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    def get_numeric(self, key: str) -> Optional[float]:
        """Extract a numeric value from the row if possible."""

        raw = self.values.get(key)
        if raw is None:
            return None
        try:
            digits = raw.replace(",", "").replace("$", "").replace("€", "")
            return float(digits)
        except ValueError:
            return None


def clean_text(value: str) -> str:
    """Perform basic text cleanup."""

    if value is None:
        return ""
    return " ".join(value.strip().split())


def _canonical_lookup(value: str, mapping: Dict[str, Set[str]]) -> Optional[str]:
    normalized = value.lower()
    for canonical, aliases in mapping.items():
        if normalized == canonical:
            return canonical
        if normalized in aliases:
            return canonical
    return None


def reconcile_unit(value: str, mapping: Optional[Dict[str, Set[str]]] = None) -> Optional[str]:
    """Determine a canonical unit string from raw text."""

    if not value:
        return None
    mapping = mapping or DEFAULT_UNIT_ALIASES
    return _canonical_lookup(value, mapping)


def reconcile_currency(value: str, mapping: Optional[Dict[str, Set[str]]] = None) -> Optional[str]:
    """Determine a canonical currency code from raw text."""

    if not value:
        return None
    mapping = mapping or DEFAULT_CURRENCY_ALIASES
    return _canonical_lookup(value, mapping)


def construct_descriptor(values: Dict[str, str], max_keys: int = 3) -> str:
    """Build a semantic descriptor for a row using key columns."""

    parts: List[str] = []
    for key in list(values.keys())[:max_keys]:
        value = values.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


def normalize_rows(
    rows: Iterable[TableRow],
    unit_column: Optional[str] = None,
    currency_column: Optional[str] = None,
    descriptor_keys: Optional[List[str]] = None,
    unit_aliases: Optional[Dict[str, Set[str]]] = None,
    currency_aliases: Optional[Dict[str, Set[str]]] = None,
) -> List[NormalizedRow]:
    """Normalize raw :class:`TableRow` instances into :class:`NormalizedRow`."""

    normalized_rows: List[NormalizedRow] = []
    for row in rows:
        cleaned_values = {key: clean_text(value) for key, value in row.values.items()}
        unit_value = cleaned_values.get(unit_column) if unit_column else None
        currency_value = cleaned_values.get(currency_column) if currency_column else None
        descriptor_source = descriptor_keys or list(cleaned_values.keys())
        descriptor = construct_descriptor({key: cleaned_values.get(key, "") for key in descriptor_source})
        normalized_rows.append(
            NormalizedRow(
                source=row.source,
                row_index=row.row_index,
                values=cleaned_values,
                unit=reconcile_unit(unit_value, unit_aliases) if unit_value else None,
                currency=reconcile_currency(currency_value, currency_aliases)
                if currency_value
                else None,
                descriptor=descriptor,
                metadata={key: str(value) for key, value in row.metadata.items()},
            )
        )
    return normalized_rows


__all__ = [
    "NormalizedRow",
    "clean_text",
    "reconcile_unit",
    "reconcile_currency",
    "construct_descriptor",
    "normalize_rows",
]
