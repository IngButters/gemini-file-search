"""Comparison orchestrator that wires ingestion, normalization, and matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .table_ingestion import ingest_table, TableRow
from .table_normalization import NormalizedRow, normalize_rows
from .table_matcher import MatchCandidate, SemanticMatcher


@dataclass
class ComparisonReport:
    """Structured output for a single base/peer comparison."""

    base_file: Path
    peer_file: Path
    matches: List[MatchCandidate]
    unmatched_base: List[NormalizedRow]
    unmatched_peer: List[NormalizedRow]

    def to_records(self) -> List[Dict[str, str]]:
        """Represent matches as serializable records."""

        records: List[Dict[str, str]] = []
        for candidate in self.matches:
            records.append(
                {
                    "base_source": candidate.base.source,
                    "base_row": str(candidate.base.row_index),
                    "peer_source": candidate.peer.source,
                    "peer_row": str(candidate.peer.row_index),
                    "score": f"{candidate.score:.3f}",
                    "descriptor_base": candidate.base.descriptor,
                    "descriptor_peer": candidate.peer.descriptor,
                    "unit": candidate.base.unit or candidate.peer.unit or "",
                    "currency": candidate.base.currency or candidate.peer.currency or "",
                    "price_delta": "" if candidate.price_delta is None else f"{candidate.price_delta:.3f}",
                }
            )
        return records

    def to_markdown(self) -> str:
        records = self.to_records()
        if not records:
            return f"### {self.base_file.name} vs {self.peer_file.name}\n\n_No matches found._"
        headers = list(records[0].keys())
        lines = [f"### {self.base_file.name} vs {self.peer_file.name}", ""]
        header_line = " | ".join(headers)
        separator = " | ".join(["---"] * len(headers))
        lines.extend([header_line, separator])
        for record in records:
            lines.append(" | ".join(record.get(key, "") for key in headers))
        return "\n".join(lines)

    def to_csv(self) -> str:
        records = self.to_records()
        if not records:
            return ""
        headers = list(records[0].keys())
        lines = [",".join(headers)]
        for record in records:
            lines.append(
                ",".join(record.get(key, "").replace(",", " ") for key in headers)
            )
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Aggregated comparison output for multiple peer files."""

    base_file: Path
    reports: List[ComparisonReport] = field(default_factory=list)

    def to_markdown(self) -> str:
        if not self.reports:
            return ""
        sections = [report.to_markdown() for report in self.reports]
        return "\n\n".join(sections)

    def to_csv(self) -> str:
        lines: List[str] = []
        for report in self.reports:
            csv_block = report.to_csv()
            if not csv_block:
                continue
            header = f"# {report.base_file.name} vs {report.peer_file.name}"
            lines.extend([header, csv_block, ""])
        return "\n".join(lines).strip()


class TableComparisonOrchestrator:
    """High level orchestration for table comparison workflows."""

    def __init__(
        self,
        embed_fn: Callable[[Sequence[str]], List[List[float]]],
        unit_column: Optional[str] = None,
        currency_column: Optional[str] = None,
        descriptor_keys: Optional[List[str]] = None,
        matcher_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.unit_column = unit_column
        self.currency_column = currency_column
        self.descriptor_keys = descriptor_keys
        self.matcher_kwargs = matcher_kwargs or {}

    def _normalize(self, rows: Iterable[TableRow]) -> List[NormalizedRow]:
        return normalize_rows(
            rows,
            unit_column=self.unit_column,
            currency_column=self.currency_column,
            descriptor_keys=self.descriptor_keys,
        )

    def _matcher(self) -> SemanticMatcher:
        return SemanticMatcher(embed_fn=self.embed_fn, **self.matcher_kwargs)

    def run(self, base_file: Path, peer_files: Iterable[Path]) -> ComparisonResult:
        base_file = Path(base_file)
        peer_paths = [Path(path) for path in peer_files]

        base_rows = self._normalize(ingest_table(base_file))
        matcher = self._matcher()

        reports: List[ComparisonReport] = []
        for peer_path in peer_paths:
            peer_rows = self._normalize(ingest_table(peer_path))
            matches = matcher.match(base_rows, peer_rows)
            matched_base_ids = {(candidate.base.source, candidate.base.row_index) for candidate in matches}
            matched_peer_ids = {(candidate.peer.source, candidate.peer.row_index) for candidate in matches}
            unmatched_base = [row for row in base_rows if (row.source, row.row_index) not in matched_base_ids]
            unmatched_peer = [row for row in peer_rows if (row.source, row.row_index) not in matched_peer_ids]
            reports.append(
                ComparisonReport(
                    base_file=base_file,
                    peer_file=peer_path,
                    matches=matches,
                    unmatched_base=unmatched_base,
                    unmatched_peer=unmatched_peer,
                )
            )

        return ComparisonResult(base_file=base_file, reports=reports)


__all__ = [
    "ComparisonReport",
    "ComparisonResult",
    "TableComparisonOrchestrator",
]
