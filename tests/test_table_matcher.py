"""Tests for the semantic matcher."""

from src.table_normalization import NormalizedRow
from src.table_matcher import SemanticMatcher


def dummy_embed(texts):
    return [[float(len(text) or 1)] for text in texts]


def make_row(descriptor, unit=None, currency=None, price=None):
    values = {"description": descriptor}
    if price is not None:
        values["price"] = str(price)
    row = NormalizedRow(
        source="source", row_index=0, values=values, descriptor=descriptor, unit=unit, currency=currency
    )
    return row


def test_matcher_filters_by_unit():
    matcher = SemanticMatcher(embed_fn=dummy_embed)
    base = [make_row("Widget", unit="kg")]
    peers = [make_row("Widget", unit="lb")]
    matches = matcher.match(base, peers)
    assert matches == []


def test_matcher_respects_price_threshold():
    matcher = SemanticMatcher(embed_fn=dummy_embed, price_threshold=0.2)
    base = [make_row("Widget", price=100, unit="kg")]
    peers = [make_row("Widget", price=90, unit="kg"), make_row("Widget", price=50, unit="kg")]
    matches = matcher.match(base, peers)
    assert len(matches) == 1
    assert matches[0].price_delta is not None
    assert matches[0].price_delta < 0.2
