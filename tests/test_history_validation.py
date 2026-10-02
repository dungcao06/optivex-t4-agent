"""Reject ambiguous chronology and nonfinite descriptive statistics."""
from types import SimpleNamespace

import pytest

from test_history import OWN, TASK, Corpus
from history import build_history, format_history


def records(text, *, cutoff="2024-09-30", doc_date="2024-09-30", **entity):
    return build_history(dict(TASK, cutoff_date=cutoff), {"entity_id": "E", **entity},
                         Corpus({"own": text}, date=doc_date), OWN,
                         [SimpleNamespace(doc_id="own")])


@pytest.mark.parametrize("dates", [
    ["2024-02-30", "2024-03-01"], ["2024-13", "2024-09"],
    ["2024-01-01", "2024-01-01"], ["2024-01", "2024-02-01"],
])
def test_invalid_duplicate_or_mixed_observation_periods_are_omitted(dates):
    text = f"date | value\n{dates[0]} | 10\n{dates[1]} | 20\n"
    assert records(text) == []


@pytest.mark.parametrize("kwargs", [{"cutoff": "2024-02-30"}, {"doc_date": "2024-02-30"}])
def test_invalid_calendar_metadata_does_not_admit_a_table(kwargs):
    assert records("date | value\n2024-01-01 | 10\n", **kwargs) == []


@pytest.mark.parametrize("first,second", [
    ("9" * 400, "1"), ("1" + "0" * 308, "-1" + "0" * 308),
    ("1" + "0" * 308, "1" + "0" * 308),
])
def test_nonfinite_cells_or_derived_statistics_are_omitted(first, second):
    assert records(f"date | value\n2024-01-01 | {first}\n2024-01-02 | {second}\n") == []


def test_single_vintage_does_not_compare_different_reference_months():
    found = records("reference_month | as_of_2024-09-26\n2024-07 | 100\n2024-08 | 200\n",
                    ref_month="2024-08")
    assert len(found) == 1
    assert found[0].axis == "vintages"
    assert [c.value for c in found[0].cells] == [200]
    assert found[0].last_change is None


@pytest.mark.parametrize("headers", [
    "as_of_2024-02-30 | as_of_2024-03-01",
    "as_of_2024-03-01 | as_of_2024-03-01",
])
def test_invalid_or_duplicate_vintage_dates_are_not_a_revision_series(headers):
    assert records(f"reference_month | {headers}\n2024-01 | 100 | 200\n", ref_month="2024-01") == []


def test_future_bad_values_cannot_erase_eligible_history():
    found = records("date | value\n2024-01-01 | 10\n2024-10-01 | unknown\n")
    assert len(found) == 1 and found[0].last == 10


def test_small_nonzero_changes_are_not_printed_as_zero():
    found = records("date | value\n2024-01-01 | 0.00000001\n2024-01-02 | 0.00000002\n")
    block = format_history(found)
    assert "0.000000015" in block
    assert "0.00000001" in block


@pytest.mark.parametrize("headers", [
    "as_of_2024-03-01 | other", "2024-03-01/2024-04-01",
])
def test_ambiguous_vintage_headers_do_not_fall_back_to_row_history(headers):
    values = " | ".join("100" for _ in headers.split("|"))
    assert records(f"reference_month | {headers}\n2024-01 | {values}\n", ref_month="2024-01") == []


def test_invalid_eligible_vintage_value_does_not_create_a_partial_revision_series():
    assert records("reference_month | as_of_2024-03-01 | as_of_2024-04-01\n"
                   "2024-01 | invalid | 200\n", ref_month="2024-01") == []


def test_formatted_history_has_original_offsets_and_citation_and_unit_boundaries():
    text = "date | value\n2024-01-01 | 10\n2024-01-02 | 20\n"
    found = records(text)
    block = format_history(found)
    assert f"own[{text.index('10')}:{text.rindex('20') + 2}]" in block
    assert "Not citable quotes" in block and "claims must quote original excerpts" in block
    assert "not necessarily target units" in block and "12 significant digits" in block
