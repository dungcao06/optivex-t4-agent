"""Deterministic history records from frozen pipe tables; context for the model, never a forecast.

A record is one numeric column of one table in a document the entity may cite and that the
retriever selected for it. Row tables (a date or month in the first column) give a series down
the rows; vintage grids (dated columns) give the entity's own reference month across vintages,
and only when exactly one reference month is named by the entity. Every raw cell keeps its exact
source offsets; derived values (last change, median of the last six) are recomputed from those
cells. Units are never inferred: the column header is reported as written. Anything ambiguous,
oversized or dated after the cutoff is omitted rather than guessed.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from claims import Ownership, may_cite

MAX_DOC_CHARS = 200_000
MAX_TABLE_ROWS = 400
MAX_TABLES = 20
MAX_RECORDS = 12
MEDIAN_WINDOW = 6
MISSING = frozenset({"", "--", "-", "—", "n/a", "na", "nan", "."})
_PERIOD = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")
_SEPARATOR = re.compile(r":?-{3,}:?")


@dataclass(frozen=True)
class Cell:
    period: str
    value: float
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class HistoryRecord:
    doc_id: str
    column: str
    label: str            # the entity's reference month for a vintage grid, else ""
    axis: str             # "rows" or "vintages"
    cells: tuple[Cell, ...]

    @property
    def last(self) -> float:
        return self.cells[-1].value

    @property
    def last_change(self) -> float | None:
        return self.cells[-1].value - self.cells[-2].value if len(self.cells) > 1 else None

    @property
    def median_window(self) -> int:
        return min(MEDIAN_WINDOW, len(self.cells))

    @property
    def median(self) -> float:
        return statistics.median(c.value for c in self.cells[-self.median_window:])


def _cells(line: str, offset: int) -> list[tuple[str, int, int]]:
    """Every stripped cell of one pipe row, edges included, with absolute offsets."""
    body = line.rstrip("\r\n")
    parts: list[tuple[str, int, int]] = []
    start = 0
    for index in range(len(body) + 1):
        if index == len(body) or body[index] == "|":
            raw = body[start:index]
            left = len(raw) - len(raw.lstrip())
            text = raw.strip()
            parts.append((text, offset + start + left, offset + start + left + len(text)))
            start = index + 1
    return parts


def _trim(row: list, lead: bool, tail: bool) -> list:
    """Drop the outer-pipe edge cells, decided once by the header's pipe style."""
    if lead and row and row[0][0] == "":
        row = row[1:]
    if tail and row and row[-1][0] == "":
        row = row[:-1]
    return row


def _tables(text: str):
    """Yield (header, rows) for bounded pipe tables, each cell with offsets."""
    offset, block, lines, found = 0, [], [], 0
    for line in text.splitlines(keepends=True) + [""]:
        if "|" in line and "\\|" not in line:
            block.append(_cells(line, offset))
            lines.append(line)
        else:
            if len(block) >= 2 and len(block) - 1 <= MAX_TABLE_ROWS:
                head = lines[0].strip()
                lead, tail = head.startswith("|"), head.endswith("|")
                header = _trim(block[0], lead, tail)
                rows = [_trim(r, lead, tail) for r in block[1:]]
                rows = [r for r in rows if not all(_SEPARATOR.fullmatch(c[0]) for c in r)]
                if len(header) >= 2 and rows and all(len(r) == len(header) for r in rows):
                    found += 1
                    if found > MAX_TABLES:
                        return
                    yield header, rows
            block, lines = [], []
        offset += len(line)


def _number(text: str) -> float | None:
    return float(text.replace(",", "")) if _NUMBER.fullmatch(text) else None


def _row_records(doc_id: str, header, rows, limit: str) -> list[HistoryRecord]:
    if not all(_PERIOD.fullmatch(r[0][0]) for r in rows):
        return []
    records = []
    for column in range(1, len(header)):
        cells, numeric = [], True
        for row in rows:
            text, start, end = row[column]
            if text.lower() in MISSING:
                continue
            value = _number(text)
            if value is None:
                numeric = False
                break
            period = row[0][0]
            if period <= limit[: len(period)]:
                cells.append(Cell(period, value, text, start, end))
        cells.sort(key=lambda cell: cell.period)
        if numeric and cells:
            records.append(HistoryRecord(doc_id, header[column][0], "", "rows", tuple(cells)))
    return records


def _vintage_records(doc_id: str, header, rows, limit: str, entity: dict) -> list[HistoryRecord]:
    labels = {row[0][0] for row in rows}
    named = {v for v in entity.values() if isinstance(v, str) and v in labels}
    if len(named) != 1:
        return []
    label = named.pop()
    matching = [row for row in rows if row[0][0] == label]
    if len(matching) != 1:
        return []
    cells = []
    for column in range(1, len(header)):
        dates = _DATE.findall(header[column][0])
        text, start, end = matching[0][column]
        value = None if text.lower() in MISSING else _number(text)
        if len(dates) == 1 and dates[0] <= limit and value is not None:
            cells.append(Cell(dates[0], value, text, start, end))
    cells.sort(key=lambda cell: cell.period)
    return [HistoryRecord(doc_id, "value", label, "vintages", tuple(cells))] if cells else []


def _relevance(record: HistoryRecord, task: dict, entity: dict) -> int:
    """Vintage rows first, then the entity's own series in a shared table, then target words."""
    column = record.column.lower()
    words = set(_WORD.findall(column))
    wanted = set(_WORD.findall(str(task.get("target", {}).get("name", "")).lower()))
    for key in ("unit", "units"):
        wanted |= set(_WORD.findall(str(entity.get(key, "")).lower()))
    named = {v.strip().lower() for k, v in entity.items() if k != "entity_id" and isinstance(v, str)}
    own = 5 if column in named else 0
    name_overlap = 2 * len(words & set(_WORD.findall(str(entity.get("name", "")).lower())))
    return (10 if record.axis == "vintages" else 0) + own + name_overlap + len(words & wanted)


def build_history(task: dict, entity: dict, corpus, owners: Ownership | None,
                  contexts: list) -> list[HistoryRecord]:
    """Structured records from selected own/shared pre-cutoff tables, most relevant first."""
    cutoff = task.get("cutoff_date")
    entity_id = entity.get("entity_id")
    if not isinstance(cutoff, str) or not _DATE.fullmatch(cutoff) or not isinstance(entity_id, str):
        return []
    selected = list(dict.fromkeys(getattr(c, "doc_id", None) for c in contexts or []))
    records: list[HistoryRecord] = []
    for doc_id in selected:
        if not isinstance(doc_id, str) or not may_cite(owners, doc_id, entity_id):
            continue
        text = corpus.doc_texts.get(doc_id)
        doc_date = corpus.doc_dates.get(doc_id)
        if (not isinstance(text, str) or len(text) > MAX_DOC_CHARS or not isinstance(doc_date, str)
                or not _DATE.fullmatch(doc_date) or doc_date > cutoff):
            continue
        limit = min(cutoff, doc_date)  # Rows dated after the document are schedules, not data.
        for header, rows in _tables(text):
            dated = [h for h in header[1:] if len(_DATE.findall(h[0])) == 1]
            if len(header) > 2 and len(dated) == len(header) - 1:
                records += _vintage_records(doc_id, header, rows, limit, entity)
            else:
                records += _row_records(doc_id, header, rows, limit)
    order = {doc_id: i for i, doc_id in enumerate(selected)}
    ranked = sorted(enumerate(records),
                    key=lambda item: (-_relevance(item[1], task, entity), order[item[1].doc_id], item[0]))
    return [record for _, record in ranked][:MAX_RECORDS]


def _fmt(value: float) -> str:
    """Plain decimals, never exponents, which models misread."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if 1e-4 <= abs(value) < 1e5:
        return format(value, ".6g")
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_history(records: list[HistoryRecord], limit: int = 1200) -> str:
    """A complete context block within `limit` characters; whole records are omitted, never cut."""
    head = ("COMPUTED FROM FROZEN TABLES (deterministic, pre-cutoff own/shared documents; "
            "context, not a forecast):\n")
    lines = []
    for record in records:
        scope = f" for {record.label}" if record.label else ""
        line = (f"- {record.doc_id} | {record.column}{scope} | {record.cells[0].period}.."
                f"{record.cells[-1].period}, n={len(record.cells)}: last {record.cells[-1].raw}"
                f" ({record.cells[-1].period})")
        if record.last_change is not None:
            line += f"; change vs previous {_fmt(record.last_change)}"
        line += f"; median of last {record.median_window} = {_fmt(record.median)}\n"
        if len(head) + sum(map(len, lines)) + len(line) <= limit:
            lines.append(line)
    return head + "".join(lines) if lines else ""
