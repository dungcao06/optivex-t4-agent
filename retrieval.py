"""Bounded, entity-aware retrieval over the supplied frozen corpus only.

Passages are slices of the scorer's original document text, so their offsets
remain valid even for multi-megabyte SEC filings and Unicode punctuation.
"""
from __future__ import annotations

import heapq
import json
import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus

_MAX_PASSAGE = 2200
_MIN_BREAK = 1600
_OVERLAP = 200
_EVIDENCE_BUDGET = 16000
_TOKEN = re.compile(r"[a-z0-9]+")
_NUMERIC_IDENTITY = re.compile(
    r"(?<![\w.+\-\u2010-\u2015\u2212])([0-9]+(?:\.[0-9]+)?)"
    r"[^\S\r\n\v\f\x1c-\x1e\x85\u2028\u2029]*[-\u2010-\u2015\u2212]"
    r"[^\S\r\n\v\f\x1c-\x1e\x85\u2028\u2029]*([a-z]{2,})\b"
)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_EPS = re.compile(r"\beps\b|earnings(?:/\(loss\))? per (?:common )?share|per diluted (?:common )?share", re.I)
_STOP = frozenset(
    "a an and are as at be been before by can do each for from given in into is it "
    "its of on or per than that the their these they this through to using was were "
    "will with within you your only frozen corpus evidence predict prediction "
    "provide report support claim claims citations passage passages cutoff post "
    "point forecast interval percent table below above absent design "
    "sec edgar filings filing available most recent company companies quarter "
    "reported released published after not use expressed relative given "
    "first months ended month known following statement support reasoning".split()
)


@dataclass(frozen=True)
class _IndexedEvidence(IndexedCorpus):
    doc_metadata: dict[str, dict] = field(default_factory=dict)


def _valid_date(value: object) -> date | None:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _compact_tables(text: str) -> list[tuple[int, int]]:
    """Recognize bounded pipe tables without normalizing the source text."""
    lines = text.splitlines(keepends=True)
    tables: list[tuple[int, int]] = []
    offset = 0
    index = 0
    while index < len(lines):
        if "|" not in lines[index]:
            offset += len(lines[index])
            index += 1
            continue
        first, start = index, offset
        rows: list[list[str]] = []
        valid = True
        while index < len(lines) and "|" in lines[index]:
            line = lines[index]
            offset += len(line)
            index += 1
            if offset - start > _MAX_PASSAGE or "\\|" in line:
                valid = False
                continue
            row = line.strip()
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            cells = [cell.strip() for cell in row.split("|")]
            if len(cells) < 2 or not all(cells) or (rows and len(cells) != len(rows[0])):
                valid = False
            rows.append(cells)
        if not valid or len(rows) < 2 or not any(c.isalpha() for c in "".join(rows[0])):
            continue
        separators = [all(re.fullmatch(r":?-{3,}:?", cell) for cell in row) for row in rows]
        has_separator = separators[1]
        data_start = 2 if has_separator else 1
        if len(rows) <= data_start or any(separators[data_start:]):
            continue
        if not has_separator and any(not any(c.isdigit() for c in "".join(row))
                                     for row in rows[data_start:]):
            continue
        if first:
            caption = lines[first - 1].strip()
            caption_start = start - len(lines[first - 1])
            if (caption and len(caption) <= 200 and "|" not in caption
                    and (caption.endswith(":") or re.search(r"\bunits?\b", caption, re.I))
                    and offset - caption_start <= _MAX_PASSAGE):
                start = caption_start
        tables.append((start, offset))
    return tables


def _passages(text: str) -> Iterator[tuple[int, int]]:
    """Yield original-text windows, keeping recognized compact tables intact."""
    tables = _compact_tables(text)
    table_starts = [left for left, _ in tables]
    start = 0
    while start < len(text):
        end = min(start + _MAX_PASSAGE, len(text))
        if end < len(text):
            lower = start + _MIN_BREAK
            paragraph = text.rfind("\n", lower, end)
            space = text.rfind(" ", lower, end)
            boundary = paragraph if paragraph >= lower else space
            if boundary >= lower:
                end = boundary + 1
        cut_before_table = False
        table_index = bisect_right(table_starts, end) - 1
        if table_index >= 0:
            left, right = tables[table_index]
            if left < end < right:
                if right - start <= _MAX_PASSAGE:
                    end = right
                else:
                    end = left
                    cut_before_table = True
        yield start, end
        if end == len(text):
            break
        start = end if cut_before_table else max(start + 1, end - _OVERLAP)
        table_index = bisect_right(table_starts, start) - 1
        if table_index >= 0:
            left, right = tables[table_index]
            if left < start < right:
                start = right


def build_index(corpus_dir: str | Path) -> IndexedCorpus:
    """Read corpus documents without fetching data or rewriting citation text."""
    root = Path(corpus_dir)
    chunks: list[Chunk] = []
    doc_texts: dict[str, str] = {}
    doc_dates: dict[str, str | None] = {}
    metadata: dict[str, dict] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json" or path.is_symlink():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id", path.stem)
        if not isinstance(doc_id, str) or not doc_id:
            continue
        if doc_id in doc_texts:
            raise ValueError(f"duplicate corpus document id: {doc_id}")
        if isinstance(doc.get("text"), str):
            text = doc["text"]
        else:
            text = " ".join(
                span.get("text", "") for span in doc.get("spans", [])
                if isinstance(span, dict) and isinstance(span.get("text", ""), str)
            )
        raw_date = doc.get("doc_date")
        doc_date = raw_date if isinstance(raw_date, str) else None
        doc_texts[doc_id] = text
        doc_dates[doc_id] = doc_date
        metadata[doc_id] = {
            key: doc[key] for key in ("cik", "ticker", "title", "source", "series_id")
            if key in doc
        }
        metadata[doc_id]["relative_path"] = path.relative_to(root).as_posix()
        for start, end in _passages(text):
            if text[start:end].strip():
                chunks.append(Chunk(doc_id, doc_date, start, end, text[start:end]))
    return _IndexedEvidence(chunks, doc_texts, doc_dates, metadata)


def _tokens(value: object) -> list[str]:
    text = str(value).lower()
    tokens = [word for word in _TOKEN.findall(text)
              if len(word) > 1 and word not in _STOP]
    # Add standalone unsigned number-word identities (2-Year vs 10-Year), with
    # two-letter-or-longer words and horizontal spacing around an explicit hyphen.
    for match in _NUMERIC_IDENTITY.finditer(text):
        previous = match.start() - 1
        while (previous >= 0 and text[previous].isspace()
               and text[previous] not in "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
            previous -= 1
        # A separated sign or range prefix is still ambiguous.
        if previous >= 0 and text[previous] in "+-\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
            continue
        if previous >= 0 and text[previous] in ".,":
            previous -= 1
            while (previous >= 0 and text[previous].isspace()
                   and text[previous] not in "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
                previous -= 1
            # Preserve prose/list boundaries, but not decimal or grouped-number tails.
            if (previous < 0 or text[previous].isdigit()
                    or text[previous] in ".,+-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"):
                continue
        tokens.append(match[1] + match[2])
    return tokens


def _cik(value: object) -> str | None:
    text = str(value).strip()
    return str(int(text)) if text.isdigit() else None


def _eps_baseline_score(text: str) -> int:
    """Recognize absolute EPS tables/prose, not contribution deltas or contents."""
    text = re.sub(r"\s+", " ", text.replace("\u200b", " ")).lower()
    best = 0
    for match in _EPS.finditer(text):
        local = text[max(0, match.start() - 90):match.end() + 280]
        if not re.search(r"\d+\.\d+", local):
            continue
        if re.search(r"non.gaap|adjusted|impacted|impact of|increased due", local):
            continue
        diluted = bool(re.search(r"dilut", local))
        table = diluted and "basic" in local
        total = diluted and bool(re.search(r"\btotal\b|consolidated earnings", local))
        income = "net income" in local and "per diluted" in local
        direct = ("diluted" in text[max(0, match.start() - 30):match.start()]
                  and bool(re.match(r"\s*(?:(?:was|were|of)\s+)?\$\s*\(?-?\d+\.\d+",
                                    text[match.end():])))
        if table or total or income or direct:
            best = max(best, 1 + 2 * table + 3 * total + income)
    return best


class EvidenceIndex:
    """BM25 retrieval with issuer affinity, source diversity and a hard size cap."""

    def __init__(self, corpus: IndexedCorpus, cutoff_date: str) -> None:
        cutoff = _valid_date(cutoff_date)
        if cutoff is None:
            raise ValueError("task cutoff_date is not an ISO calendar date")
        self.corpus = corpus
        self.metadata = getattr(corpus, "doc_metadata", {})
        self.chunks = [
            chunk for chunk in corpus.chunks
            if (parsed := _valid_date(chunk.doc_date)) is not None and parsed <= cutoff
        ]
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._eps_baselines: list[tuple[int, int]] = []
        for index, chunk in enumerate(self.chunks):
            if baseline_score := _eps_baseline_score(chunk.text):
                self._eps_baselines.append((baseline_score, index))
            counts = Counter(_tokens(chunk.text))
            self._lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self._postings[token].append((index, count))
        self._avg_len = max(1.0, sum(self._lengths) / max(1, len(self.chunks)))

    def _affinities(self, entity: dict) -> tuple[dict[str, float], set[str], set[str]]:
        """Prefer a declared issuer/subtree; retain unowned shared macro sources."""
        doc_ids = {chunk.doc_id for chunk in self.chunks}
        wanted_cik = _cik(entity.get("cik", ""))
        wanted_tickers = {str(entity.get(key, "")).casefold() for key in ("ticker", "entity_id")}
        wanted_tickers.discard("")
        matches: set[str] = set()
        other_issuers: set[str] = set()
        for doc_id in doc_ids:
            meta = self.metadata.get(doc_id, {})
            found_cik = _cik(meta.get("cik", ""))
            if found_cik is None:
                edgar = re.search(r"(?:^|_)EDGAR_(\d+)(?:_|$)", doc_id, re.I)
                if edgar:
                    found_cik = _cik(edgar.group(1))
            ticker = str(meta.get("ticker", "")).casefold()
            if wanted_cik and found_cik:
                (matches if wanted_cik == found_cik else other_issuers).add(doc_id)
            elif ticker:
                (matches if ticker in wanted_tickers else other_issuers).add(doc_id)

        # A shared corpus/ reference imposes no filter. More specific references
        # are used only when they actually resolve to supplied corpus documents.
        raw_refs = entity.get("corpus_ref", [])
        refs = raw_refs if isinstance(raw_refs, list) else [raw_refs]
        prefixes: list[str] = []
        for raw in refs:
            if not isinstance(raw, str):
                continue
            ref = raw.replace("\\", "/").removeprefix("./").removeprefix("corpus/").strip("/")
            if ref and ref != "corpus" and ".." not in ref.split("/"):
                prefixes.append(ref)
        ref_matches: set[str] = set()
        for doc_id in doc_ids:
            relative = self.metadata.get(doc_id, {}).get("relative_path", "")
            if any(relative == prefix or relative.startswith(prefix + "/") or doc_id == prefix
                   for prefix in prefixes):
                ref_matches.add(doc_id)
        excluded = other_issuers if matches else set()
        if ref_matches:
            excluded = excluded | {
                doc_id for doc_id in doc_ids - ref_matches
                if "/" in self.metadata.get(doc_id, {}).get("relative_path", "")
            }
        affinities = {doc_id: 1.0 for doc_id in doc_ids}
        for doc_id in matches | ref_matches:
            affinities[doc_id] = 3.0
        return affinities, excluded, matches

    def retrieve(self, task: dict, entity: dict, top_k: int = 8) -> list[Chunk]:
        if top_k <= 0 or not self.chunks:
            return []
        weights: dict[str, float] = {}
        affinities, excluded, issuer_matches = self._affinities(entity)

        def add(value: object, weight: float) -> None:
            for token in set(_tokens(value)):
                if not token.isdigit():
                    weights[token] = weights.get(token, 0.0) + weight

        add(task.get("prompt", ""), 1.0)
        target = task.get("target", {})
        add(target.get("name", "") if isinstance(target, dict) else "", 3.0)
        for key, value in entity.items():
            if key == "corpus_ref" or not isinstance(value, (str, int, float)):
                continue
            if issuer_matches and key in {
                "name", "entity_id", "ticker", "cik", "sector", "industry", "currency",
                "quarter_reported", "prior_year_quarter", "expected_report_date",
            }:
                # Attribution is already established by metadata. Repeating the
                # issuer's name/CIK in BM25 instead promotes filing cover pages.
                continue
            add(key, 0.25)
            add(value, 1.5 if key in {"name", "entity_id", "ticker", "cik", "series_id", "series_name", "series_fred", "description", "tenor"} else 0.5)

        scores = [0.0] * len(self.chunks)
        count = len(self.chunks)
        for token, weight in weights.items():
            postings = self._postings.get(token, [])
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                norm = 1.5 * (0.25 + 0.75 * self._lengths[index] / self._avg_len)
                scores[index] += weight * idf * frequency * 2.5 / (frequency + norm)
        candidates = [
            (scores[index] * affinities[chunk.doc_id], index)
            for index, chunk in enumerate(self.chunks)
            if chunk.doc_id not in excluded and scores[index] > 0
        ]
        if not candidates:
            # Lexical mismatch is not missing evidence. Offer real eligible
            # passages, preferring issuer affinity and then publication recency;
            # downstream code must still assess whether they support a forecast.
            candidates = [
                ((affinities[chunk.doc_id] - 1) * 1_000_000
                 + date.fromisoformat(chunk.doc_date).toordinal(), index)
                for index, chunk in enumerate(self.chunks)
                if chunk.doc_id not in excluded
            ]
        heap = [
            (-score, self.chunks[index].doc_id, self.chunks[index].span_start,
             index, 0, score)
            for score, index in candidates
            if len(self.chunks[index].text) <= _EVIDENCE_BUDGET
        ]
        heapq.heapify(heap)
        selected: list[Chunk] = []
        selected_tokens: list[set[str]] = []
        doc_counts: Counter = Counter()
        used = 0
        while heap and len(selected) < top_k:
            # Diminishing returns allow independent documents to compete without
            # forcing a quota of irrelevant sources into each prompt. Priorities
            # can only fall after accepting a passage from that document. Lazy
            # heap updates avoid rescanning all length/duplicate rejects.
            _, doc_id, start, index, seen_count, base_score = heapq.heappop(heap)
            chunk = self.chunks[index]
            if used + len(chunk.text) > _EVIDENCE_BUDGET:
                continue
            if seen_count != doc_counts[doc_id]:
                current_count = doc_counts[doc_id]
                score = base_score / (1 + 0.6 * current_count)
                heapq.heappush(heap, (-score, doc_id, start, index, current_count, base_score))
                continue
            tokens = set(_tokens(chunk.text))
            duplicate = False
            for previous, previous_tokens in zip(selected, selected_tokens):
                intersection = max(0, min(previous.span_end, chunk.span_end) - max(previous.span_start, chunk.span_start))
                if previous.doc_id == chunk.doc_id and intersection > min(len(previous.text), len(chunk.text)) / 3:
                    duplicate = True
                    break
                if chunk.text == previous.text:
                    duplicate = True
                    break
                union = tokens | previous_tokens
                if previous.doc_id == chunk.doc_id and union and len(tokens & previous_tokens) / len(union) > 0.90:
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(chunk)
            selected_tokens.append(tokens)
            doc_counts[chunk.doc_id] += 1
            used += len(chunk.text)
        # Complement the driver ranking with at most one absolute target-value
        # passage. Do not expand the lexical query or disturb the leading drivers.
        # Recognition is metric-based, independent of task/entity identifiers.
        target_text = json.dumps(target, ensure_ascii=False) + " " + str(task.get("prompt", ""))
        if (top_k > 1 and _EPS.search(target_text.replace("_", " "))
                and not any(_eps_baseline_score(c.text) for c in selected)):
            baselines = [
                (score * affinities[self.chunks[index].doc_id], index)
                for score, index in self._eps_baselines
                if self.chunks[index].doc_id not in excluded
                and (not issuer_matches or self.chunks[index].doc_id in issuer_matches)
            ]
            baselines.sort(key=lambda item: (-item[0], self.chunks[item[1]].doc_id,
                                             self.chunks[item[1]].span_start))
            if baselines:
                complement = self.chunks[baselines[0][1]]
                # Existing coverage (including overlapping windows) needs no slot.
                covered = any(
                    c.text == complement.text or (
                        c.doc_id == complement.doc_id
                        and min(c.span_end, complement.span_end) - max(c.span_start, complement.span_start)
                        > min(len(c.text), len(complement.text)) / 3
                    ) for c in selected
                )
                if not covered:
                    keep = list(selected)
                    remaining = used
                    while keep and (len(keep) >= top_k or remaining + len(complement.text) > _EVIDENCE_BUDGET):
                        remaining -= len(keep.pop().text)
                    if keep and remaining + len(complement.text) <= _EVIDENCE_BUDGET:
                        selected = keep + [complement]
        return selected
