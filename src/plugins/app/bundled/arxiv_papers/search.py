"""arXiv search through the `arxiv` package: query building, rate limiting and a small result cache.

arXiv's API terms: at most one request every three seconds, one connection at a time, for all of a user's
machines together; going over can get the address blocked for a while. So:
- Every search in the process goes through one gate (RATE_GATE), whichever panel or searcher sends it:
  one request at a time, at least MIN_INTERVAL apart.
- A request is never retried automatically. After an HTTP 429 ("slow down") the gate refuses to contact
  arXiv for a cool-down that doubles on each further 429 (COOLDOWN_START up to COOLDOWN_MAX).
- Identical searches are answered from a cache of result metadata (which the terms allow storing).
No Qt here, so it can be tested and scripted on its own.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .catalog import tag_phrase

MIN_INTERVAL = 3.1  # seconds between requests (arXiv API terms of use)
COOLDOWN_START = 120.0  # seconds without requests after arXiv answers 429
COOLDOWN_MAX = 1800.0
ACKNOWLEDGEMENT = "Thank you to arXiv for use of its open access interoperability."
CACHE_TTL = 24 * 3600
CACHE_LIMIT = 60
SORTS = {"relevance": "Relevance", "newest": "SubmittedDate", "updated": "LastUpdatedDate"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str  # ISO date
    updated: str
    primary_category: str
    categories: list[str] = field(default_factory=list)
    abs_url: str = ""
    pdf_url: str = ""

    @classmethod
    def from_result(cls, result) -> "Paper":
        return cls(
            arxiv_id=result.get_short_id(),
            title=" ".join(result.title.split()),
            authors=[author.name for author in result.authors],
            summary=" ".join(result.summary.split()),
            published=result.published.date().isoformat() if result.published else "",
            updated=result.updated.date().isoformat() if result.updated else "",
            primary_category=result.primary_category or "",
            categories=list(result.categories or []),
            abs_url=result.entry_id or "",
            pdf_url=result.pdf_url or "",
        )

    def citation(self) -> str:
        """A short BibTeX entry for the paper."""
        surname = (self.authors[0].split()[-1] if self.authors else "arxiv").lower()
        surname = "".join(ch for ch in surname if ch.isalnum()) or "arxiv"
        key = f"{surname}{self.published[:4]}{self.arxiv_id.split('v')[0].replace('.', '')[-4:]}"
        return (
            f"@misc{{{key},\n"
            f"  title         = {{{self.title}}},\n"
            f"  author        = {{{' and '.join(self.authors)}}},\n"
            f"  year          = {{{self.published[:4]}}},\n"
            f"  eprint        = {{{self.arxiv_id.split('v')[0]}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  primaryClass  = {{{self.primary_category}}},\n"
            f"  url           = {{{self.abs_url}}}\n"
            f"}}"
        )


class SearchError(Exception):
    """A search that failed, with a message fit to show in the panel."""


class RateLimited(SearchError):
    """arXiv answered 429: it wants fewer requests."""


class RateGate:
    """Process-wide spacing and cool-down for requests to arXiv."""

    def __init__(self):
        self.lock = threading.Lock()  # held for the whole request: one connection at a time
        self.reset()

    def reset(self):
        self.last_request = None  # time.monotonic() when the last request finished
        self.blocked_until = 0.0
        self.cooldown = COOLDOWN_START

    def remaining_cooldown(self) -> float:
        return max(0.0, self.blocked_until - time.monotonic())

    def wait_turn(self):
        """Sleep until the next request is allowed. Call with `lock` held."""
        if self.last_request is not None:
            wait = MIN_INTERVAL - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)

    def finished(self, rate_limited: bool):
        self.last_request = time.monotonic()
        if rate_limited:
            self.blocked_until = self.last_request + self.cooldown
            self.cooldown = min(COOLDOWN_MAX, self.cooldown * 2)
        else:
            self.cooldown = COOLDOWN_START


RATE_GATE = RateGate()


def _cooldown_message(seconds: float) -> str:
    wait = f"{round(seconds / 60)} minutes" if seconds >= 90 else f"{max(1, round(seconds))} seconds"
    return f"arXiv asked ThetaIDE to slow down. Searching again in about {wait}; saved results still open."


def build_query(tags, text: str = "", match_all: bool = True) -> str:
    """arXiv query for the selected tags (each tag's phrase, matched anywhere) plus free-text words.

    >>> build_query(["offline-rl", "cql"], "benchmark")
    'all:"offline reinforcement learning" AND all:"conservative q-learning" AND all:benchmark'
    """
    tag_terms = [f'all:"{tag_phrase(tag)}"' for tag in dict.fromkeys(tags) if tag_phrase(tag)]
    words = [w for w in "".join(ch if ch.isalnum() or ch in "-_." else " " for ch in text).split() if w]
    text_terms = [f"all:{w}" for w in words]
    if not match_all and len(tag_terms) > 1:
        tag_terms = ["(" + " OR ".join(tag_terms) + ")"]
    return " AND ".join(tag_terms + text_terms)


_inject_lock = threading.Lock()
_injected = False


def _use_system_certificates():
    """Verify HTTPS against the OS certificate store when truststore is installed.

    Networks that inspect HTTPS (antivirus, corporate proxies) re-sign certificates with a root that only
    the OS store trusts; without this, every request fails with CERTIFICATE_VERIFY_FAILED.
    """
    global _injected
    with _inject_lock:
        if _injected:
            return
        _injected = True
        try:
            import truststore

            truststore.inject_into_ssl()
        except Exception:
            pass


class ArxivSearcher:
    """Runs searches one at a time, spaced out for the API, with an optional on-disk cache."""

    def __init__(self, cache_path: Path | None = None, fetch=None):
        self.cache_path = Path(cache_path) if cache_path else None
        self._fetch = fetch or self._fetch_from_arxiv
        self._cache_lock = threading.Lock()
        self._cache: dict[str, dict] = self._load_cache()

    # ── cache ────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if not self.cache_path or not self.cache_path.is_file():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_cache(self):
        if not self.cache_path:
            return
        newest = sorted(self._cache.items(), key=lambda item: item[1].get("time", 0))[-CACHE_LIMIT:]
        self._cache = dict(newest)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except OSError:
            pass

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 25, sort: str = "relevance") -> tuple[list[Paper], bool]:
        """Papers matching `query`, and whether they came from the cache. Raises SearchError."""
        if not query.strip():
            raise SearchError("Choose a tag or type something to search for.")
        sort = sort if sort in SORTS else "relevance"
        key = json.dumps([query, int(max_results), sort])
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit and time.time() - hit.get("time", 0) < CACHE_TTL:
            return [Paper(**item) for item in hit["papers"]], True
        with RATE_GATE.lock:
            remaining = RATE_GATE.remaining_cooldown()
            if remaining > 0:
                raise RateLimited(_cooldown_message(remaining))
            RATE_GATE.wait_turn()
            rate_limited = False
            try:
                papers = self._fetch(query, int(max_results), sort)
            except RateLimited:
                rate_limited = True
                raise RateLimited(_cooldown_message(RATE_GATE.cooldown)) from None
            finally:
                RATE_GATE.finished(rate_limited)
        with self._cache_lock:
            self._cache[key] = {"time": time.time(), "papers": [asdict(p) for p in papers]}
            self._save_cache()
        return papers, False

    @staticmethod
    def _fetch_from_arxiv(query: str, max_results: int, sort: str) -> list[Paper]:
        try:
            import arxiv
        except ImportError as exc:
            raise SearchError("The arxiv package is not installed: pip install arxiv") from exc
        _use_system_certificates()
        # No automatic retries: a failed request is reported, and the user decides when to search again.
        client = arxiv.Client(page_size=max_results, delay_seconds=MIN_INTERVAL, num_retries=0)
        search = arxiv.Search(query=query, max_results=max_results,
                              sort_by=getattr(arxiv.SortCriterion, SORTS[sort]))
        try:
            return [Paper.from_result(result) for result in client.results(search)]
        except arxiv.HTTPError as exc:
            if getattr(exc, "status", None) in (429, 503):
                raise RateLimited("rate limited") from exc
            raise SearchError(f"arXiv returned an error (HTTP {getattr(exc, 'status', '?')}).") from exc
        except arxiv.UnexpectedEmptyPageError as exc:
            raise SearchError("arXiv returned an empty page. Try the search again.") from exc
        except Exception as exc:  # requests' connection, timeout and SSL errors
            name = type(exc).__name__
            if "SSL" in name or "SSL" in str(exc)[:200]:
                raise SearchError("Couldn't verify arxiv.org's certificate. Installing truststore "
                                  "(pip install truststore) lets ThetaIDE use the system's certificates.") from exc
            raise SearchError(f"Couldn't reach arXiv ({name}). Check your connection.") from exc
