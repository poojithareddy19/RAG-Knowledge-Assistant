"""Embed the data summaries and retrieve the ones a question is about.

This is the retrieval layer, and it serves two callers. Before SQL generation
it tells the model that float 1901393 exists, worked the Arabian Sea, and
stopped reporting in 2021, so the model is not guessing what is in the schema.
On the summaries route it is the corpus itself: a question about what the
database holds is answered from these summaries, with the float or region each
one describes as the citation.
"""

from __future__ import annotations

import re

import numpy as np

from src.embeddings.embedding_model import get_embedding_model
from src.semantic.summaries import collect
from src.utils.config import get_config
from src.utils.db import cursor
from src.utils.schemas import Summary

__all__ = ["IDENTIFIER", "SemanticIndex", "Summary", "as_context"]

UPSERT = """
INSERT INTO {table} (subject_kind, subject_id, content, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT (subject_kind, subject_id) DO UPDATE
SET content  = EXCLUDED.content,
    embedding = EXCLUDED.embedding,
    built_at = now()
"""

# Everything not just rebuilt is about a subject the database no longer holds.
# The upsert alone never removed anything, so when the synthetic floats were
# replaced by real ones, all forty synthetic summaries stayed: floats 2900000
# to 2900039, "730 profiles" each, retrievable and quoted as context for floats
# that were not in the tables. Deleted in the same transaction as the upsert,
# so a reader never sees a half-rebuilt index.
PRUNE = """
DELETE FROM {table} AS t
WHERE NOT EXISTS (
    SELECT 1
    FROM unnest(%s::text[], %s::text[]) AS keep(subject_kind, subject_id)
    WHERE keep.subject_kind = t.subject_kind
      AND keep.subject_id = t.subject_id
)
"""

SEARCH = """
SELECT
    subject_kind,
    subject_id,
    content,
    1 - (embedding <=> %s) AS similarity
FROM {table}
ORDER BY embedding <=> %s
LIMIT %s
"""

BY_ID = """
SELECT subject_kind, subject_id, content
FROM {table}
WHERE subject_id = ANY(%s)
"""

# A WMO float identifier. Long enough not to catch a year or a depth.
IDENTIFIER = re.compile(r"\b\d{5,}\b")


class SemanticIndex:
    def __init__(self, table: str | None = None) -> None:
        cfg = get_config().get("semantic", {})

        self.table = table or cfg.get("table", "data_summaries")
        self.top_k = int(cfg.get("top_k", 4))
        self.min_similarity = float(cfg.get("min_similarity", 0.25))

    def refresh(self) -> dict:
        """Rebuild every summary from the current contents of the database."""
        subjects = collect()

        if not subjects:
            return {"subjects": 0}

        embedder = get_embedding_model()
        vectors = embedder.embed_passages([text for _, _, text in subjects])

        rows = [
            (kind, subject, text, vectors[i])
            for i, (kind, subject, text) in enumerate(subjects)
        ]

        with cursor() as cur:
            cur.executemany(UPSERT.format(table=self.table), rows)
            cur.execute(
                PRUNE.format(table=self.table),
                (
                    [kind for kind, _, _ in subjects],
                    [subject for _, subject, _ in subjects],
                ),
            )
            removed = cur.rowcount

        kinds: dict[str, int] = {}

        for kind, _, _ in subjects:
            kinds[kind] = kinds.get(kind, 0) + 1

        return {"subjects": len(subjects), "removed": removed, **kinds}

    def search(self, question: str, k: int | None = None) -> list[Summary]:
        """The summaries a question is about: named ones first, then nearest.

        An identifier is looked up literally rather than embedded. Float
        summaries are near-identical sentences whose only distinguishing token
        is a long number, which is exactly what embeddings represent worst:
        asking for float 2900007 scores it 0.703 against 0.701 for an unrelated
        float. A question that names a float should get that float.
        """
        limit = k or self.top_k

        named = self._by_identifier(question)

        vector = np.asarray(
            get_embedding_model().embed_query(question),
            dtype=np.float32,
        ).reshape(-1)

        with cursor(readonly=True) as cur:
            cur.execute(
                SEARCH.format(table=self.table),
                (vector, vector, limit),
            )
            rows = cur.fetchall()

        nearest = [
            Summary(
                kind=kind,
                subject=subject,
                text=content,
                score=float(similarity),
            )
            for kind, subject, content, similarity in rows
            if similarity >= self.min_similarity
        ]

        seen = {(hit.kind, hit.subject) for hit in named}

        for hit in nearest:
            if (hit.kind, hit.subject) not in seen:
                named.append(hit)
                seen.add((hit.kind, hit.subject))

        return named[:limit]

    def _by_identifier(self, question: str) -> list[Summary]:
        """Summaries whose subject the question names outright."""
        identifiers = IDENTIFIER.findall(question)

        if not identifiers:
            return []

        with cursor(readonly=True) as cur:
            cur.execute(
                BY_ID.format(table=self.table),
                (identifiers,),
            )
            rows = cur.fetchall()

        return [
            Summary(
                kind=kind,
                subject=subject,
                text=content,
                score=1.0,
            )
            for kind, subject, content in rows
        ]

    def count(self) -> int:
        with cursor(readonly=True) as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])


def as_context(summaries: list[Summary]) -> str:
    """Render retrieved summaries as the prompt block the generator injects."""
    return "\n".join(f"- {summary.text}" for summary in summaries)
