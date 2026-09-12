from pathlib import Path

CATALOG = Path("db/schema_catalog.md")
QUERY_DIR = Path("db/queries")


def load_catalog():
    if not CATALOG.exists():
        raise FileNotFoundError(f"missing {CATALOG}")
    return CATALOG.read_text(encoding="utf-8")


def load_examples(limit=4):
    if not QUERY_DIR.exists():
        return []

    files = sorted(QUERY_DIR.glob("*.sql"))[:limit]
    out = []

    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        lines = text.splitlines()

        question = (
            lines[0].lstrip("-\t").lstrip()
            if lines
            else f.stem
        )
        sql = "\n".join(lines[1:]).strip()

        out.append({
            "question": question,
            "sql": sql,
        })

    return out


def build_context(include_examples=True):
    parts = [load_catalog()]

    if include_examples:
        for ex in load_examples():
            parts.append(
                f"Question:\t{ex['question']}\n"
                f"SQL:\n{ex['sql']}"
            )

    return "\n\n".join(parts)