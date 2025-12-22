#!/usr/bin/env python3
"""
Convert a Markdown document into JSONL chunks for RAG.

Each JSONL line has the shape:
  {"id": "...", "text": "...", "metadata": {...}}

Chunking strategy:
- Split by markdown headings (# .. ######) into "sections"
- For each section, build text prefixed with its section path
- If too long, split by paragraphs; if a paragraph is too long, split by characters
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Section:
    headings: List[str]  # full heading stack/path
    start_line: int
    end_line: int
    content: str

    @property
    def section_path(self) -> str:
        return " > ".join(self.headings).strip()


def _parse_pdf_pages(pages: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse strings like:
      - "8p"
      - "12p ~ 15p"
      - "12p~15p"
    Returns (start, end) ints when possible.
    """
    nums = [int(n) for n in re.findall(r"(\d+)\s*p", pages)]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], nums[0]
    return nums[0], nums[-1]


def _load_page_map(page_map_path: str) -> List[Dict[str, str]]:
    """
    Load page mapping rules.
    Expected JSON format:
      [
        {"match": "<substring to find in section_path>", "pages": "12p ~ 15p", "priority": 0},
        ...
      ]
    """
    with open(page_map_path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise ValueError("page map JSON must be a list of objects")
    rules: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"page map entry #{i} must be an object")
        match = str(item.get("match", "")).strip()
        pages = str(item.get("pages", "")).strip()
        priority_raw = item.get("priority", 0)
        try:
            priority = int(priority_raw)
        except Exception as e:
            raise ValueError(f"page map entry #{i} has invalid 'priority': {priority_raw!r}") from e
        if not match or not pages:
            raise ValueError(f"page map entry #{i} must have non-empty 'match' and 'pages'")
        rules.append({"match": match, "pages": pages, "priority": str(priority)})
    return rules


def _match_pages(section_path: str, rules: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not rules:
        return None
    best_pages: Optional[str] = None
    best_priority = -10**9
    best_len = -1
    for r in rules:
        m = r["match"]
        if m and (m in section_path):
            pr = int(r.get("priority", "0"))
            # Prefer higher priority first, then more specific (longest match).
            if (pr > best_priority) or (pr == best_priority and len(m) > best_len):
                best_priority = pr
                best_len = len(m)
                best_pages = r["pages"]
    return best_pages


def normalize_text(text: str) -> str:
    # Normalize line endings and trim trailing spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse excessive blank lines (3+ -> 2).
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def parse_sections(md_text: str) -> List[Section]:
    lines = md_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    headings: List[str] = []
    sections: List[Section] = []

    cur_start = 1
    cur_content: List[str] = []
    cur_headings: List[str] = []

    def flush(end_line: int) -> None:
        nonlocal cur_start, cur_content, cur_headings
        content = normalize_text("\n".join(cur_content))
        if content:
            sections.append(
                Section(
                    headings=list(cur_headings) if cur_headings else ["(no heading)"],
                    start_line=cur_start,
                    end_line=end_line,
                    content=content,
                )
            )
        cur_content = []
        cur_start = end_line + 1

    for idx, line in enumerate(lines, start=1):
        m = HEADING_RE.match(line)
        if m:
            # Finalize previous section (content under previous headings).
            flush(idx - 1)

            level = len(m.group(1))
            title = m.group(2).strip()

            # Update heading stack to this level.
            # Markdown levels are 1-based; stack length should become level-1 before append.
            if level <= 0:
                level = 1
            headings = headings[: max(level - 1, 0)]
            headings.append(title)

            cur_headings = list(headings)
            cur_start = idx + 1  # content starts after heading line
            continue

        cur_content.append(line)

    flush(len(lines))
    return sections


def split_by_chars(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    step = max(1, max_chars - max(0, overlap_chars))
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - max(0, overlap_chars)
    return chunks


def chunk_text(text: str, max_chars: int, overlap_chars: int = 150, overlap_paras: int = 1) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paras = re.split(r"\n{2,}", text)
    paras = [p.strip() for p in paras if p.strip()]

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def cur_text() -> str:
        return "\n\n".join(cur).strip()

    def cur_recalc_len() -> int:
        if not cur:
            return 0
        # account for "\n\n" separators
        return sum(len(p) for p in cur) + 2 * (len(cur) - 1)

    for p in paras:
        if len(p) > max_chars:
            # Flush current buffer first.
            if cur:
                chunks.append(cur_text())
                cur = []
                cur_len = 0
            chunks.extend(split_by_chars(p, max_chars=max_chars, overlap_chars=overlap_chars))
            continue

        add_len = len(p) if not cur else (2 + len(p))
        if cur and (cur_len + add_len) > max_chars:
            chunks.append(cur_text())
            cur = cur[-overlap_paras:] if overlap_paras > 0 else []
            cur_len = cur_recalc_len()

        cur.append(p)
        cur_len = cur_recalc_len()

    if cur:
        chunks.append(cur_text())
    return [c for c in chunks if c.strip()]


def iter_jsonl_records(
    input_path: str,
    md_text: str,
    max_chars: int,
    overlap_chars: int,
    page_map_rules: Optional[List[Dict[str, str]]] = None,
) -> Iterable[dict]:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rel_source = os.path.relpath(os.path.abspath(input_path), root_dir)

    base = os.path.splitext(os.path.basename(input_path))[0]
    sections = parse_sections(md_text)

    global_chunk_id = 0
    for s_idx, s in enumerate(sections):
        pages = _match_pages(s.section_path, page_map_rules)
        page_start, page_end = _parse_pdf_pages(pages) if pages else (None, None)

        # Prefix the chunk text with its section path to improve retrieval context.
        prefixed = f"{s.section_path}\n\n{s.content}".strip()
        chunks = chunk_text(prefixed, max_chars=max_chars, overlap_chars=overlap_chars)
        for c_idx, c in enumerate(chunks):
            global_chunk_id += 1
            metadata: Dict[str, Any] = {
                "source": rel_source,
                "title": base,
                "section": s.headings,
                "section_path": s.section_path,
                "section_index": s_idx,
                "chunk_index_in_section": c_idx,
                "start_line": s.start_line,
                "end_line": s.end_line,
                "max_chars": max_chars,
                "overlap_chars": overlap_chars,
            }
            if pages:
                metadata["pdf_pages"] = pages
            if page_start is not None:
                metadata["pdf_page_start"] = page_start
            if page_end is not None:
                metadata["pdf_page_end"] = page_end

            yield {
                "id": f"{base}::{global_chunk_id:05d}",
                "text": c,
                "metadata": metadata,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Markdown to JSONL chunks for RAG.")
    parser.add_argument("input", help="Input Markdown file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSONL file path (default: same dir, .jsonl extension)",
        default=None,
    )
    parser.add_argument("--max-chars", type=int, default=1200, help="Max chars per chunk (default: 1200)")
    parser.add_argument("--overlap-chars", type=int, default=150, help="Overlap chars for long splits (default: 150)")
    parser.add_argument(
        "--page-map",
        default=None,
        help="Optional JSON page map file to add pdf page numbers into metadata",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base}.jsonl")

    with open(input_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    page_map_rules: Optional[List[Dict[str, str]]] = None
    if args.page_map:
        page_map_rules = _load_page_map(args.page_map)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for rec in iter_jsonl_records(
            input_path=input_path,
            md_text=md_text,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            page_map_rules=page_map_rules,
        ):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote JSONL: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
