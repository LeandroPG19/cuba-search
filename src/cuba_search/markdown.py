"""V17: HTML → Markdown structured conversion.

Converts HTML to clean Markdown preserving headings, tables, lists,
code blocks, and links. Produces structured output for AI consumption.
Uses dispatch table pattern to keep CC ≤ 7 per function.
"""
import re
from typing import Any

from bs4 import BeautifulSoup, Tag


def html_to_markdown(html: str) -> str:
    """Convert HTML to clean Markdown.

    Preserves: headings, tables, lists, code, links, bold, italic.
    Strips: scripts, styles, nav, footer, ads.

    Args:
        html: Raw or readability-cleaned HTML.

    Returns:
        Markdown-formatted string.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    parts: list[str] = []
    _walk(soup, parts)

    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Tag handler dispatch table (CC ≤ 3 per handler) ──────────────

def _handle_heading(el: Tag, parts: list[str]) -> None:
    """Convert heading tag to markdown."""
    level = int(el.name[1])
    text = el.get_text(strip=True)
    if text:
        parts.append(f"\n{'#' * level} {text}\n")


def _handle_pre(el: Tag, parts: list[str]) -> None:
    """Convert pre/code block to markdown fenced block."""
    code = el.get_text()
    lang = ""
    code_tag = el.find("code")
    if code_tag and isinstance(code_tag, Tag):
        classes = code_tag.get("class", [])
        if isinstance(classes, list):
            for c in classes:
                if isinstance(c, str) and c.startswith("language-"):
                    lang = c[9:]
                    break
    parts.append(f"\n```{lang}\n{code.strip()}\n```\n")


def _handle_inline_code(el: Tag, parts: list[str]) -> None:
    """Convert inline code to markdown backticks."""
    parts.append(f"`{el.get_text(strip=True)}`")


def _handle_link(el: Tag, parts: list[str]) -> None:
    """Convert anchor to markdown link."""
    href = el.get("href", "")
    text = el.get_text(strip=True)
    if text and href:
        parts.append(f"[{text}]({href})")
    elif text:
        parts.append(text)


def _handle_bold(el: Tag, parts: list[str]) -> None:
    """Convert bold tags to markdown."""
    text = el.get_text(strip=True)
    if text:
        parts.append(f"**{text}**")


def _handle_italic(el: Tag, parts: list[str]) -> None:
    """Convert italic tags to markdown."""
    text = el.get_text(strip=True)
    if text:
        parts.append(f"*{text}*")


def _handle_block(el: Tag, parts: list[str]) -> None:
    """Convert block elements — recurse into children."""
    for child in el.children:
        _walk(child, parts)
    parts.append("\n")


# Dispatch table: tag name → handler function
_TAG_HANDLERS: dict[str, Any] = {
    "h1": _handle_heading,
    "h2": _handle_heading,
    "h3": _handle_heading,
    "h4": _handle_heading,
    "h5": _handle_heading,
    "h6": _handle_heading,
    "pre": _handle_pre,
    "a": _handle_link,
    "strong": _handle_bold,
    "b": _handle_bold,
    "em": _handle_italic,
    "i": _handle_italic,
    "p": _handle_block,
    "div": _handle_block,
    "section": _handle_block,
    "article": _handle_block,
    "blockquote": _handle_block,
    "table": lambda el, parts: _convert_table(el, parts),
    "ul": lambda el, parts: _convert_list(el, parts, ordered=False),
    "ol": lambda el, parts: _convert_list(el, parts, ordered=True),
}


def _walk(element: Any, parts: list[str]) -> None:
    """Recursively walk DOM and emit markdown via dispatch table.

    Args:
        element: BeautifulSoup element.
        parts: Accumulator list.
    """
    if isinstance(element, str):
        text = element.strip()
        if text:
            parts.append(text)
        return

    if not isinstance(element, Tag):
        return

    tag = element.name

    # Inline code special case (needs parent check)
    if tag == "code" and element.parent and element.parent.name != "pre":
        _handle_inline_code(element, parts)
        return

    # Line breaks
    if tag == "br":
        parts.append("\n")
        return

    # Dispatch to handler
    handler = _TAG_HANDLERS.get(tag)
    if handler:
        handler(element, parts)
        return

    # Default: recurse into children
    for child in element.children:
        _walk(child, parts)


def _convert_table(table: Tag, parts: list[str]) -> None:
    """Convert HTML table to Markdown table.

    Args:
        table: BeautifulSoup table element.
        parts: Accumulator list.
    """
    rows = table.find_all("tr")
    if not rows:
        return

    md_rows: list[list[str]] = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        md_rows.append([c.get_text(strip=True) for c in cells])

    if not md_rows:
        return

    # Normalize column count
    max_cols = max(len(r) for r in md_rows)
    for row in md_rows:
        while len(row) < max_cols:
            row.append("")

    # Header + separator
    parts.append("\n| " + " | ".join(md_rows[0]) + " |")
    parts.append("| " + " | ".join("---" for _ in md_rows[0]) + " |")

    # Data rows
    for row in md_rows[1:]:
        parts.append("| " + " | ".join(row) + " |")
    parts.append("")


def _convert_list(lst: Tag, parts: list[str], ordered: bool = False) -> None:
    """Convert HTML list to Markdown list.

    Args:
        lst: BeautifulSoup ul/ol element.
        parts: Accumulator list.
        ordered: Whether list is ordered.
    """
    items = lst.find_all("li", recursive=False)
    for i, item in enumerate(items):
        prefix = f"{i + 1}. " if ordered else "- "
        text = item.get_text(strip=True)
        parts.append(f"{prefix}{text}")
    parts.append("")
