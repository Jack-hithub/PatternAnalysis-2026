"""Render the saved feasibility plan as a readable, linked, single-page PDF.

This renderer reads FEASIBILITY.md; it does not insert experiment results or
edit the scientific plan. ReportLab is required only for document generation.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

PROJECT = Path(__file__).resolve().parents[1]
PURPLE = colors.HexColor("#51247A")
GRAY = colors.HexColor("#4A4A4A")


def inline_markdown(text: str) -> str:
    """Convert only the source's bold and Markdown links to safe PDF markup."""
    text = (text.replace("\u2014", " - ").replace("\u2013", "-")
            .replace("\u2011", "-").replace("\u00a0", " "))
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                  r'<link href="\2" color="#51247A"><u>\1</u></link>', text)
    return text


def render(source: Path, output: Path) -> int:
    paragraphs = [" ".join(block.splitlines())
                  for block in source.read_text(encoding="utf-8").strip().split("\n\n")]
    if not paragraphs or not paragraphs[0].startswith("# "):
        raise ValueError("Expected a Markdown title followed by the saved plan.")
    title = paragraphs.pop(0)[2:]
    subtitle = paragraphs.pop(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    page_counter = {"count": 0}
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=11,
                          leading=14.1, textColor=colors.HexColor("#202020"),
                          alignment=TA_LEFT, spaceAfter=8)
    heading = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold",
                             fontSize=18, leading=21, textColor=PURPLE, spaceAfter=7)
    meta = ParagraphStyle("Metadata", parent=body, fontSize=9.2, leading=11.5,
                          textColor=GRAY, spaceAfter=11)
    refs = ParagraphStyle("References", parent=body, fontSize=9.2, leading=11.7,
                          textColor=GRAY, spaceAfter=0)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=43,
                            leftMargin=43, topMargin=38, bottomMargin=40,
                            title=title, author="Baoxu Wang (s4927782)",
                            subject="COMP3710 HipMRI fixed execution feasibility plan",
                            pageCompression=1)
    story = [Paragraph(inline_markdown(title), heading),
             Paragraph(inline_markdown(subtitle), meta)]
    for text in paragraphs:
        story.append(Paragraph(inline_markdown(text), refs if text.startswith("References:") else body))
    def footer(canvas, document):
        page_counter["count"] += 1
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D8CDDF"))
        canvas.line(43, 32, A4[0]-43, 32)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(43, 20, "COMP3710 | Feasibility plan | 28 October 2026")
        canvas.drawRightString(A4[0]-43, 20, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    if page_counter["count"] != 1:
        raise ValueError(f"Feasibility plan must fit one readable page; rendered {page_counter['count']}.")
    return page_counter["count"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "FEASIBILITY.md")
    parser.add_argument("--output", type=Path,
                        default=PROJECT.parents[2] / "output/pdf/Demo3_Feasibility_Plan.pdf")
    args = parser.parse_args(argv)
    count = render(args.source, args.output)
    print(f"Rendered {count} page: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
