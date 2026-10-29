"""Render the saved Markdown report to a paginated, linked, readable PDF.

Supports H1-H4, paragraphs, emphasis, inline code/links, lists, fenced code,
local PNG/JPEG images, and GFM pipe tables. Source content is never modified.
The Markdown Contents block becomes a linked PDF contents table with page
numbers resolved across layout passes; the report body starts on a new page.
The feasibility review starts and ends on page boundaries, using the same
semantic content as the README. Use --draft while final submission review remains pending.
ReportLab and Pillow are document-generation dependencies, not model runtime.
"""
from __future__ import annotations

import argparse
from copy import copy
from hashlib import sha256
import html
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

from PIL import Image as PILImage
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

PROJECT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = A4
MARGIN = 43
CONTENT_W = PAGE_W - 2*MARGIN - 12
CONTENT_H = PAGE_H - 52 - 43 - 12
PURPLE = colors.HexColor("#51247A")
INK = colors.HexColor("#202834")
MUTED = colors.HexColor("#506071")
LINK_BASE = "https://github.com/Jack-hithub/PatternAnalysis-2026/blob/topic-recognition/recognition/ImprovedUNet3D_s4927782/"
TOKEN = re.compile(r"(`[^`]+`|\[[^\]]+\]\([^\n)]+\)|https?://[^\s<>]+|\*\*[^*]+\*\*|(?<![\w/*])\*(?![\s*])[^*]+?(?<!\s)\*(?![\w/*]))")
BULLET = re.compile(r"^(\s*)([-+*]|\d+[.)])\s+(.*)$")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$")


def register_fonts():
    fonts = Path(reportlab.__file__).parent/"fonts"
    for name, filename in (("Body", "Vera.ttf"), ("Body-Bold", "VeraBd.ttf"),
                           ("Body-Italic", "VeraIt.ttf"), ("Body-BoldItalic", "VeraBI.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts/filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                  italic="Body-Italic", boldItalic="Body-BoldItalic")
    candidates = [Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")]
    candidates.extend(fonts.parent.parent.rglob("DejaVuSansMono.ttf"))
    mono = next((p for p in candidates if p.is_file()), None)
    if mono is None:
        # ReportLab's built-in Courier remains vector text if no embeddable mono
        # font is available. Body text always uses the bundled embedded family.
        return "Courier"
    if "Mono" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Mono", str(mono)))
    return "Mono"


def normalize(text):
    text = text.replace("\u2192", " to ")
    for char in "\u2010\u2011\u2012\u2013\u2014\u2212":
        text = text.replace(char, "-")
    return text.replace("\u00a0", " ")


def inline(text, mono="Mono"):
    text = normalize(text)
    pieces, cursor = [], 0
    for match in TOKEN.finditer(text):
        pieces.append(html.escape(text[cursor:match.start()]))
        token = match.group()
        if token.startswith("`"):
            pieces.append(f'<font name="{mono}" size="9.2">{html.escape(token[1:-1])}</font>')
        elif token.startswith("["):
            label, destination = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token).groups()
            target = destination if urlsplit(destination).scheme else LINK_BASE + quote(unquote(destination), safe="/#")
            pieces.append('<link href="'+html.escape(target, quote=True)+'" color="#51247A"><u>'
                          +html.escape(label)+'</u></link>')
        elif token.startswith("http"):
            destination = token.rstrip(".,;:")
            suffix = token[len(destination):]
            pieces.append('<link href="'+html.escape(destination, quote=True)+'" color="#51247A">'
                          +html.escape(destination)+'</link>'+html.escape(suffix))
        elif token.startswith("**"):
            pieces.append("<b>"+html.escape(token[2:-2])+"</b>")
        else:
            pieces.append("<i>"+html.escape(token[1:-1])+"</i>")
        cursor = match.end()
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def styles(*, feasibility=False):
    base = ParagraphStyle("Body", fontName="Body", fontSize=10.5,
                          leading=14.3 if not feasibility else 13.4,
                          textColor=INK, spaceAfter=8 if not feasibility else 6,
                          splitLongWords=True, allowWidows=0, allowOrphans=0,
                          bulletFontName="Body", bulletFontSize=10.5)
    result = {"body": base}
    for level, size in ((1, 19.5), (2, 14.5), (3, 12.0), (4, 10.8)):
        result[f"h{level}"] = ParagraphStyle(f"H{level}", parent=base,
            fontName="Body-Bold", fontSize=size, leading=size*1.22,
            textColor=PURPLE if level < 3 else INK,
            spaceBefore=0 if feasibility else (11 if level > 1 else 0),
            spaceAfter=7, keepWithNext=True)
    result["caption"] = ParagraphStyle("Caption", parent=base, fontSize=9.0,
        leading=11.7, textColor=MUTED, spaceBefore=6, spaceAfter=11)
    result["cell"] = ParagraphStyle("Cell", parent=base, fontSize=9.3,
        leading=12.1, spaceAfter=0, allowWidows=1, allowOrphans=1)
    result["cell_header"] = ParagraphStyle("CellHeader", parent=result["cell"],
        fontName="Body-Bold", textColor=colors.white)
    result["draft"] = ParagraphStyle("Draft", parent=base, fontName="Body-Bold",
        fontSize=11, leading=15, textColor=colors.HexColor("#7A4610"),
        backColor=colors.HexColor("#FFF2DC"), borderPadding=8, spaceAfter=16)
    return result


class CodeBlock(Flowable):
    """Vector monospace snippets with safe visual wrapping and page splitting."""
    def __init__(self, lines, mono, wrapped=False):
        super().__init__()
        self.lines, self.mono, self.wrapped = list(lines), mono, wrapped
        self.font_size, self.leading, self.padding = 9.0, 12.0, 9
        self.spaceBefore, self.spaceAfter = 4, 10

    def wrap(self, available_width, available_height):
        self.width = available_width
        if not self.wrapped:
            result = []
            for line in self.lines:
                remainder = normalize(line.expandtabs(4))
                if not remainder:
                    result.append("")
                while remainder:
                    count = len(remainder)
                    while count > 1 and pdfmetrics.stringWidth(remainder[:count], self.mono, self.font_size) > self.width-2*self.padding:
                        count -= 1
                    # Prefer whitespace while retaining every source character.
                    if count < len(remainder):
                        space = remainder.rfind(" ", max(1, count//2), count+1)
                        if space > 0:
                            count = space+1
                    result.append(remainder[:count])
                    remainder = remainder[count:]
            self.lines, self.wrapped = result or [""], True
        self.height = len(self.lines)*self.leading + 2*self.padding
        return self.width, self.height

    def split(self, available_width, available_height):
        self.wrap(available_width, available_height)
        count = int((available_height-2*self.padding)/self.leading)
        if count < 2:
            return []
        return [CodeBlock(self.lines[:count], self.mono, True),
                CodeBlock(self.lines[count:], self.mono, True)]

    def draw(self):
        c = self.canv
        c.setFillColor(colors.HexColor("#F0F3F6"))
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont(self.mono, self.font_size)
        for index, line in enumerate(self.lines):
            c.drawString(self.padding, self.height-self.padding-self.font_size-index*self.leading, line)


def table_cells(line):
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", text)]


def table_separator(line):
    cells = table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells)


def visible(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", normalize(text))
    return re.sub(r"[*`]", "", text)


def make_table(lines, style, mono):
    rows = [table_cells(line) for line in lines]
    rows = [rows[0]]+rows[2:]
    count = len(rows[0])
    if count < 1 or any(len(row) != count for row in rows):
        raise ValueError("Markdown table has inconsistent column counts.")
    measures = [min(190, max(50, max(pdfmetrics.stringWidth(visible(row[j]), "Body", 9.3)
                                   for row in rows)+15)) for j in range(count)]
    widths = [CONTENT_W*x/sum(measures) for x in measures]
    numeric = []
    for j in range(count):
        values = [visible(row[j]) for row in rows[1:]]
        numeric.append(bool(values) and sum(bool(re.fullmatch(r"[\s\d.,%+±×()/-]+|N/?A|undefined|None", v)) for v in values) >= .6*len(values))
    data = []
    for i, row in enumerate(rows):
        current = []
        for j, cell in enumerate(row):
            cell_style = copy(style["cell_header" if i == 0 else "cell"])
            if numeric[j]:
                cell_style.alignment = TA_RIGHT
            current.append(Paragraph(inline(cell, mono), cell_style))
        data.append(current)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1, splitInRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), PURPLE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F3F0F6"), colors.white]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LINEBELOW", (0,0), (-1,0), .4, PURPLE),
        ("LINEBELOW", (0,1), (-1,-1), .35, colors.HexColor("#D8DDE2")),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    table.spaceBefore, table.spaceAfter = 4, 11
    return table


def make_image(path, alt, source_root, style, mono):
    if urlsplit(path).scheme:
        raise ValueError("Use a saved local figure; renderer does not download images.")
    image_path = (source_root/unquote(path)).resolve(strict=True)
    with PILImage.open(image_path) as im:
        width, height = im.size
    caption = Paragraph(inline(alt, mono), style["caption"])
    _, caption_height = caption.wrap(CONTENT_W, CONTENT_H)
    scale = min(CONTENT_W/width, (CONTENT_H-caption_height-35)/height)
    figure = Image(str(image_path), width=width*scale, height=height*scale)
    figure.hAlign = "CENTER"
    return KeepTogether([figure, caption])


def markdown_story(text, source_root, mono, *, feasibility=False, manage_feasibility=True):
    # Preserve the complete feasibility section as a separate, readable page.
    if manage_feasibility:
        found = re.search(r"^## Feasibility review\s*$", text, re.M)
        if found:
            following = re.search(r"^## ", text[found.end():], re.M)
            end = found.end()+following.start() if following else len(text)
            before = markdown_story(text[:found.start()], source_root, mono, manage_feasibility=False)
            middle = markdown_story(text[found.start():end], source_root, mono, feasibility=True, manage_feasibility=False)
            after = markdown_story(text[end:], source_root, mono, manage_feasibility=False)
            return before+[PageBreak(), KeepTogether(middle)]+([PageBreak()]+after if after else [])
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines, story, style = text.splitlines(), [], styles(feasibility=feasibility)
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        fence = re.match(r"^\s*(```+|~~~+)", line)
        if fence:
            marker, block = fence.group(1), []
            index += 1
            while index < len(lines) and not lines[index].lstrip().startswith(marker):
                block.append(lines[index]); index += 1
            if index == len(lines):
                raise ValueError("Unclosed Markdown code fence.")
            code = CodeBlock(block, mono)
            _, height = code.wrap(CONTENT_W, CONTENT_H)
            story.append(KeepTogether([code]) if height < 200 else code)
            index += 1
            continue
        heading = HEADING.match(line)
        if heading:
            level = min(4, len(heading.group(1)))
            p = Paragraph(inline(heading.group(2), mono), style[f"h{level}"])
            p._heading_level = level
            p._heading_label = visible(heading.group(2))
            story.append(p)
            index += 1
            if len(heading.group(1)) == 2 and p._heading_label == "Contents":
                # Keep this indexing flowable at story level so multiBuild can
                # discover it. GitHub retains the source Markdown anchor list.
                contents = TableOfContents()
                contents.levelStyles = [ParagraphStyle("ContentsEntry", parent=style["body"],
                    fontSize=10.2, leading=15, textColor=PURPLE,
                    leftIndent=0, firstLineIndent=0, spaceBefore=1, spaceAfter=2)]
                contents.dotsMinLevel = 0
                story.append(contents)
                while index < len(lines):
                    following = HEADING.match(lines[index])
                    if following and len(following.group(1)) <= 2:
                        break
                    index += 1
                if index < len(lines):
                    story.append(PageBreak())
            continue
        image = IMAGE.fullmatch(line)
        if image:
            story.append(make_image(image.group(2), image.group(1), source_root, style, mono))
            index += 1
            continue
        if "|" in line and index+1 < len(lines) and table_separator(lines[index+1]):
            block = [line, lines[index+1]]
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                block.append(lines[index]); index += 1
            table = make_table(block, style, mono)
            _, table_height = table.wrap(CONTENT_W, CONTENT_H)
            # Keep short metric tables intact; long listings retain repeated
            # headers and row splitting across pages.
            story.append(KeepTogether([table]) if table_height <= 250 else table)
            continue
        bullet = BULLET.match(line)
        if bullet:
            indent, marker, item = bullet.groups()
            index += 1
            more = [item]
            while index < len(lines) and lines[index].strip() and not BULLET.match(lines[index]):
                if not lines[index].startswith((" ", "\t")):
                    break
                more.append(lines[index].strip()); index += 1
            item_style = ParagraphStyle("ListItem", parent=style["body"],
                leftIndent=16+len(indent)*4, firstLineIndent=0,
                bulletIndent=len(indent)*4, spaceAfter=4)
            story.append(Paragraph(inline(" ".join(more), mono), item_style,
                                   bulletText="•" if marker in "-+*" else marker))
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip():
            if HEADING.match(lines[index]) or BULLET.match(lines[index]) or IMAGE.fullmatch(lines[index]) or re.match(r"^\s*(```|~~~)", lines[index]):
                break
            if "|" in lines[index] and index+1 < len(lines) and table_separator(lines[index+1]):
                break
            paragraph.append(lines[index].strip()); index += 1
        story.append(Paragraph(inline(" ".join(paragraph), mono), style["body"]))
    return story


class ReportDoc(SimpleDocTemplate):
    def beforeDocument(self):
        # Every multiBuild pass must produce the same anchors and final counts.
        self.heading_counter = 0
        self.rendered_pages = []
        self.contents_entries = []
        super().beforeDocument()

    def afterFlowable(self, flowable):
        if not hasattr(flowable, "_heading_level") or flowable._heading_level > 2:
            return
        counter = getattr(self, "heading_counter", 0)+1
        self.heading_counter = counter
        key = f"section_{counter}"
        self.canv.bookmarkPage(key)
        level = 0 if flowable._heading_level == 1 else 1
        if counter == 1:
            level = 0
        self.canv.addOutlineEntry(flowable._heading_label, key, level=level, closed=False)
        if flowable._heading_level == 2 and flowable._heading_label != "Contents":
            self.notify("TOCEntry", (0, html.escape(flowable._heading_label), self.page, key))
            self.contents_entries.append({"title": flowable._heading_label,
                                          "page": self.page, "bookmark": key})


def render(source, output, *, draft=False):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if source == output:
        raise ValueError("PDF output must not overwrite its Markdown source.")
    text = source.read_text(encoding="utf-8")
    mono = register_fonts()
    story = markdown_story(text, source.parent, mono)
    if draft:
        story.insert(0, Paragraph("WORKING DRAFT: final submission review pending", styles()["draft"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem+".building.pdf")
    title = next((m.group(2) for line in text.splitlines() if (m := HEADING.match(line)) and len(m.group(1)) == 1), "HipMRI segmentation report")
    doc = ReportDoc(str(temporary), pagesize=A4,
                    leftMargin=MARGIN, rightMargin=MARGIN,
                    topMargin=52, bottomMargin=43,
                    title=title+(" - WORKING DRAFT" if draft else ""),
                    author="Baoxu Wang (s4927782)",
                    subject="COMP3710 HipMRI segmentation study", pageCompression=1)
    def chrome(canvas, document):
        document.rendered_pages.append(document.page)
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D8CDDF"))
        canvas.setLineWidth(.45)
        canvas.line(MARGIN, PAGE_H-33, PAGE_W-MARGIN, PAGE_H-33)
        canvas.setFont("Body", 8.0)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE_H-24, "COMP3710 | HipMRI segmentation report"+(" | WORKING DRAFT" if draft else ""))
        canvas.drawString(MARGIN, 22, "Baoxu Wang (s4927782)")
        canvas.drawRightString(PAGE_W-MARGIN, 22, f"Page {document.page}")
        canvas.restoreState()
    passes = doc.multiBuild(story, onFirstPage=chrome, onLaterPages=chrome)
    if not temporary.is_file() or not temporary.stat().st_size:
        raise ValueError("Report generation produced no readable file.")
    temporary.replace(output)
    return {"status": "rendered", "pages": len(doc.rendered_pages), "draft": draft,
            "contents": doc.contents_entries, "layout_passes": passes,
            "source_sha256": sha256(text.encode()).hexdigest(),
            "output": str(output), "bytes": output.stat().st_size,
            "output_sha256": sha256(output.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT/"README.md")
    parser.add_argument("--output", type=Path,
                        default=PROJECT.parents[2]/"output/pdf/Demo3_README_Report.pdf")
    parser.add_argument("--draft", action="store_true", help="Mark all pages as a working draft with final submission review pending.")
    args = parser.parse_args(argv)
    print(json.dumps(render(args.source, args.output, draft=args.draft), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
