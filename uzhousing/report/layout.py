"""The bulletin's visual system: one place that decides how the report looks.

The layout follows the quarterly market review the project was given as a
structure reference — a designed publication rather than a printed dataframe:
a cover, a contents page with page numbers, centred teal section headings,
bulleted commentary beside each table, header-banded tables whose change
columns are coloured by direction, tinted source and note callouts, and a page
number set in a filled circle.

Nothing here decides what the report *says*. It receives finished text and
finished tables and gives them a consistent appearance, so that changing the
palette or the table banding is one edit rather than thirty.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

PAGE = A4
MARGIN = 17 * mm
CONTENT_WIDTH = PAGE[0] - 2 * MARGIN

#: Colours are named for their job, not their hue, so a re-skin is one edit.
PALETTE = {
    "ink": colors.HexColor("#1A1A1A"),
    "muted": colors.HexColor("#5A6A6A"),
    "deep": colors.HexColor("#14615F"),      # cover bands, headings
    "primary": colors.HexColor("#1C7C7A"),   # section headings, rules
    "band": colors.HexColor("#2E8B88"),      # table header fill
    "tint": colors.HexColor("#E8F2EF"),      # callout and zebra fill
    "tint_strong": colors.HexColor("#D2E7E2"),
    "pale": colors.HexColor("#F3F8F7"),
    "rise": colors.HexColor("#0E8A45"),      # a value that went up
    "fall": colors.HexColor("#C0392B"),      # a value that went down
    "flat": colors.HexColor("#5A6A6A"),
    "paper": colors.white,
}

#: Matplotlib wants strings; the same palette drives the figures.
HEX = {name: value.hexval().replace("0x", "#")[:7] for name, value in PALETTE.items()}

#: A change column is recognised by this prefix and rendered in rise/fall
#: colour. Table builders elsewhere name their columns with it.
DELTA = "Δ"

FONT, FONT_BOLD, FONT_ITALIC = "Bulletin", "Bulletin-Bold", "Bulletin-Italic"


def register_fonts() -> None:
    """Register a Unicode family. Uzbek Latin needs oʻ/gʻ and the ² sign."""
    if FONT in pdfmetrics.getRegisteredFontNames():
        return
    import matplotlib
    ttf = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in ((FONT, "DejaVuSans.ttf"), (FONT_BOLD, "DejaVuSans-Bold.ttf"),
                       (FONT_ITALIC, "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(ttf / file)))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_BOLD, italic=FONT_ITALIC)


def stylesheet() -> dict:
    """Paragraph styles for every kind of text the bulletin sets."""
    register_fonts()
    base = getSampleStyleSheet()
    styles = {}

    def style(name, parent="BodyText", **kw):
        kw.setdefault("fontName", FONT)
        styles[name] = ParagraphStyle(name, parent=base[parent], **kw)
        return styles[name]

    style("body", fontSize=8.6, leading=12.2, textColor=PALETTE["ink"],
          alignment=TA_JUSTIFY, spaceAfter=5)
    style("bullet", fontSize=8.6, leading=12.2, textColor=PALETTE["ink"],
          alignment=TA_JUSTIFY, leftIndent=13, bulletIndent=1, spaceAfter=4,
          bulletFontName=FONT, bulletFontSize=5.5, bulletColor=PALETTE["primary"],
          bulletOffsetY=-2)
    style("subbullet", fontSize=8.4, leading=11.6, textColor=PALETTE["ink"],
          alignment=TA_JUSTIFY, leftIndent=25, bulletIndent=14, spaceAfter=3,
          bulletFontName=FONT, bulletFontSize=8, bulletColor=PALETTE["band"])
    # Centred caps in teal, as every section of the reference opens.
    style("heading", fontSize=12.5, leading=16, textColor=PALETTE["primary"],
          alignment=TA_CENTER, fontName=FONT_BOLD, spaceBefore=2, spaceAfter=7)
    # Same appearance as a section heading, but not collected into the
    # contents page: the contents page's own title must not list itself.
    style("heading_plain", fontSize=12.5, leading=16, textColor=PALETTE["primary"],
          alignment=TA_CENTER, fontName=FONT_BOLD, spaceBefore=2, spaceAfter=7)
    style("subheading", fontSize=9.6, leading=13, textColor=PALETTE["deep"],
          alignment=TA_CENTER, fontName=FONT_BOLD, spaceBefore=6, spaceAfter=4)
    style("tablehead", fontSize=8, leading=10.5, textColor=PALETTE["paper"],
          fontName=FONT_BOLD, alignment=TA_CENTER)
    style("cell", fontSize=8, leading=10.5, textColor=PALETTE["ink"])
    style("cellnum", fontSize=8, leading=10.5, textColor=PALETTE["ink"],
          alignment=TA_RIGHT)
    style("caption", fontSize=8.4, leading=11, textColor=PALETTE["deep"],
          fontName=FONT_BOLD, alignment=TA_CENTER, spaceBefore=2, spaceAfter=4)
    style("note", fontSize=7.8, leading=10.6, textColor=PALETTE["ink"],
          alignment=TA_JUSTIFY, spaceAfter=0)
    style("source", fontSize=7.4, leading=9.6, textColor=PALETTE["muted"],
          spaceBefore=2, spaceAfter=6)
    style("toc", fontSize=9.2, leading=15, textColor=PALETTE["ink"])
    style("toc_page", fontSize=9.2, leading=15, textColor=PALETTE["primary"],
          fontName=FONT_BOLD, alignment=TA_RIGHT)
    style("abbrev", fontSize=7.8, leading=11, textColor=PALETTE["ink"])
    return styles


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

def fmt(value, decimals=2) -> str:
    """One way of writing a number, so no two tables disagree about it."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return "—"
    if isinstance(value, (int,)) or (isinstance(value, float) and float(value).is_integer()
                                     and abs(value) >= 1000):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, float):
        return f"{value:,.{decimals}f}".replace(",", " ")
    return str(value)


def fmt_change(value) -> tuple[str, colors.Color]:
    """A signed percentage and the colour its direction is drawn in."""
    if value is None or pd.isna(value):
        return "—", PALETTE["flat"]
    if value > 0:
        return f"+{value:.1f}%", PALETTE["rise"]
    if value < 0:
        return f"{value:.1f}%", PALETTE["fall"]
    return "0.0%", PALETTE["flat"]


# --------------------------------------------------------------------------
# flowables
# --------------------------------------------------------------------------

def heading(text, styles, level=1, toc=True):
    """A section title. Level 1 entries are the ones the contents page lists."""
    name = ("heading" if toc else "heading_plain") if level == 1 else "subheading"
    return Paragraph(escape(str(text)), styles[name])


def wrap_title(title, width=26):
    """Break the cover title into lines short enough to set at 23pt."""
    lines, current = [], ""
    for word in str(title).split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:4]


def split_bold(text):
    """``**bold**`` markup as (text, bold) pairs, for a format without markup."""
    return [(part, index % 2 == 1) for index, part in enumerate(str(text).split("**"))
            if part]


def abbreviations_table(entries, styles):
    """The reference opens with a short glossary; this is the same idea."""
    rows = [[Paragraph(f"<b>{escape(short)}</b>", styles["abbrev"]),
             Paragraph(escape(long), styles["abbrev"])] for short, long in entries]
    table = Table(rows, colWidths=[70, CONTENT_WIDTH - 70], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def body(text, styles, name="body"):
    return Paragraph(escape(str(text)) if "<" not in str(text) else str(text), styles[name])


def bullets(items, styles, level=1):
    """Commentary as the reference sets it: a small teal marker per point.

    ``items`` may hold plain strings or ``(text, sub-items)`` pairs; a nested
    list is indented and marked with a dash, as the reference's second level is.
    """
    out, style = [], styles["bullet" if level == 1 else "subbullet"]
    marker = "■" if level == 1 else "–"
    for item in items:
        text, children = item if isinstance(item, tuple) else (item, ())
        out.append(Paragraph(_inline(text), style, bulletText=marker))
        if children:
            out.extend(bullets(children, styles, level + 1))
    return out


def _inline(text) -> str:
    """Allow **bold** in authored text without hand-writing markup everywhere."""
    text = escape(str(text))
    parts = text.split("**")
    return "".join(part if index % 2 == 0 else f"<b>{part}</b>"
                   for index, part in enumerate(parts))


def callout(label, paragraphs, styles, tone="source", width=None):
    """A tinted box, as the reference uses for its 'Manba' and 'Eslatma' notes."""
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    flow = []
    for index, text in enumerate(paragraphs):
        prefix = f"<b>{escape(label)}:</b> " if index == 0 and label else ""
        flow.append(Paragraph(prefix + _inline(text), styles["note"]))
        if index + 1 < len(paragraphs):
            flow.append(Spacer(1, 4))
    fill = PALETTE["tint"] if tone == "source" else PALETTE["pale"]
    table = Table([[flow]], colWidths=[width or CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, PALETTE["primary"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def data_table(frame: pd.DataFrame, styles, *, width=None, first_column_width=None):
    """A banded table; change columns are coloured by direction, as the reference does."""
    width = width or CONTENT_WIDTH
    columns = list(frame.columns)
    delta = [index for index, name in enumerate(columns) if str(name).startswith(DELTA)]
    numeric = [index for index, name in enumerate(columns)
               if index in delta or pd.api.types.is_numeric_dtype(frame[name])]
    header = [Paragraph(escape(str(name)), styles["tablehead"]) for name in columns]
    rows = [header]
    for _, record in frame.iterrows():
        cells = []
        for index, name in enumerate(columns):
            value = record[name]
            if index in delta:
                text, colour = fmt_change(value)
                cells.append(Paragraph(f'<font color="{colour.hexval().replace("0x", "#")[:7]}">'
                                       f"<b>{text}</b></font>", styles["cellnum"]))
            elif index in numeric:
                cells.append(Paragraph(escape(fmt(value)), styles["cellnum"]))
            else:
                cells.append(Paragraph(escape("—" if pd.isna(value) else str(value)),
                                       styles["cell"]))
        rows.append(cells)
    label_width = first_column_width or min(0.30 * width, max(0.22 * width, width / len(columns) * 1.5))
    rest = (width - label_width) / max(len(columns) - 1, 1)
    widths = [label_width] + [rest] * (len(columns) - 1)
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALETTE["band"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALETTE["paper"], PALETTE["pale"]]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, PALETTE["deep"]),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, PALETTE["tint_strong"]),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, PALETTE["band"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return table


def two_column(left, right, styles, ratio=0.44):
    """Commentary beside its table, the reference's standard page unit."""
    left_width = CONTENT_WIDTH * ratio
    table = Table([[left, right]], colWidths=[left_width, CONTENT_WIDTH - left_width])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, -1), 0), ("RIGHTPADDING", (0, 0), (0, -1), 10),
        ("LEFTPADDING", (1, 0), (1, -1), 0), ("RIGHTPADDING", (1, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def contents_table(entries, styles):
    """A contents list with dot leaders and the page each section starts on."""
    rows = [[Paragraph(escape(title), styles["toc"]),
             Paragraph(str(page), styles["toc_page"])] for title, page in entries]
    table = Table(rows, colWidths=[CONTENT_WIDTH - 40, 40], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, PALETTE["tint_strong"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------

def draw_cover(canvas, title_lines, period, strapline, footnote):
    """The opening page: teal bands, the title, and the period it covers."""
    register_fonts()
    width, height = PAGE
    canvas.saveState()
    canvas.setFillColor(PALETTE["tint"])
    canvas.rect(0, 0, width, height, stroke=0, fill=1)

    def polygon(points, colour, alpha=1.0):
        path = canvas.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        path.close()
        canvas.setFillColor(colour)
        canvas.setFillAlpha(alpha)
        canvas.drawPath(path, stroke=0, fill=1)
        canvas.setFillAlpha(1)

    # A teal masthead with a diagonal foot, and a darker band crossing it —
    # the reference cover's geometry, without its photograph.
    polygon([(0, height), (width, height), (width, height * 0.66), (0, height * 0.58)],
            PALETTE["band"])
    polygon([(0, height * 0.62), (width, height * 0.78), (width, height * 0.72),
             (0, height * 0.56)], PALETTE["deep"], alpha=0.95)

    # A plain roofline mark rather than a borrowed emblem.
    canvas.setStrokeColor(PALETTE["paper"])
    canvas.setLineWidth(2.4)
    cx, cy = width / 2, height * 0.88
    canvas.line(cx - 34, cy, cx, cy + 22)
    canvas.line(cx, cy + 22, cx + 34, cy)
    for offset in (-22, -8, 6, 20):
        canvas.line(cx + offset, cy - 4, cx + offset, cy - 26)
    canvas.line(cx - 36, cy - 31, cx + 36, cy - 31)

    canvas.setFillColor(PALETTE["paper"])
    canvas.setFont(FONT_BOLD, 10.5)
    canvas.drawCentredString(cx, height * 0.815, strapline)

    # Title block, bottom-left as in the reference.
    canvas.setFillColor(PALETTE["deep"])
    y = height * 0.40
    canvas.setFont(FONT_BOLD, 23)
    for line in title_lines:
        canvas.drawString(MARGIN, y, line.upper())
        y -= 30
    canvas.setFillColor(PALETTE["deep"])
    canvas.rect(MARGIN - 6, y - 16, CONTENT_WIDTH * 0.78, 40, stroke=0, fill=1)
    canvas.setFillColor(PALETTE["paper"])
    canvas.setFont(FONT_BOLD, 21)
    canvas.drawString(MARGIN + 4, y - 4, period)

    canvas.setFillColor(PALETTE["deep"])
    canvas.setFont(FONT, 8.4)
    text = canvas.beginText(MARGIN, 46)
    for line in footnote:
        text.textLine(line)
    canvas.drawText(text)
    canvas.restoreState()


def draw_furniture(canvas, doc, footer_left):
    """Rule, footer note and the page number set in a filled circle."""
    register_fonts()
    width = PAGE[0]
    canvas.saveState()
    canvas.setStrokeColor(PALETTE["primary"])
    canvas.setLineWidth(1.1)
    canvas.line(MARGIN, 34, width - MARGIN, 34)
    canvas.setFont(FONT, 7)
    canvas.setFillColor(PALETTE["muted"])
    canvas.drawString(MARGIN, 40, footer_left)
    canvas.setFillColor(PALETTE["primary"])
    canvas.circle(width / 2, 22, 9, stroke=0, fill=1)
    canvas.setFillColor(PALETTE["paper"])
    canvas.setFont(FONT_BOLD, 8)
    canvas.drawCentredString(width / 2, 19, str(canvas.getPageNumber() - 1))
    canvas.restoreState()


class BulletinDoc(BaseDocTemplate):
    """A cover page plus body pages, with headings collected for the contents.

    Page numbers on the contents page come from reportlab's two-pass build, so
    they are the pages the sections actually land on rather than a guess.
    """

    def __init__(self, path, *, cover, footer, **kw):
        super().__init__(str(path), pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN + 12, **kw)
        frame = Frame(MARGIN, MARGIN + 12, CONTENT_WIDTH,
                      PAGE[1] - 2 * MARGIN - 12, id="body")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame],
                         onPage=lambda c, d: draw_cover(c, **cover)),
            PageTemplate(id="body", frames=[frame],
                         onPage=lambda c, d: draw_furniture(c, d, footer)),
        ])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "heading":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page - 1))


def table_of_contents(styles) -> TableOfContents:
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("tocentry", parent=styles["toc"], fontName=FONT,
                                      leftIndent=6, firstLineIndent=-6,
                                      spaceBefore=3, leading=15)]
    return toc


# --------------------------------------------------------------------------
# matplotlib
# --------------------------------------------------------------------------

def figure_style() -> dict:
    """rcParams putting the figures in the same palette as the pages."""
    return {
        "font.size": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": HEX["deep"], "axes.labelcolor": HEX["muted"],
        "axes.edgecolor": HEX["tint_strong"], "axes.facecolor": "white",
        "figure.facecolor": "white", "text.color": HEX["ink"],
        "xtick.color": HEX["muted"], "ytick.color": HEX["muted"],
        "grid.color": HEX["tint_strong"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    }


def label_bars(ax, bars, values, *, decimals=1, horizontal=False):
    """Print each bar's value, so no figure depends on reading its length."""
    for bar, value in zip(bars, values):
        if value is None or pd.isna(value):
            continue
        text = f"{value:,.{decimals}f}".replace(",", " ")
        if horizontal:
            ax.annotate(text, (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=HEX["ink"])
        else:
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=HEX["ink"])


# --------------------------------------------------------------------------
# Word
# --------------------------------------------------------------------------
# The Word file carries the same palette and the same banded tables, so the
# two formats are recognisably one report. python-docx has no API for cell
# shading, so the one piece of raw XML the format needs lives here.

def _hex(name: str) -> str:
    return HEX[name].lstrip("#").upper()


def docx_shade(cell, colour: str) -> None:
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:fill"), colour)
    cell._tc.get_or_add_tcPr().append(shading)


def docx_runs(paragraph, pieces):
    """Write (text, bold) pairs into a paragraph."""
    for text, bold in pieces:
        run = paragraph.add_run(text)
        run.bold = bold
    return paragraph


def docx_table(document, frame):
    """A banded table whose change columns are coloured by direction."""
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.shared import Pt, RGBColor
    columns = list(frame.columns)
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    for cell, name in zip(table.rows[0].cells, columns):
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(name))
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor.from_string("FFFFFF")
        docx_shade(cell, _hex("band"))
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for index, (_, record) in enumerate(frame.iterrows()):
        cells = table.add_row().cells
        for cell, name in zip(cells, columns):
            value = record[name]
            run = cell.paragraphs[0].add_run()
            run.font.size = Pt(8.5)
            if str(name).startswith(DELTA):
                text, colour = fmt_change(value)
                run.text = text
                run.bold = True
                run.font.color.rgb = RGBColor.from_string(
                    colour.hexval().replace("0x", "")[:6].upper())
            elif pd.api.types.is_number(value):
                run.text = fmt(value)
            else:
                run.text = "—" if pd.isna(value) else str(value)
            if index % 2:
                docx_shade(cell, _hex("pale"))
    return table


def docx_callout(document, label, paragraphs):
    """A tinted single-cell box, matching the PDF's source and note callouts."""
    from docx.shared import Pt
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    table = document.add_table(rows=1, cols=1)
    cell = table.rows[0].cells[0]
    docx_shade(cell, _hex("tint"))
    cell.text = ""
    for index, text in enumerate(paragraphs):
        paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
        if index == 0 and label:
            run = paragraph.add_run(f"{label}: ")
            run.bold = True
            run.font.size = Pt(8.5)
        for piece, bold in split_bold(text):
            run = paragraph.add_run(piece)
            run.bold = bold
            run.font.size = Pt(8.5)
    return table
