"""Build the submission PDF: repo link, app link, BITS Lab screenshot, README content.

usage (needs reportlab): python tools/make_submission_pdf.py README OUT [--repo URL] [--app URL]
       [--screenshot PNG] [--screenshot-caption TEXT] [--extra PNG::CAPTION ...] [--bits-id ID]
"""
import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ap = argparse.ArgumentParser()
ap.add_argument("readme"); ap.add_argument("out")
ap.add_argument("--repo", default=""); ap.add_argument("--app", default="")
ap.add_argument("--screenshot", default="")
ap.add_argument("--screenshot-caption", default="")
ap.add_argument("--extra", nargs="*", default=[], help="more screenshots as PATH::CAPTION")
ap.add_argument("--bits-id", default="")
a = ap.parse_args()

ss = getSampleStyleSheet()
ACCENT = colors.HexColor("#1f4e79")
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
cell = ParagraphStyle("c", parent=body, fontSize=8.3, leading=10.5)
cellh = ParagraphStyle("ch", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=ACCENT, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13.5, textColor=ACCENT, spaceBefore=10, spaceAfter=5)
h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
code = ParagraphStyle("code", parent=ss["Code"], fontSize=7.8, leading=9.5, backColor=colors.HexColor("#f3f4f6"),
                      borderPadding=5, leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=8)
title = ParagraphStyle("t", parent=h1, fontSize=22, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle("s", parent=body, alignment=TA_CENTER, fontSize=11, leading=15)
missing = ParagraphStyle("m", parent=body, textColor=colors.HexColor("#b00020"))

REPO = a.repo or "&lt;GitHub repository link to be added&gt;"
APP = a.app or "&lt;Streamlit app link to be added&gt;"


def link(url, fallback):
    return f'<link href="{url}" color="blue"><u>{url}</u></link>' if url else f'<font color="#b00020">{fallback}</font>'


def inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace("&lt;GITHUB_REPO_URL&gt;", a.repo or '<font color="#b00020">[GitHub repository link to be added]</font>')
    t = t.replace("&lt;STREAMLIT_APP_URL&gt;", a.app or '<font color="#b00020">[Streamlit app link to be added]</font>')
    t = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?!\w)", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="blue">\1</link>', t)
    t = re.sub(r'(?<![">])(https?://[^\s<)]+)', r'<link href="\1" color="blue">\1</link>', t)
    return t


def md_table(rows, width):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
    data = [[Paragraph(inline(c), cellh if i == 0 else cell) for c in r] for i, r in enumerate(cells)]
    n = len(cells[0])
    if n == 2:
        first = max(len(r[0]) for r in cells)
        w = [width * (0.28 if first > 12 else 0.2), width * (0.72 if first > 12 else 0.8)]
    elif n == 7:
        w = [width * 0.28] + [width * 0.12] * 6
    else:
        w = [width / n] * n
    t = Table(data, colWidths=w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def readme_flowables(text, width):
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            blk = "\n".join(lines[i + 1:j]).replace("├──", "|--").replace("└──", "`--").replace("│", "|")
            out.append(Preformatted(blk, code)); i = j + 1; continue
        if ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out += [md_table(lines[i:j], width), Spacer(1, 6)]; i = j; continue
        if ln.startswith("# "):
            out.append(Paragraph(inline(ln[2:]), h1))
        elif ln.startswith("## "):
            out.append(Paragraph(inline(ln[3:]), h2))
        elif ln.startswith("### "):
            out.append(Paragraph(inline(ln[4:]), h3))
        elif re.match(r"^\s*[-*] ", ln):
            out.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", ln)), body, bulletText="•"))
        elif re.match(r"^\d+\. ", ln):
            n, rest = ln.split(". ", 1)
            out.append(Paragraph(inline(rest), body, bulletText=f"{n}."))
        elif ln.strip() == "---":
            out.append(Spacer(1, 4))
        elif ln.strip():
            para = [ln.strip()]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not re.match(r"^(#|\||```|\s*[-*] |\d+\. |---)", lines[i + 1])):
                i += 1
                para.append(lines[i].strip())
            out.append(Paragraph(inline(" ".join(para)), body))
        i += 1
    return out


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, "ML Assignment 2 — Credit Card Default Prediction")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="ML Assignment 2 - Credit Card Default Prediction", author="Himanshu Sharma")
W = A4[0] - 4 * cm

story = [
    Spacer(1, 1.5 * cm),
    Paragraph("Machine Learning — Assignment 2", title),
    Paragraph("M.Tech (AIML) · Work Integrated Learning Programmes, BITS Pilani", sub),
    Paragraph("Credit Card Default Prediction with Five Classifiers and a Streamlit App", sub),
    Spacer(1, 0.4 * cm),
    Paragraph("Name: Himanshu Sharma &nbsp;&nbsp;|&nbsp;&nbsp; BITS ID: " + (a.bits_id or "____________________"), sub),
    Spacer(1, 1 * cm),
    Paragraph("1. GitHub Repository Link", h2),
    Paragraph(link(a.repo, "&lt;GitHub repository link to be added&gt;"), body),
    Paragraph("Contains the complete source code, requirements.txt, README.md, test_data.csv and the "
              "model/ folder with all five saved classifiers.", body),
    Paragraph("2. Live Streamlit App Link", h2),
    Paragraph(link(a.app, "&lt;Streamlit Community Cloud link to be added&gt;"), body),
    Paragraph("3. Screenshot — Execution on BITS Virtual Lab", h2),
]
cap = ParagraphStyle("cap", parent=body, fontSize=8.5, textColor=colors.HexColor("#444444"), alignment=TA_CENTER)


def shot(path, caption, max_h):
    img = Image(path)
    scale = min(W / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return KeepTogether([img, Spacer(1, 3), Paragraph(caption, cap), Spacer(1, 10)])


if a.screenshot and Path(a.screenshot).exists():
    story.append(shot(a.screenshot, a.screenshot_caption, 11 * cm))
else:
    box = Table([[Paragraph("BITS Virtual Lab screenshot to be inserted here", missing)]],
                colWidths=[W], rowHeights=[6 * cm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#b00020")),
                             ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(box)

if a.extra:
    story += [PageBreak(), Paragraph("3 (contd.). Additional screenshots — BITS Virtual Lab execution", h2),
              Paragraph("All screenshots below were taken on the BITS Virtual Lab session of user s2026ak05703. "
                        "The remote-desktop session token in the browser address bar has been masked.", body),
              Spacer(1, 6)]
    for n, item in enumerate(a.extra, start=2):
        path, _, caption = item.partition("::")
        story.append(shot(path, f"Figure {n}. {caption}", 10.5 * cm))
story += [PageBreak(), Paragraph("4. GitHub README Content", h2), Spacer(1, 4)]
story += readme_flowables(Path(a.readme).read_text(), W)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", a.out)
