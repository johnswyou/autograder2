"""Write the synthetic fixtures under tests/fixtures/synthetic/. Takes no arguments.

Format facts copied from real Classkick exports (docs/export-notes.md, and the object
comparison in the task 0.3 report):

- A page is 612 x 792 pt and holds exactly one image, the slide: 1004 x 2008 px, RGB,
  FlateDecode, with a DeviceGray FlateDecode /SMask, drawn at x 200..591.56 and
  y 8.88..792 pt (391.56 x 783.12 pt, 184.6 px per inch). Page k is slide k, and
  untouched slides are still pages.
- The canvas outside the printed page is transparent with RGB (0, 0, 0) under it, so
  the raw image without its mask shows black there.
- The printed page fills slide pixels x 51..953, y 51..1219 (903 x 1169 px, about 106
  px per inch of paper). It is opaque, except that its right column and bottom row
  have alpha 128.
- Stylus ink outside the printed page is ink-coloured RGB under partial alpha. The
  measured stylus and typed-text colour is (64, 64, 65).
- A pasted phone photo is fitted to the printed page's width at (51, 50), 3:4 portrait,
  so it ends near y 1254 and leaves the lower canvas transparent.
- Sidebar text is Helvetica (Type1, WinAnsiEncoding, not embedded) starting at x = 15
  pt. A submission has "Student: <name>", "Assignment: <title>", "Page k / N" and
  "Class Code: XXX XXX" at 10 pt on baselines 764.82, 741.70, 718.58 and 695.46, and
  "Printed from Classkick" at 8 pt on 86.256. A blank export has only "Assignment:"
  at 753.26 and "Page k / N" at 730.14, plus the same last line. A long title runs
  past x = 200 pt onto the slide; the fixture title is long enough to do so.
- Producer and Creator are "PDFKit". File names are "<title> - <student name>.pdf",
  where the title itself contains " - ".
- A PDF that is not a Classkick export (phone scanning app) has its own page count
  and one full-page photo per page, here a JPEG at 200 px per inch, with no sidebar.

Not copied: PDFKit's object order and its flipped content-stream matrices (the
effective image matrix and text origins are the same), the extra Info keys reportlab
always writes (Title "untitled", Author "anonymous" and others), and the coarse alpha
levels of real stylus strokes.
"""

import json
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from io import BytesIO
from math import ceil, cos, pi, sin
from pathlib import Path
from typing import NamedTuple

import pypdfium2 as pdfium
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from reportlab import rl_config
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

from redpen.schemas.common import DiagramElement
from redpen.schemas.grade import CriterionGrade, Grade
from redpen.schemas.key import Answer, Criterion, Key, KeyPart, PartialCredit, Policies
from redpen.schemas.reading import Reading
from redpen.schemas.spec import Figure, Part, Spec
from redpen.schemas.submission import IngestPage, Map, MapEntry, SetAside

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"
TITLE = "Practice Set 3 - Forces and Functions"
CLASS_CODE = "FXT 042"

PAGE_W, PAGE_H = 612, 792
SLIDE_W, SLIDE_H = 1004, 2008
SLIDE_BOX = (200, 8.88, 391.56, 783.12)  # x, y, width, height in points
PRINTED_AT, PRINTED_SIZE = (51, 51), (903, 1169)
INK = (64, 64, 65)

rl_config.useA85 = 0  # plain FlateDecode streams, as in the real exports


# --- The assignment ---------------------------------------------------------------

# Prompts are plain text; "^2" is printed as a superscript and " - " as a minus.
MC = {
    "Q1": (
        "A 2.0 kg cart on a level track speeds up from rest to 6.0 m/s in 3.0 s. "
        "What is the magnitude of the net force on the cart?",
        ["1.0 N", "2.0 N", "4.0 N", "6.0 N", "12 N"],
    ),
    "Q2": (
        "If 3(x - 2) = 2x + 5, what is the value of x?",
        ["1", "3", "7", "11", "13"],
    ),
    "Q3": (
        "A ball is thrown straight up, and air resistance is negligible. At the "
        "highest point of its path, which is true of its velocity v and its "
        "acceleration a?",
        [
            "v = 0 and a = 0",
            "v = 0 and a points down",
            "v points up and a = 0",
            "v and a both point up",
            "v points down and a = 0",
        ],
    ),
}
Q4 = (
    "A crate of mass 4.0 kg is pulled to the right across a level floor by a "
    "horizontal rope, as shown below. The tension in the rope is 20 N, and the "
    "coefficient of kinetic friction between the crate and the floor is 0.25. "
    "Use g = 10 m/s^2."
)
Q5 = "Let f(x) = x^2 - 6x + 5."
PARTS = {
    "Q4.a": "On the dot below, draw and label the forces (not components) exerted on "
    "the crate. Draw each force as an arrow starting on the dot.",
    "Q4.b": "Calculate the magnitude of the acceleration of the crate.",
    "Q4.c": "The rope is cut while the crate is still moving. Describe the motion of "
    "the crate after the rope is cut, and explain your answer in terms of the forces "
    "on the crate.",
    "Q5.a": "Write f(x) in the form (x - h)^2 + k by completing the square.",
    "Q5.b": "Find the minimum value of f(x).",
    "Q5.c": "Sketch the graph of y = f(x) on the axes below. Label the vertex and the "
    "x-intercepts.",
}

# Where things are printed, in points from the top-left of the page. A region is
# (page, x0, y0, x1, y1): empty space in which a student answers in place.
OPTION_ROWS = {"Q1": 205, "Q2": 357, "Q3": 509}  # top of option (A); rows 17 pt apart
OPTION_X = 96  # x of "(A)"
# (page, top) of each printed label of a free-response part, and of the
# introductions "4." and "5."
LABEL_TOP = {
    "Q4.a": (2, 238),
    "Q4.b": (2, 478),
    "Q4.c": (3, 62),
    "Q5.a": (3, 582),
    "Q5.b": (4, 62),
    "Q5.c": (4, 318),
}
INTRO_TOP = {"Q4": (2, 62), "Q5": (3, 556)}
PAGE_NUMBER = 756  # baseline of the page number, the lowest ink on every page
REGION = {
    "name": (1, 104, 82, 330, 112),
    "Q1": (1, 290, 196, 540, 292),
    "Q2": (1, 290, 348, 540, 444),
    "Q3": (1, 290, 500, 540, 596),
    "Q4.a": (2, 150, 275, 462, 468),
    "Q4.b": (2, 72, 498, 540, 735),
    "Q4.c": (3, 72, 112, 540, 530),
    "Q5.a": (3, 72, 610, 540, 738),
    "Q5.b": (4, 72, 86, 540, 300),
    "Q5.c": (4, 160, 352, 480, 650),
}
DOT = (306, 372)  # Q4(a)
AXES_ORIGIN, AXES_UNIT = (234, 512), 24  # Q5(c): x from -1 to 7, y from -5 to 6


def answer(**filled: object) -> Answer:
    empty: dict[str, object] = {
        "choices": [],
        "value": None,
        "unit": None,
        "rel_tol": None,
        "accept_also": [],
        "expression": None,
        "elements": [],
        "must_include": [],
    }
    return Answer.model_validate(empty | filled)


def criterion(
    cid: str,
    points: str,
    kind: str,
    requirement: str,
    partial: tuple[str, str] | None = None,
    carry_forward: bool = False,
) -> Criterion:
    levels = (
        [PartialCredit(points=Decimal(partial[0]), when=partial[1])] if partial else []
    )
    return Criterion.model_validate(
        {
            "id": cid,
            "points": Decimal(points),
            "kind": kind,
            "requirement": requirement,
            "partial": levels,
            "carry_forward": carry_forward,
        }
    )


def element(label: str, kind: str, direction: str | None, position: str | None):
    return DiagramElement(
        label=label, kind=kind, direction=direction, position=position
    )


FORCES = [
    element("N", "force", "up", "starts on the dot"),
    element("mg", "force", "down", "starts on the dot"),
    element("T", "force", "right", "starts on the dot"),
    element("f", "force", "left", "starts on the dot"),
]
PARABOLA = [
    element("y = f(x)", "curve", "opens upward", None),
    element("vertex", "point", None, "(3, -4)"),
    element("x-intercept", "point", None, "(1, 0)"),
    element("x-intercept", "point", None, "(5, 0)"),
]


def key() -> Key:
    def mc(pid: str, page: int, correct: str, solution: str, errors: list[str]):
        stem, options = MC[pid]
        listed = " ".join(
            f"({letter}) {o}" for letter, o in zip("ABCDE", options, strict=True)
        )
        return KeyPart(
            id=pid,
            label=pid[1:],
            home_page=page,
            response_type="multiple_choice",
            depends_on=[],
            state="verified",
            prompt=f"{stem} {listed}",
            answer=answer(choices=[correct]),
            solution=solution,
            criteria=[
                criterion(f"{pid}.c1", "1", "final_answer", f"Chooses {correct}.")
            ],
            common_errors=errors,
        )

    def fr(
        pid: str, page: int, rtype: str, ans: Answer, solution: str, criteria, errors
    ):
        stem = Q4 if pid.startswith("Q4") else Q5
        return KeyPart(
            id=pid,
            label=f"{pid[1]}({pid[-1]})",
            home_page=page,
            response_type=rtype,
            depends_on=["Q5.a"] if pid == "Q5.b" else [],
            state="verified",
            prompt=f"{stem} {PARTS[pid]}",
            answer=ans,
            solution=solution,
            criteria=criteria,
            common_errors=errors,
        )

    return Key(
        policies=Policies(),
        parts=[
            mc(
                "Q1",
                1,
                "C",
                "a = (6.0 m/s)/(3.0 s) = 2.0 m/s^2, so F = m a = (2.0 kg)(2.0 m/s^2) "
                "= 4.0 N.",
                [
                    "Uses F = m v and gets 12 N (E).",
                    "Takes the acceleration as the net force and gets 2.0 N (B).",
                ],
            ),
            mc(
                "Q2",
                1,
                "D",
                "3x - 6 = 2x + 5, so x = 11.",
                ["Writes 3x - 2 = 2x + 5 and gets 7 (C)."],
            ),
            mc(
                "Q3",
                1,
                "B",
                "At the top v = 0, but gravity still acts, so a = g downward.",
                ["Thinks a = 0 where v = 0 (A)."],
            ),
            fr(
                "Q4.a",
                2,
                "free_body_diagram",
                answer(elements=FORCES),
                "Four forces: weight mg = 40 N down, normal force N = 40 N up, tension "
                "T = 20 N right, kinetic friction f = 10 N left.",
                [
                    criterion(
                        "Q4.a.c1",
                        "1",
                        "final_answer",
                        "Draws the weight downward and the normal force upward, both "
                        "labelled.",
                    ),
                    criterion(
                        "Q4.a.c2",
                        "1",
                        "final_answer",
                        "Draws the tension to the right and kinetic friction to the "
                        "left, both labelled, and no other forces.",
                        ("0.5", "Only one of the two horizontal forces is drawn."),
                    ),
                ],
                ["Leaves out friction.", "Adds a 'force of motion' to the right."],
            ),
            fr(
                "Q4.b",
                2,
                "numeric",
                answer(value=2.5, unit="m/s^2"),
                "f = mu m g = 0.25 (4.0)(10) = 10 N\nF_net = T - f = 20 - 10 = 10 N\n"
                "a = F_net / m = 10 / 4.0 = 2.5 m/s^2",
                [
                    criterion(
                        "Q4.b.c1",
                        "2",
                        "method",
                        "Applies Newton's second law with net force T - f, where "
                        "f = mu m g = 10 N.",
                        (
                            "1",
                            "Applies Newton's second law but leaves friction out of "
                            "the net force (a = T/m).",
                        ),
                    ),
                    criterion(
                        "Q4.b.c2",
                        "1",
                        "final_answer",
                        "States a = 2.5 m/s^2 with units.",
                    ),
                ],
                ["Leaves out friction and gets 5.0 m/s^2."],
            ),
            fr(
                "Q4.c",
                3,
                "explanation",
                answer(
                    must_include=[
                        "The crate slows down and stops.",
                        "After the rope is cut, kinetic friction is the only "
                        "horizontal force.",
                        "Friction opposes the motion, so the net force points "
                        "opposite to the velocity.",
                    ]
                ),
                "With the rope gone the only horizontal force is kinetic friction, "
                "10 N opposite to the velocity, so the crate slows at 2.5 m/s^2 until "
                "it stops.",
                [
                    criterion(
                        "Q4.c.c1",
                        "1",
                        "final_answer",
                        "States that the crate slows down.",
                    ),
                    criterion(
                        "Q4.c.c2",
                        "1",
                        "method",
                        "Explains that friction is then the only horizontal force and "
                        "that it opposes the motion.",
                    ),
                ],
                ["Says the crate keeps moving at constant speed (forgets friction)."],
            ),
            fr(
                "Q5.a",
                3,
                "symbolic",
                answer(expression="(x - 3)^2 - 4"),
                "x^2 - 6x + 5 = (x^2 - 6x + 9) - 9 + 5 = (x - 3)^2 - 4",
                [
                    criterion(
                        "Q5.a.c1",
                        "2",
                        "final_answer",
                        "Writes f(x) = (x - 3)^2 - 4.",
                        ("1", "Correct squared term (x - 3)^2 with a wrong constant."),
                    )
                ],
                ["Adds 9 instead of subtracting it and gets (x - 3)^2 + 4."],
            ),
            fr(
                "Q5.b",
                4,
                "numeric",
                answer(value=-4.0),
                "From (x - 3)^2 - 4 the minimum is at x = 3, and f(3) = -4.",
                [
                    criterion(
                        "Q5.b.c1",
                        "1",
                        "method",
                        "Locates the minimum at x = 3 (from the vertex form, x = -b/2a "
                        "or f'(x) = 0) and evaluates f there.",
                    ),
                    criterion(
                        "Q5.b.c2",
                        "1",
                        "final_answer",
                        "States the minimum value -4.",
                        carry_forward=True,
                    ),
                ],
                ["Gives the x-coordinate 3 instead of the minimum value."],
            ),
            fr(
                "Q5.c",
                4,
                "graph",
                answer(elements=PARABOLA),
                "An upward parabola with vertex (3, -4), x-intercepts 1 and 5 and "
                "y-intercept 5.",
                [
                    criterion(
                        "Q5.c.c1",
                        "1",
                        "final_answer",
                        "Draws an upward-opening parabola with its vertex at (3, -4), "
                        "labelled.",
                    ),
                    criterion(
                        "Q5.c.c2",
                        "1",
                        "final_answer",
                        "Marks the x-intercepts at x = 1 and x = 5.",
                    ),
                ],
                ["Puts the vertex at (-3, -4)."],
            ),
        ],
    )


# --- The blank as a vector PDF ----------------------------------------------------

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=14.5)


def markup(text: str) -> str:
    return re.sub(r"\^(\w)", r"<super>\1</super>", text.replace(" - ", " &#8211; "))


def para(c: Canvas, x: float, top: float, text: str, bold: str = "") -> float:
    """Draw a wrapped paragraph whose first line starts at `top`; return its bottom."""
    if bold:
        c.setFont("Times-Bold", 11)
        c.drawString(x, PAGE_H - top - 11, bold)
        x += 18 if len(bold) < 4 else 24
    p = Paragraph(markup(text), BODY)
    _, height = p.wrap(540 - x, PAGE_H)
    p.drawOn(c, x, PAGE_H - top - height)
    return top + height


def header(c: Canvas, page: int) -> None:
    c.setFont("Helvetica", 8)
    c.setFillGray(0.35)
    c.drawString(72, PAGE_H - 40, "PRACTICE SET 3  ·  FORCES AND FUNCTIONS")
    c.setFillGray(0)
    c.setFont("Times-Roman", 10)
    c.drawCentredString(PAGE_W / 2, PAGE_H - PAGE_NUMBER, str(page))


def stem_top(pid: str) -> float:
    """Top of multiple-choice question `pid`'s label and stem."""
    lines = 2 if len(MC[pid][0]) > 85 else 1
    return OPTION_ROWS[pid] - 6 - 14.5 * lines - 4


def crate_floor() -> float:
    """Top of the floor line in question 4's figure, below the introduction."""
    _, height = Paragraph(markup(Q4), BODY).wrap(540 - 90, PAGE_H)
    return INTRO_TOP["Q4"][1] + height + 92


def arrow(c: Canvas, x0: float, y0: float, x1: float, y1: float) -> None:
    """A line with a filled head at (x1, y1); y measured from the page top."""
    c.line(x0, PAGE_H - y0, x1, PAGE_H - y1)
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    head = c.beginPath()
    head.moveTo(x1, PAGE_H - y1)
    head.lineTo(x1 - 7 * ux - 3 * uy, PAGE_H - (y1 - 7 * uy + 3 * ux))
    head.lineTo(x1 - 7 * ux + 3 * uy, PAGE_H - (y1 - 7 * uy - 3 * ux))
    head.close()
    c.drawPath(head, fill=1, stroke=0)


def draw_blank(path: Path) -> None:
    c = Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1, pageCompression=1)
    c.setTitle(TITLE)

    header(c, 1)
    c.setFont("Times-Bold", 16)
    c.drawString(72, PAGE_H - 76, "Practice Set 3: Forces and Functions")
    c.setFont("Times-Roman", 11)
    c.drawString(72, PAGE_H - 106, "Name:")
    c.line(104, PAGE_H - 108, 300, PAGE_H - 108)
    c.setFont("Times-Italic", 11)
    c.drawString(72, PAGE_H - 140, "Questions 1–3. Choose the one best answer, A to E.")
    for pid, (stem, options) in MC.items():
        top = OPTION_ROWS[pid]
        para(c, 72, stem_top(pid), stem, bold=f"{pid[1]}.")
        for row, (letter, option) in enumerate(zip("ABCDE", options, strict=True)):
            para(c, OPTION_X, top + 17 * row, f"({letter})  {option}")
    c.showPage()

    header(c, 2)
    para(c, 72, INTRO_TOP["Q4"][1], Q4, bold="4.")
    floor = crate_floor()  # the figure: crate on a hatched floor, rope to the right
    c.line(196, PAGE_H - floor, 416, PAGE_H - floor)
    for x in range(200, 416, 10):
        c.line(x, PAGE_H - floor - 1, x - 6, PAGE_H - floor - 8)
    c.rect(236, PAGE_H - floor, 60, 44)
    c.setFont("Times-Roman", 10)
    c.drawCentredString(266, PAGE_H - floor + 18, "4.0 kg")
    arrow(c, 296, floor - 22, 384, floor - 22)
    c.setFont("Times-Italic", 10)
    c.drawString(330, PAGE_H - floor + 28, "rope")
    para(c, 72, LABEL_TOP["Q4.a"][1], PARTS["Q4.a"], bold="(a)")
    c.circle(DOT[0], PAGE_H - DOT[1], 2.6, fill=1)
    para(c, 72, LABEL_TOP["Q4.b"][1], PARTS["Q4.b"], bold="(b)")
    c.showPage()

    header(c, 3)
    para(c, 72, LABEL_TOP["Q4.c"][1], PARTS["Q4.c"], bold="(c)")
    para(c, 72, INTRO_TOP["Q5"][1], Q5, bold="5.")
    para(c, 72, LABEL_TOP["Q5.a"][1], PARTS["Q5.a"], bold="(a)")
    c.showPage()

    header(c, 4)
    para(c, 72, LABEL_TOP["Q5.b"][1], PARTS["Q5.b"], bold="(b)")
    para(c, 72, LABEL_TOP["Q5.c"][1], PARTS["Q5.c"], bold="(c)")
    ox, oy = AXES_ORIGIN
    u = AXES_UNIT
    c.setStrokeGray(0.8)
    c.setLineWidth(0.4)
    for i in range(-1, 8):
        c.line(ox + i * u, PAGE_H - (oy - 6 * u), ox + i * u, PAGE_H - (oy + 5 * u))
    for j in range(-5, 7):
        c.line(ox - u, PAGE_H - (oy - j * u), ox + 7 * u, PAGE_H - (oy - j * u))
    c.setStrokeGray(0)
    c.setLineWidth(0.9)
    arrow(c, ox - u - 6, oy, ox + 7 * u + 10, oy)
    arrow(c, ox, oy + 5 * u + 6, ox, oy - 6 * u - 10)
    c.setFont("Times-Roman", 8)
    for i in range(-1, 8):
        if i:
            c.drawCentredString(ox + i * u, PAGE_H - oy - 10, str(i))
    for j in range(-5, 7):
        if j:
            c.drawRightString(ox - 4, PAGE_H - (oy - j * u) - 3, str(j))
    c.setFont("Times-Italic", 10)
    c.drawString(ox + 7 * u + 12, PAGE_H - oy - 4, "x")
    c.drawString(ox + 4, PAGE_H - (oy - 6 * u - 12), "y")
    c.showPage()
    c.save()


def spec() -> Spec:
    """The spec that parsing blank.pdf should give: the key's ids, labels, types,
    home pages, prompts and dependencies, options A to E on the multiple-choice
    questions, and bands and figure boxes as fractions of blank.pdf's pages. A band
    runs from a part's label to the next label or introduction on its page, or to
    the page number. Question 4's figure belongs to all three of its parts; the dot
    and the axes are answer space, not figures."""
    floor = crate_floor()
    figure = Figure(
        page=2,
        box=[196 / PAGE_W, (floor - 46) / PAGE_H, 416 / PAGE_W, (floor + 9) / PAGE_H],
    )
    tops = {pid: (1, stem_top(pid)) for pid in MC} | LABEL_TOP
    starts = [*tops.values(), *INTRO_TOP.values()]
    parts = []
    for part in KEY.parts:
        page, top = tops[part.id]
        below = [t for p, t in starts if p == page and t > top]
        parts.append(
            Part(
                id=part.id,
                label=part.label,
                prompt=part.prompt,
                response_type=part.response_type,
                home_page=part.home_page,
                band=[top / PAGE_H, min(below, default=PAGE_NUMBER) / PAGE_H],
                figures=[figure] if part.id.startswith("Q4") else [],
                depends_on=part.depends_on,
                options=list("ABCDE") if part.id in MC else [],
            )
        )
    return Spec(parts=parts)


# --- Slides and Classkick-style PDFs ----------------------------------------------


def render_pages(path: Path, size: tuple[int, int]) -> list[Image.Image]:
    """The printed pages at `size` pixels."""
    pdf = pdfium.PdfDocument(path)
    scale = 2 * size[1] / PAGE_H
    pages = [
        page.render(scale=scale).to_pil().convert("RGB").resize(size, Image.LANCZOS)
        for page in pdf
    ]
    pdf.close()
    return pages


def blank_slide(printed: Image.Image) -> Image.Image:
    """A slide with the printed page on a transparent (0, 0, 0, 0) canvas."""
    page = printed.convert("RGBA")
    w, h = PRINTED_SIZE
    alpha = Image.new("L", PRINTED_SIZE, 255)
    draw = ImageDraw.Draw(alpha)
    draw.line([(w - 1, 0), (w - 1, h - 1)], fill=128)
    draw.line([(0, h - 1), (w - 1, h - 1)], fill=128)
    page.putalpha(alpha)
    slide = Image.new("RGBA", (SLIDE_W, SLIDE_H), (0, 0, 0, 0))
    slide.paste(page, PRINTED_AT)
    return slide


def write_classkick(path: Path, slides: list[Image.Image], student: str | None) -> None:
    c = Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1, pageCompression=1)
    c.setProducer("PDFKit")
    c.setCreator("PDFKit")
    n = len(slides)
    for k, slide in enumerate(slides, 1):
        c.drawImage(ImageReader(slide), *SLIDE_BOX, mask="auto")
        if student is None:
            lines = [(753.26, f"Assignment: {TITLE}"), (730.14, f"Page {k} / {n}")]
        else:
            lines = [
                (764.82, f"Student: {student}"),
                (741.70, f"Assignment: {TITLE}"),
                (718.58, f"Page {k} / {n}"),
                (695.46, f"Class Code: {CLASS_CODE}"),
            ]
        c.setFont("Helvetica", 10)
        for y, text in lines:
            c.drawString(15, y, text)
        c.setFont("Helvetica", 8)
        c.drawString(15, 86.256, "Printed from Classkick")
        c.showPage()
    c.save()


def write_scan(path: Path, pages: list[Image.Image]) -> None:
    """A phone-scanner PDF: one full-page JPEG per page, no text."""
    c = Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1, pageCompression=1)
    for page in pages:
        jpeg = BytesIO()
        page.save(jpeg, "JPEG", quality=60)
        jpeg.seek(0)
        c.drawImage(ImageReader(jpeg), 0, 0, PAGE_W, PAGE_H)
        c.showPage()
    c.save()


# --- Drawing student work ---------------------------------------------------------


def font_file(family: str) -> str:
    # fc-match falls back to a plain font when the family is missing; the per-glyph
    # jitter below then does the work of making it look handwritten.
    found = subprocess.run(["fc-match", "-f", "%{file}", family], capture_output=True)
    return found.stdout.decode()


HAND, HAND_BOLD, SANS = (
    font_file(f) for f in ("Comic Neue", "Comic Neue:bold", "Nimbus Sans")
)


@cache
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@dataclass(frozen=True)
class Pen:
    size: float  # letter size, in points of the printed page
    spacing: float  # line spacing, in points
    width: float  # stroke width, in points
    color: tuple[int, int, int]
    typed: bool = False
    bold: bool = False


STYLUS = Pen(21, 31, 2.0, INK, bold=True)
BLUE_STYLUS = Pen(21, 31, 2.0, (30, 70, 190), bold=True)
KEYBOARD = Pen(14, 21, 2.0, INK, typed=True)
PEN = Pen(15, 22, 1.2, (30, 40, 105))
SMALL_PEN = Pen(9.5, 13.5, 0.8, (30, 40, 105))
HIGHLIGHT = (255, 226, 0)
SLIDE_SCALE = PRINTED_SIZE[1] / PAGE_H  # slide pixels per point of the printed page
SCAN_SCALE = 200 / 72
PHOTO_AT = (51, 50)
TURNED_PAGE = 3  # the rotated class's page photographed sideways

# A block of work as it ends up on a page: part id (or "name"), role, basis, pixel box.
Placed = tuple[str, str, str, tuple[int, int, int, int]]


class Sheet:
    """An image a student works on, with printed-page points mapped to its pixels."""

    def __init__(self, image: Image.Image, scale: float, offset=(0, 0)) -> None:
        self.image, self.scale, self.offset = image, scale, offset
        self.blocks: list[tuple[str, str, str, Image.Image]] = []

    def px(self, x: float, y: float) -> tuple[float, float]:
        return self.offset[0] + x * self.scale, self.offset[1] + y * self.scale

    def mask(self) -> Image.Image:
        return Image.new("L", self.image.size)

    def put(self, part, mask, color, region=None, role="work", basis="position"):
        """Lay one block of work on the sheet and remember where it went."""
        if region:  # work done in place must stay in the empty space it belongs to
            _, x0, y0, x1, y1 = REGION[region]
            (left, top), (right, bottom) = self.px(x0, y0), self.px(x1, y1)
            box = mask.getbbox()
            assert box, part
            inside = box[0] >= left - 2 and box[1] >= top - 2
            assert inside and box[2] <= right + 2 and box[3] <= bottom + 2, part
        if self.image.mode == "RGBA":
            layer = Image.new("RGBA", self.image.size, color)
            layer.putalpha(mask)
            self.image.alpha_composite(layer)
        else:
            self.image.paste(color, (0, 0), mask)
        self.blocks.append((part, role, basis, mask))

    def placed(self, move=lambda mask: mask, shift=(0, 0)) -> list[Placed]:
        out = []
        for part, role, basis, mask in self.blocks:
            left, top, right, bottom = move(mask).getbbox()
            box = (left + shift[0], top + shift[1], right + shift[0], bottom + shift[1])
            out.append((part, role, basis, box))
        return out


def write(sheet: Sheet, mask, x: float, y: float, text: str, pen: Pen, rng) -> None:
    """One line of text with its top-left at (x, y) points."""
    left, top = sheet.px(x, y)
    size = pen.size * sheet.scale
    if pen.typed:
        typeface = font(SANS, round(size))
        ImageDraw.Draw(mask).text((left, top), text, font=typeface, fill=255)
        return
    slope, start = rng.uniform(-0.03, 0.03), left
    for ch in text:
        typeface = font(
            HAND_BOLD if pen.bold else HAND, round(size * rng.uniform(0.9, 1.08))
        )
        advance = typeface.getlength(ch)
        if not ch.isspace():
            ascent, descent = typeface.getmetrics()
            glyph = Image.new("L", (round(advance) + 8, ascent + descent + 8))
            ImageDraw.Draw(glyph).text((4, 4), ch, font=typeface, fill=255)
            turned = glyph.rotate(
                rng.uniform(-7, 7), Image.Resampling.BICUBIC, expand=True
            )
            gx = left - 4 - (turned.width - glyph.width) / 2
            gy = top - 4 - (turned.height - glyph.height) / 2 + slope * (left - start)
            mask.paste(
                255, (round(gx), round(gy + rng.uniform(-0.05, 0.05) * size)), turned
            )
        left += advance * rng.uniform(0.97, 1.1)


def write_lines(sheet, mask, x, y, lines, pen, rng) -> None:
    for i, line in enumerate(lines):
        write(sheet, mask, x + rng.uniform(-3, 3), y + i * pen.spacing, line, pen, rng)


def stroke(sheet: Sheet, mask, points, pen: Pen) -> None:
    line = Image.new("L", mask.size)
    width = max(1, round(pen.width * sheet.scale))
    xy = [sheet.px(x, y) for x, y in points]
    ImageDraw.Draw(line).line(xy, fill=255, width=width, joint="curve")
    mask.paste(255, (0, 0), line.filter(ImageFilter.GaussianBlur(0.6)))


def dot(sheet: Sheet, mask, x: float, y: float, pen: Pen) -> None:
    (cx, cy), r = sheet.px(x, y), pen.width * sheet.scale * 1.8
    ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)


def loop(sheet, mask, cx, cy, rx, ry, pen, rng) -> None:
    """A hand-drawn ellipse whose ends overlap."""
    phase = rng.uniform(0, 2 * pi)
    turns = [phase + i * 0.15 for i in range(48)]
    wobble = [1 + 0.07 * sin(3 * t + phase) for t in turns]
    points = [
        (cx + rx * w * cos(t), cy + ry * w * sin(t))
        for t, w in zip(turns, wobble, strict=True)
    ]
    stroke(sheet, mask, points, pen)


def ink_arrow(sheet, mask, x0, y0, x1, y1, pen) -> None:
    stroke(sheet, mask, [(x0, y0), (x1, y1)], pen)
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    ux, uy, head = (x1 - x0) / length, (y1 - y0) / length, min(7.0, length / 2)
    barbs = [
        (x1 - head * ux + 4 * uy, y1 - head * uy - 4 * ux),
        (x1, y1),
        (x1 - head * ux - 4 * uy, y1 - head * uy + 4 * ux),
    ]
    stroke(sheet, mask, barbs, pen)


STEP = {"up": (0, -1), "down": (0, 1), "right": (1, 0), "left": (-1, 0)}
LENGTH = {"N": 52, "mg": 52, "T": 30, "f": 17}  # proportional to 40, 40, 20, 10 N


def fbd(sheet, mask, cx, cy, forces, pen, rng, scale=1.0) -> None:
    for force in forces:
        dx, dy = STEP[force.direction]
        n, sway = LENGTH[force.label] * scale, rng.uniform(-2, 2) * scale
        ex, ey = cx + dx * n + sway * dy, cy + dy * n + sway * dx
        ink_arrow(sheet, mask, cx, cy, ex, ey, pen)
        width = 0.6 * pen.size * len(force.label)
        lx = ex - width - 4 if dx < 0 else ex + 4
        ly = ey - pen.size * (0.15 if dy > 0 else 0.9 if dy < 0 else 1.15)
        write(sheet, mask, lx, ly, force.label, pen, rng)


def point(position: str) -> tuple[float, float]:
    x, y = position.strip("()").split(",")
    return float(x), float(y)


def parabola(sheet, mask, ox, oy, unit, elements, pen, rng, axes=False) -> None:
    """y = (x - h)^2 + k on axes with x from -1 to 7 and y from -5 to 6.

    Axes drawn by hand (`axes=True`) get ticks, every second one numbered, and the
    x-intercepts are labelled beside their dots; printed axes carry their own numbers.
    """

    def at(x: float, y: float) -> tuple[float, float]:
        return ox + x * unit, oy - y * unit

    def minus(text: str) -> str:
        return text.replace("-", "−")

    if axes:
        ink_arrow(sheet, mask, *at(-1, 0), *at(7.3, 0), pen)
        ink_arrow(sheet, mask, *at(0, -5), *at(0, 6.3), pen)
        for i in (-1, 1, 2, 3, 4, 5, 6, 7):
            stroke(sheet, mask, [at(i, 0.15), at(i, -0.15)], pen)
        for j in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
            stroke(sheet, mask, [at(-0.15, j), at(0.15, j)], pen)
        for i in (2, 4, 6):  # below the axis, where the curve is far below or above
            write(sheet, mask, *at(i - 0.2, -0.3), str(i), pen, rng)
        for j in (-4, -2, 2, 4, 6):
            width = 0.55 * pen.size * len(minus(str(j))) / unit
            write(sheet, mask, *at(-0.35 - width, j + 0.45), minus(str(j)), pen, rng)
    points = {e.label: e.position for e in elements if e.position}
    h, k = point(points["vertex"])
    reach = (6 - k) ** 0.5
    xs = [h - reach + i * reach / 20 for i in range(41)]
    curve = [at(x, (x - h) ** 2 + k) for x in xs if x >= -0.9]
    stroke(sheet, mask, curve, pen)
    for e in elements:
        if e.kind == "point":
            x, y = point(e.position)
            dot(sheet, mask, *at(x, y), pen)
            if axes and e.label == "x-intercept":
                # left root: just inside the arms; right root: outside the right arm
                text = minus(f"({x:g}, {y:g})")
                write(
                    sheet, mask, *at(x + 0.2 if x < h else x + 0.5, 1.1), text, pen, rng
                )
    write(sheet, mask, *at(h + 0.6, k - 0.2), minus(f"({h:g}, {k:g})"), pen, rng)


def sign(sheet: Sheet, name: str, pen: Pen, rng) -> None:
    mask = sheet.mask()
    write(sheet, mask, 110, 88 + (pen.size < 18) * 4, name, pen, rng)
    sheet.put("name", mask, pen.color, region="name")


def answer_in_place(sheets: dict[int, Sheet], works, marks, pen: Pen, rng) -> None:
    """Draw each part's work in the empty space beside it on its home page."""
    for pid, work in works.items():
        part = PART[pid]
        sheet, mask = sheets[part.home_page], sheets[part.home_page].mask()
        if pid in MC:
            letter, how = work.final["choices"][0], marks[pid]
            row = OPTION_ROWS[pid] + 17 * "ABCDE".index(letter)
            if how in ("listed", "labelled"):
                continue  # written elsewhere
            if how == "circle":
                loop(sheet, mask, OPTION_X + 8, row + 7.5, 13, 9, pen, rng)
                sheet.put(pid, mask, pen.color)
            elif how == "highlight":
                text = f"({letter})  {MC[pid][1]['ABCDE'.index(letter)]}"
                right = OPTION_X + stringWidth(text, "Times-Roman", 11) + 3
                box = [*sheet.px(OPTION_X - 3, row + 1), *sheet.px(right, row + 14)]
                ImageDraw.Draw(mask).rectangle(box, fill=105)
                sheet.put(pid, mask, HIGHLIGHT)
            elif how == "beside":
                write(
                    sheet,
                    mask,
                    REGION[pid][1] + 12,
                    OPTION_ROWS[pid] + 24,
                    letter,
                    pen,
                    rng,
                )
                sheet.put(pid, mask, pen.color, region=pid)
            else:  # "margin": left of the printed page, on the slide canvas
                write(sheet, mask, -30, OPTION_ROWS[pid] - 30, letter, pen, rng)
                sheet.put(pid, mask, pen.color)
        elif part.response_type == "free_body_diagram":
            fbd(sheet, mask, *DOT, work.elements, pen, rng)
            sheet.put(pid, mask, pen.color, region=pid)
        elif part.response_type == "graph":
            parabola(sheet, mask, *AXES_ORIGIN, AXES_UNIT, work.elements, pen, rng)
            sheet.put(pid, mask, pen.color, region=pid)
        else:
            _, x0, y0, _, _ = REGION[pid]
            on_page = len(work.lines) - work.below
            write_lines(sheet, mask, x0 + 8, y0 + 6, work.lines[:on_page], pen, rng)
            sheet.put(pid, mask, pen.color, region=pid)
            if work.below:  # carried on below the printed page, on the slide canvas
                mask = sheet.mask()
                write_lines(sheet, mask, x0 + 8, 802, work.lines[on_page:], pen, rng)
                sheet.put(pid, mask, pen.color, role="continuation")


def notebook(spacing: float) -> Sheet:
    """Ruled paper the size of the printed page, with rules `spacing` points apart."""
    paper = Image.new("RGB", PRINTED_SIZE, (247, 247, 242))
    draw, s = ImageDraw.Draw(paper), SLIDE_SCALE
    rule = 54.0
    while rule < 770:
        draw.line([(0, rule * s), (PRINTED_SIZE[0], rule * s)], fill=(165, 192, 225))
        rule += spacing
    draw.line([(58 * s, 0), (58 * s, PRINTED_SIZE[1])], fill=(225, 140, 140), width=2)
    return Sheet(paper, s)


def answer_on_notebook(sheet: Sheet, works, pen: Pen, rng) -> None:
    """Write labelled answers down the ruled sheet, two blank rules between them."""
    row = 1
    for pid, work in works.items():
        mask, top = sheet.mask(), 54 + row * pen.spacing - 0.95 * pen.size
        write(sheet, mask, 64, top, pid[1:].replace(".", "") + ")", pen, rng)
        kind = PART[pid].response_type
        if pid in MC:
            letter = work.final["choices"][0]
            write(sheet, mask, 64 + 2.2 * pen.size, top, letter, pen, rng)
        elif kind == "free_body_diagram":
            scale = min(1.0, pen.size / 12)
            cx, cy = 64 + 7 * pen.size, top + (LENGTH["N"] + 20) * scale
            fbd(sheet, mask, cx, cy, work.elements, pen, rng, scale)
        elif kind == "graph":
            unit = pen.size * 1.4
            origin = (64 + 2.5 * pen.size + unit, top + 6.6 * unit)
            parabola(sheet, mask, *origin, unit, work.elements, pen, rng, axes=True)
        else:
            write_lines(sheet, mask, 64 + 2.6 * pen.size, top, work.lines, pen, rng)
        sheet.put(pid, mask, pen.color, basis="student_label")
        bottom = mask.getbbox()[3] / sheet.scale
        row = ceil((bottom + 0.95 * pen.size - 54) / pen.spacing) + 2


def perspective(src, dst) -> list[float]:
    """Coefficients for Image.transform that take each point of src to dst."""
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, v])
    for i in range(8):  # Gauss-Jordan elimination with partial pivoting
        pivot = max(range(i, 8), key=lambda r: abs(rows[r][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        rows[i] = [a / rows[i][i] for a in rows[i]]
        for r in range(8):
            if r != i:
                f = rows[r][i]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[i], strict=True)]
    return [row[8] for row in rows]


def light(size: tuple[int, int], rng, low: float) -> Image.Image:
    """Uneven room light with a soft shadow, warm-tinted, in coarse steps.

    The steps make long runs of equal pixels, which keeps the slides small under
    the export's plain Flate compression.
    """
    w, h = size
    across = Image.linear_gradient("L").rotate(90).resize(size)
    down = Image.linear_gradient("L").resize(size)
    if rng.random() < 0.5:
        across = across.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    level = Image.blend(across, down, rng.uniform(0.3, 0.7))
    level = level.point(lambda v: round(250 - (250 - low) * v / 255))
    shadow = Image.new("L", (w // 8, h // 8))
    x, y = rng.uniform(0, w / 8), rng.uniform(h / 16, h / 8)
    ImageDraw.Draw(shadow).ellipse([x - 40, y - 25, x + 50, y + 30], fill=45)
    shadow = shadow.filter(ImageFilter.GaussianBlur(8)).resize(size)
    level = ImageChops.subtract(level, shadow).point(lambda v: v // 4 * 4)
    tint = [level.point(lambda v, t=t: round(v * t)) for t in (1, 0.97, 0.91)]
    return Image.merge("RGB", tint)


def photograph(sheet: Sheet, rng, turned=False, low=165.0):
    """A skewed, unevenly lit phone photo of the sheet, and the warp it applied."""
    paper = sheet.image
    if turned:
        paper = paper.transpose(Image.Transpose.ROTATE_90)
    w, h = (903, 677) if turned else (903, 1204)
    pw, ph = paper.size

    def jitter() -> float:
        return rng.uniform(-0.02, 0.02)

    corners = [
        (w * (0.02 + jitter()), h * (0.015 + jitter())),
        (w * (0.98 + jitter()), h * (0.02 + jitter())),
        (w * (0.985 + jitter()), h * (0.98 + jitter())),
        (w * (0.015 + jitter()), h * (0.985 + jitter())),
    ]
    coeffs = perspective(corners, [(0, 0), (pw, 0), (pw, ph), (0, ph)])

    def warp(image: Image.Image) -> Image.Image:
        if turned:
            image = image.transpose(Image.Transpose.ROTATE_90)
        desk = (88, 82, 74) if image.mode == "RGB" else 0
        return image.transform(
            (w, h),
            Image.Transform.PERSPECTIVE,
            coeffs,
            Image.Resampling.BICUBIC,
            fillcolor=desk,
        )

    photo = ImageChops.multiply(warp(sheet.image), light((w, h), rng, low))
    return photo.filter(ImageFilter.GaussianBlur(0.7)), warp


def photo_slide(photo: Image.Image) -> Image.Image:
    """A slide with a pasted photo fitted to the printed page's width."""
    rgba = photo.convert("RGBA")
    alpha = Image.new("L", photo.size, 255)
    ImageDraw.Draw(alpha).line(
        [(photo.width - 1, 0), (photo.width - 1, photo.height)], 191
    )
    rgba.putalpha(alpha)
    slide = Image.new("RGBA", (SLIDE_W, SLIDE_H), (0, 0, 0, 0))
    slide.paste(rgba, PHOTO_AT)
    return slide


def scan(sheet: Sheet, rng) -> tuple[Image.Image, list[Placed]]:
    """A scanning app's flattened, nearly straight page."""
    angle = rng.uniform(-0.6, 0.6)

    def turn(image: Image.Image) -> Image.Image:
        fill = (255, 255, 255) if image.mode == "RGB" else 0
        return image.rotate(angle, Image.Resampling.BICUBIC, fillcolor=fill)

    page = ImageChops.multiply(turn(sheet.image), light(sheet.image.size, rng, 228))
    return page.filter(ImageFilter.GaussianBlur(0.5)), sheet.placed(turn)


# --- The submissions --------------------------------------------------------------


class Work(NamedTuple):
    lines: list[str]  # what the student wrote, without their own part label
    final: dict  # the FinalAnswer fields the student filled
    awards: list[str]  # points per criterion, in key order
    why: str
    elements: list[DiagramElement] = []
    below: int = 0  # lines carried on below the printed page
    overruled: str | None = None  # why the grader overrules the code's answer check


def choice(pid: str, letter: str) -> Work:
    right = PART[pid].answer.choices[0]
    if letter == right:
        return Work([], {"choices": [letter]}, ["1"], f"Chooses {letter}, the answer.")
    why = f"Chooses {letter}; the answer is {right}."
    return Work([], {"choices": [letter]}, ["0"], why)


KEY = key()
PART = {p.id: p for p in KEY.parts}
CORRECT = {
    "Q1": choice("Q1", "C"),
    "Q2": choice("Q2", "D"),
    "Q3": choice("Q3", "B"),
    "Q4.a": Work(
        [], {}, ["1", "1"], "All four forces, labelled, from the dot.", FORCES
    ),
    "Q4.b": Work(
        [
            "f = µmg = 0.25(4.0)(10) = 10 N",
            "Fnet = T − f = 20 − 10 = 10 N",
            "a = Fnet/m = 10/4.0 = 2.5 m/s²",
        ],
        {"value": 2.5, "unit": "m/s^2"},
        ["2", "1"],
        "Newton's second law with T - f, and the right answer with units.",
    ),
    "Q4.c": Work(
        [
            "It slows down and stops.",
            "With the rope cut, friction is",
            "the only horizontal force, and",
            "it points against the motion.",
        ],
        {},
        ["1", "1"],
        "Says it slows down, because friction alone opposes the motion.",
    ),
    "Q5.a": Work(
        ["x² − 6x + 9 − 9 + 5", "= (x − 3)² − 4"],
        {"expression": "(x - 3)^2 - 4"},
        ["2"],
        "Completes the square correctly.",
    ),
    "Q5.b": Work(
        ["(x − 3)² − 4 is least at x = 3", "min = f(3) = −4"],
        {"value": -4.0},
        ["1", "1"],
        "Reads the minimum, -4, from the vertex form.",
    ),
    "Q5.c": Work(
        [], {}, ["1", "1"], "Upward parabola, vertex labelled, roots marked.", PARABOLA
    ),
}


class Student(NamedTuple):
    layout: str
    name: str
    works: dict[str, Work]  # the parts answered
    marks: dict[str, str]  # how each multiple-choice answer is shown


def students() -> list[Student]:
    slip = Work(
        CORRECT["Q4.b"].lines[:2] + ["a = Fnet/m = 10/4.0 = 2.0 m/s²"],
        {"value": 2.0, "unit": "m/s^2"},
        ["2", "0"],
        "Right method, but 10/4.0 is worked out as 2.0. The slip is inside this "
        "part, so nothing carries forward.",
    )
    long_q5a = CORRECT["Q5.a"]._replace(
        lines=[
            "f(x) = x² − 6x + 5",
            "= x² − 6x + 9 − 9 + 5",
            "= (x² − 6x + 9) − 4",
            "= (x − 3)² − 4",
            "so h = 3 and k = −4",
        ],
        below=2,
    )
    no_friction = Work(
        ["Fnet = ma", "a = T/m = 20/4.0", "a = 5.0 m/s²"],
        {"value": 5.0, "unit": "m/s^2"},
        ["1", "0"],
        "Newton's second law without friction in the net force: the listed partial "
        "credit on c1, and 5.0 m/s^2 is wrong.",
    )
    carry = {
        "Q5.a": Work(
            ["x² − 6x + 5", "= (x − 3)² + 9 − 5", "= (x − 3)² + 4"],
            {"expression": "(x - 3)^2 + 4"},
            ["1"],
            "Right squared term, but adds 9 instead of subtracting it: the listed "
            "partial credit.",
        ),
        "Q5.b": Work(
            ["vertex at (3, 4)", "min = 4"],
            {"value": 4.0},
            ["1", "1"],
            "Reads the minimum from the vertex of the 5(a) result. 4 follows from "
            "the wrong (x - 3)^2 + 4, so the final answer earns carry-forward credit.",
            overruled="The answer check finds 4, not -4, but 4 is the minimum of the "
            "student's own (x - 3)^2 + 4 from 5(a): carry-forward credit.",
        ),
    }  # 5(c) is drawn correctly from the given f(x), independent of the wrong 5(a)
    derivative = Work(
        ["f'(x) = 2x − 6 = 0", "x = 3", "f(3) = 9 − 18 + 5 = −4"],
        {"value": -4.0},
        ["1", "1"],
        "A different valid method: f'(x) = 0 gives x = 3, and f(3) = -4.",
    )
    vertex_formula = Work(
        ["x = −b/2a = 6/2 = 3", "f(3) = 9 − 18 + 5 = −4"],
        {"value": -4.0},
        ["1", "1"],
        "A different valid method: x = -b/2a gives x = 3, and f(3) = -4.",
    )
    three_forces = Work(
        [],
        {},
        ["1", "0.5"],
        "Weight, normal force and tension are right. Friction is missing: the "
        "listed partial credit on c2.",
        FORCES[:3],
    )
    half = {pid: CORRECT[pid] for pid in ("Q1", "Q3", "Q4.b", "Q5.a")}
    scanned = {k: v for k, v in CORRECT.items() if k not in ("Q4.c", "Q5.a")}
    return [
        Student(
            "typed-in-place",
            "Avery Example",
            CORRECT | {"Q2": choice("Q2", "C")},
            dict.fromkeys(MC, "beside"),
        ),
        Student(
            "ink-in-place",
            "Blake Sample",
            CORRECT | {"Q4.b": slip, "Q5.a": long_q5a},
            {"Q1": "circle", "Q2": "margin", "Q3": "circle"},
        ),
        Student(
            "photo-per-page",
            "Casey Mockup",
            CORRECT | {"Q4.b": no_friction},
            dict.fromkeys(MC, "circle"),
        ),
        Student(
            "all-on-foreign-pages",
            "Devon Stub",
            CORRECT | carry,
            dict.fromkeys(MC, "highlight"),
        ),
        Student(
            "rotated",
            "Emery Dummy",
            CORRECT | {"Q5.b": derivative},
            dict.fromkeys(MC, "circle"),
        ),
        Student(
            "partly-blank",
            "Finley Fakename",
            {**half, "Q4.a": three_forces},
            {"Q1": "highlight", "Q3": "circle"},
        ),
        Student("blank", "Gray Fixture", {}, {}),
        Student(
            "letters-listed",
            "Harper Testcase",
            CORRECT | {"Q3": choice("Q3", "A")},
            dict.fromkeys(MC, "listed"),
        ),
        Student(
            "small-writing",
            "Indigo Placeholder",
            CORRECT,
            dict.fromkeys(MC, "labelled"),
        ),
        Student(
            "low-resolution", "Jules Synthetic", CORRECT, dict.fromkeys(MC, "labelled")
        ),
        Student(
            "not-classkick",
            "Kit Specimen",
            scanned | {"Q1": choice("Q1", "B"), "Q5.b": vertex_formula},
            dict.fromkeys(MC, "circle"),
        ),
    ]


def small_sheet() -> tuple[Image.Image, list[Placed]]:
    """A full sheet of small handwriting, photographed and fitted onto a slide."""
    rng = random.Random("small-writing sheet")
    sheet = notebook(SMALL_PEN.spacing)
    answer_on_notebook(sheet, CORRECT, SMALL_PEN, rng)
    photo, warp = photograph(sheet, rng, low=185)
    return photo, sheet.placed(warp, PHOTO_AT)


def build(
    student: Student, printed, at_200ppi
) -> list[tuple[Image.Image, list[Placed]]]:
    """Each page of the submission, with where each block of work landed on it."""
    rng = random.Random(student.name)
    layout, works, marks = student.layout, student.works, student.marks

    def slides(pen: Pen, name: bool) -> dict[int, Sheet]:
        sheets = {
            k: Sheet(blank_slide(p), SLIDE_SCALE, PRINTED_AT)
            for k, p in enumerate(printed, 1)
        }
        if name:
            sign(sheets[1], student.name, pen, rng)
        answer_in_place(sheets, works, marks, pen, rng)
        return sheets

    if layout in ("typed-in-place", "ink-in-place", "partly-blank", "blank"):
        pen = KEYBOARD if layout == "typed-in-place" else STYLUS
        sheets = slides(pen, name=layout in ("typed-in-place", "ink-in-place"))
        return [(s.image, s.placed()) for s in sheets.values()]
    if layout == "letters-listed":
        sheets = slides(BLUE_STYLUS, name=True)
        for i, pid in enumerate(MC):
            mask = sheets[4].mask()
            text = f"{i + 1}. {works[pid].final['choices'][0]}"
            write(
                sheets[4],
                mask,
                30,
                806 + i * BLUE_STYLUS.spacing,
                text,
                BLUE_STYLUS,
                rng,
            )
            sheets[4].put(pid, mask, BLUE_STYLUS.color, basis="student_label")
        return [(s.image, s.placed()) for s in sheets.values()]
    if layout in ("photo-per-page", "rotated"):
        sheets = {k: Sheet(p.copy(), SLIDE_SCALE) for k, p in enumerate(printed, 1)}
        sign(sheets[1], student.name, PEN, rng)
        answer_in_place(sheets, works, marks, PEN, rng)
        pages = []
        for k, sheet in sheets.items():
            turned = layout == "rotated" and k == TURNED_PAGE
            photo, warp = photograph(sheet, rng, turned=turned)
            pages.append((photo_slide(photo), sheet.placed(warp, PHOTO_AT)))
        return pages
    if layout == "all-on-foreign-pages":
        first = Sheet(blank_slide(printed[0]), SLIDE_SCALE, PRINTED_AT)
        answer_in_place({1: first}, {k: works[k] for k in MC}, marks, PEN, rng)
        pages = [(first.image, first.placed())]
        for problem in ("Q4", "Q5"):
            sheet = notebook(PEN.spacing)
            mine = {k: v for k, v in works.items() if k.startswith(problem + ".")}
            answer_on_notebook(sheet, mine, PEN, rng)
            photo, warp = photograph(sheet, rng)
            pages.append((photo_slide(photo), sheet.placed(warp, PHOTO_AT)))
        return pages + [(blank_slide(printed[3]), [])]
    if layout in ("small-writing", "low-resolution"):
        photo, placed = small_sheet()
        if layout == "low-resolution":
            w, h = photo.size
            photo = photo.resize((w // 6, h // 6), Image.Resampling.BOX).resize((w, h))
        rest = [(blank_slide(p), []) for p in printed[1:]]
        return [(photo_slide(photo), placed), *rest]
    assert layout == "not-classkick"
    kept = {1: 1, 2: 2, 4: 3}  # blank page -> scanned page; page 3 was never scanned
    sheets = {k: Sheet(at_200ppi[k - 1].copy(), SCAN_SCALE) for k in kept}
    sign(sheets[1], student.name, PEN, rng)
    answer_in_place(sheets, works, marks, PEN, rng)
    return [scan(sheet, rng) for sheet in sheets.values()]


def fraction(box: tuple[int, int, int, int], size: tuple[int, int]) -> list[float]:
    """A pixel box, padded by 3 pixels, as fractions of the page image."""
    (w, h), (left, top, right, bottom) = size, box
    edges = (max(left - 3, 0) / w, max(top - 3, 0) / h, min(right + 3, w) / w)
    return [round(v, 4) for v in (*edges, min(bottom + 3, h) / h)]


def truth(student: Student, pages: list[tuple[Image.Image, list[Placed]]]) -> dict:
    classkick = student.layout != "not-classkick"
    size = (SLIDE_W, SLIDE_H) if classkick else (1700, 2200)
    unreadable = student.layout == "low-resolution"
    entries, set_aside = [], []
    for k, (_, placed) in enumerate(pages, 1):
        for part, role, basis, box in placed:
            if part == "name":
                reason = "The student's name on the name line."
                set_aside.append(
                    SetAside(page=k, box=fraction(box, size), reason=reason)
                )
            else:
                # On the unreadable sheet the map records where the work physically
                # is. Its part labels cannot be read, so this attribution is not
                # something a correct mapper can be expected to reproduce.
                entry = MapEntry(
                    part_id=part,
                    page=k,
                    box=fraction(box, size),
                    basis="content" if unreadable else basis,
                    role=role,
                    confidence="low" if unreadable else "high",
                )
                entries.append(entry)
    unanswered = [p.id for p in KEY.parts if p.id not in student.works]
    readings, grades = [], []
    for part in KEY.parts:
        work = student.works.get(part.id)
        nothing = {"choices": [], "value": None, "unit": None, "expression": None}
        reading = {
            "part_id": part.id,
            "status": "blank",
            "medium": None,
            "lines": [],
            "final_answer": nothing,
            "elements": [],
            "uncertain": [],
            "legibility": None,
        }
        if work and unreadable:
            reading |= {"status": "unreadable", "legibility": "poor"}
        elif work:
            written = student.marks.get(part.id) in (
                "beside",
                "margin",
                "listed",
                "labelled",
            )
            medium = "handwritten"
            if student.layout == "typed-in-place":
                drawn = part.response_type in ("free_body_diagram", "graph")
                medium = "mixed" if drawn else "typed"
            reading |= {
                "status": "answered",
                "medium": medium,
                "lines": work.final["choices"][:1] if written else work.lines,
                "final_answer": nothing | work.final,
                "elements": work.elements,
                "legibility": "clear",
            }
        readings.append(Reading.model_validate(reading))
        if unreadable:
            continue  # an unreadable response's score is withheld, never zero
        awards = work.awards if work else ["0"] * len(part.criteria)
        why = work.why if work else "No response."
        criteria = [
            CriterionGrade(
                id=c.id,
                awarded=Decimal(a),
                evidence_line=None,
                evidence_element=None,
                rationale=why,
                feedback="",
            )
            for c, a in zip(part.criteria, awards, strict=True)
        ]
        grades.append(
            Grade(
                part_id=part.id,
                criteria=criteria,
                transcript_dispute=None,
                belongs_elsewhere=None,
                code_overruled=work.overruled if work else None,
            )
        )
    flags = ["low_resolution", "unreadable"] if unreadable else []
    if any(g.code_overruled for g in grades):
        flags.append("code_overruled")
    ingest = [
        IngestPage(
            number=k,
            width_px=size[0],
            height_px=size[1],
            slide_number=k if classkick else None,
            native_ppi=SLIDE_W * 72 / SLIDE_BOX[2] if classkick else 200.0,
        )
        for k in range(1, len(pages) + 1)
    ]
    found = Map(entries=entries, unanswered=unanswered, set_aside=set_aside)
    # Clockwise degrees that turn each page's work upright. The sideways photo holds
    # the paper turned a quarter counter-clockwise (ROTATE_90 in `photograph`).
    rotations = [
        90 if student.layout == "rotated" and k == TURNED_PAGE else 0
        for k in range(1, len(pages) + 1)
    ]
    return {
        "layout_class": student.layout,
        "student_name": student.name,
        "source_kind": "classkick" if classkick else "other",
        "pages": [p.model_dump(mode="json") for p in ingest],
        "rotations": rotations,
        "map": found.model_dump(mode="json"),
        "readings": [r.model_dump(mode="json") for r in readings],
        "grades": [g.model_dump(mode="json") for g in grades],
        "flags": flags,
    }


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    (OUT / "key.json").write_text(KEY.model_dump_json(indent=2) + "\n")
    draw_blank(OUT / "blank.pdf")
    (OUT / "spec.json").write_text(spec().model_dump_json(indent=2) + "\n")
    printed = render_pages(OUT / "blank.pdf", PRINTED_SIZE)
    at_200ppi = render_pages(OUT / "blank.pdf", (1700, 2200))
    write_classkick(
        OUT / "blank_classkick.pdf", [blank_slide(p) for p in printed], None
    )
    for student in students():
        pages = build(student, printed, at_200ppi)
        pdf = OUT / f"{TITLE} - {student.name}.pdf"
        if student.layout == "not-classkick":
            write_scan(pdf, [image for image, _ in pages])
        else:
            write_classkick(pdf, [image for image, _ in pages], student.name)
        pdf.with_suffix(".json").write_text(
            json.dumps(truth(student, pages), indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
