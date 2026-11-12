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
effective image matrix and text origins are the same), and the coarse alpha levels
of real stylus strokes.
"""

import re
import shutil
from decimal import Decimal
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from reportlab import rl_config
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

from redpen.schemas.common import DiagramElement
from redpen.schemas.key import Answer, Criterion, Key, KeyPart, PartialCredit, Policies

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
REGION = {
    "name": (1, 104, 92, 300, 108),
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
            response_type=rtype,  # type: ignore[arg-type]
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
    c.drawCentredString(PAGE_W / 2, PAGE_H - 756, str(page))


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
        stem_top = top - 6 - 14.5 * (2 if len(stem) > 85 else 1)
        para(c, 72, stem_top - 4, stem, bold=f"{pid[1]}.")
        for row, (letter, option) in enumerate(zip("ABCDE", options, strict=True)):
            para(c, OPTION_X, top + 17 * row, f"({letter})  {option}")
    c.showPage()

    header(c, 2)
    bottom = para(c, 72, 62, Q4, bold="4.")
    floor = bottom + 92  # the figure: crate on a hatched floor, rope to the right
    c.line(196, PAGE_H - floor, 416, PAGE_H - floor)
    for x in range(200, 416, 10):
        c.line(x, PAGE_H - floor - 1, x - 6, PAGE_H - floor - 8)
    c.rect(236, PAGE_H - floor, 60, 44)
    c.setFont("Times-Roman", 10)
    c.drawCentredString(266, PAGE_H - floor + 18, "4.0 kg")
    arrow(c, 296, floor - 22, 384, floor - 22)
    c.setFont("Times-Italic", 10)
    c.drawString(330, PAGE_H - floor + 28, "rope")
    para(c, 72, 238, PARTS["Q4.a"], bold="(a)")
    c.circle(DOT[0], PAGE_H - DOT[1], 2.6, fill=1)
    para(c, 72, 478, PARTS["Q4.b"], bold="(b)")
    c.showPage()

    header(c, 3)
    para(c, 72, 62, PARTS["Q4.c"], bold="(c)")
    para(c, 72, 556, Q5, bold="5.")
    para(c, 72, 582, PARTS["Q5.a"], bold="(a)")
    c.showPage()

    header(c, 4)
    para(c, 72, 62, PARTS["Q5.b"], bold="(b)")
    para(c, 72, 318, PARTS["Q5.c"], bold="(c)")
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


# --- Slides and Classkick-style PDFs ----------------------------------------------


def render_pages(path: Path) -> list[Image.Image]:
    """The printed pages at slide resolution (903 x 1169)."""
    pdf = pdfium.PdfDocument(path)
    scale = 2 * PRINTED_SIZE[1] / PAGE_H
    pages = [
        page.render(scale=scale)
        .to_pil()
        .convert("RGB")
        .resize(PRINTED_SIZE, Image.LANCZOS)
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


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    (OUT / "key.json").write_text(key().model_dump_json(indent=2) + "\n")
    draw_blank(OUT / "blank.pdf")
    printed = render_pages(OUT / "blank.pdf")
    write_classkick(
        OUT / "blank_classkick.pdf", [blank_slide(p) for p in printed], None
    )


if __name__ == "__main__":
    main()
