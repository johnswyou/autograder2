"""Page images and regions of a PDF, for a blank assignment or a submission.

A Classkick export (docs/export-notes.md) is recognised by its structure: each page
holds the sidebar text and one image at the slide box, and nothing else. Its working
image is that slide image at native size, composited over white. Any other PDF is
rendered at 200 DPI. A PDF that matches on some pages only is `other`.

Boxes are [x0, y0, x1, y1] in fractions of the page image, from its top-left.
"""

# pypdfium2 ships no stubs and leaves many parameters and returns unannotated. The
# rest of strict mode stays on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot
from typing import Literal

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from redpen.imaging.prep import crop

type SourceKind = Literal["classkick", "other"]

DPI = 200
# Where the slide sits on a Classkick page, in points: left, bottom, right, top.
SLIDE_BOX = (200.0, 8.88, 591.56, 792.0)
SIDEBAR_X = 15.0  # where every sidebar line starts, in points
TOLERANCE = 1.0  # points
MAX_ENLARGEMENT = 2.0  # past native resolution (export-notes decision 4)


@dataclass(frozen=True)
class Sidebar:
    student_name: str | None  # None on a blank export
    assignment_title: str | None
    slide_number: int
    slide_count: int


@dataclass(frozen=True)
class TextSpan:
    text: str
    box: list[float]


@dataclass(frozen=True)
class Page:
    image: Image.Image  # RGB
    # The lowest resolution of the page's images, in pixels per inch of the page;
    # None if it has no image.
    native_ppi: float | None
    sidebar: Sidebar | None  # on a Classkick page only
    text: list[TextSpan]  # an `other` page's text layer; empty on a Classkick page


@dataclass(frozen=True)
class Region:
    image: Image.Image  # RGB
    box: list[float]  # the box rendered, widened to whole slide pixels on Classkick
    # Image pixels per native pixel, so above 1 holds no new detail. None on a page
    # with no image, whose vector content has no ceiling.
    scale: float | None


def sidebar(page: pdfium.PdfPage) -> Sidebar | None:
    """The sidebar of a Classkick export page, or None if the page is not one."""
    if pdfium_c.FPDFPage_GetAnnotCount(page.raw):
        return None
    images: list[pdfium.PdfImage] = []
    lines: list[str] = []
    for obj in page.get_objects(max_depth=0, textpage=page.get_textpage()):
        if isinstance(obj, pdfium.PdfImage):
            images.append(obj)
        elif (
            isinstance(obj, pdfium.PdfTextObj)
            and abs(obj.get_matrix().e - SIDEBAR_X) <= TOLERANCE
        ):
            lines.append(obj.extract().strip())
        else:
            return None
    if len(images) != 1 or "Printed from Classkick" not in lines:
        return None
    bounds = images[0].get_bounds()
    if any(abs(a - b) > TOLERANCE for a, b in zip(bounds, SLIDE_BOX, strict=True)):
        return None
    text = "\n".join(lines)
    slide = re.search(r"^Page (\d+) / (\d+)$", text, re.MULTILINE)
    if slide is None:
        return None

    def field(label: str) -> str | None:
        found = re.search(rf"^{label}: (.*)$", text, re.MULTILINE)
        return (found[1].strip() or None) if found else None

    return Sidebar(
        student_name=field("Student"),
        assignment_title=field("Assignment"),
        slide_number=int(slide[1]),
        slide_count=int(slide[2]),
    )


def classkick_sidebars(pdf: pdfium.PdfDocument) -> list[Sidebar] | None:
    """Each page's sidebar if every page is a Classkick page, else None."""
    found = [s for s in (sidebar(page) for page in pdf) if s is not None]
    return found if found and len(found) == len(pdf) else None


def render_page(page: pdfium.PdfPage, kind: SourceKind) -> Page:
    """The working image of a page, with what is recorded about it at ingest."""
    if kind == "classkick":
        slide = _slide(page)
        return Page(_composite(slide), _native_ppi([slide]), sidebar(page), [])
    # pyright infers an int scale from the default; pypdfium2 takes a float.
    bitmap = page.render(scale=DPI / 72)  # pyright: ignore[reportArgumentType]
    image = bitmap.to_pil().convert("RGB")
    to_bitmap = bitmap.get_posconv(page).to_bitmap
    textpage = page.get_textpage()
    spans: list[TextSpan] = []
    for index in range(textpage.count_rects()):
        left, bottom, right, top = textpage.get_rect(index)
        text = textpage.get_text_bounded(left, bottom, right, top).strip()
        (ax, ay), (bx, by) = to_bitmap(left, top), to_bitmap(right, bottom)
        xs = sorted(min(max(x / image.width, 0.0), 1.0) for x in (ax, bx))
        ys = sorted(min(max(y / image.height, 0.0), 1.0) for y in (ay, by))
        if text:
            spans.append(TextSpan(text, [xs[0], ys[0], xs[1], ys[1]]))
    return Page(image, _native_ppi(_images(page)), None, spans)


def render_region(
    page: pdfium.PdfPage, kind: SourceKind, box: Sequence[float], longest_edge: int
) -> Region:
    """`box` of the page's working image with its longest edge near `longest_edge`
    pixels: rendered from the PDF at that scale, or on a Classkick page cropped from
    the slide image and resampled, enlarging by at most 2x."""
    if kind == "classkick":
        part, used = crop(_composite(_slide(page)), box)
        factor = min(longest_edge / max(part.size), MAX_ENLARGEMENT)
        size = (round(part.width * factor), round(part.height * factor))
        return Region(part.resize(size, Image.Resampling.LANCZOS), used, factor)
    width, height = page.get_size()  # as displayed, after any /Rotate
    x0, y0, x1, y1 = box
    scale = longest_edge / max((x1 - x0) * width, (y1 - y0) * height)
    cut = (x0 * width, (1 - y1) * height, (1 - x1) * width, y0 * height)
    bitmap = page.render(scale=scale, crop=cut)  # pyright: ignore[reportArgumentType]
    image = bitmap.to_pil().convert("RGB")
    ppi = _native_ppi(_images(page))
    return Region(image, list(box), scale * 72 / ppi if ppi else None)


def _images(page: pdfium.PdfPage) -> list[pdfium.PdfImage]:
    return [
        obj
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])
        if isinstance(obj, pdfium.PdfImage)
    ]


def _slide(page: pdfium.PdfPage) -> pdfium.PdfImage:
    [slide] = _images(page)
    return slide


def _composite(image: pdfium.PdfImage) -> Image.Image:
    """The image at its native pixel size with its soft mask, over white."""
    rgba = image.get_bitmap(render=True, scale_to_original=True).to_pil()
    white = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(white, rgba.convert("RGBA")).convert("RGB")


def _native_ppi(images: list[pdfium.PdfImage]) -> float | None:
    # shortcut: an image inside a Form XObject is measured against the form's own
    # space, not the page; compose the form matrices if such a PDF turns up.
    ppis: list[float] = []
    for image in images:
        matrix = image.get_matrix()
        width, height = hypot(matrix.a, matrix.b), hypot(matrix.c, matrix.d)
        px_width, px_height = image.get_px_size()
        if width and height:
            ppis.append(72 * min(px_width / width, px_height / height))
    return min(ppis, default=None)
