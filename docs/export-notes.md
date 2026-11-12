# Export notes

What real submission files contain, measured on 12 November 2026 from 178 student PDFs and 25 blank assignments across two terms. This answers the format questions of plan task 3.1 ahead of schedule, because the files turned out to be available. No student names appear here.

## What a Classkick export is

173 of the 178 student PDFs are Classkick exports (PDF producer `PDFKit`). Every one has the same structure.

| Question | Answer |
|---|---|
| Does the PDF include pages the student never opened? | Yes. The page count equals the slide count of the assignment in all 173 files. Page *k* is slide *k*. |
| Where are the student name and slide number? | In a real text layer in a left sidebar: `Student: <name>`, `Assignment: <title>`, `Page k / N`, `Class Code: XXX XXX`, and `Printed from Classkick` near the bottom. The lines start at x = 15 pt. |
| Are typed answers real text? | No. The sidebar is the only text. Everything on the slide is flattened into one image. |
| At what resolution are pasted photos embedded? | They are not embedded separately. Each page holds exactly one image object, the whole slide, 1004 × 2008 pixels, placed at the box (200, 0, 592, 783) in points. That is 184 pixels per inch on the PDF page. |
| Do page size and aspect ratio match the blank? | The PDF page is 612 × 792 pt, the same as the blank. The slide is not: it is a 1:2 portrait canvas. The printed letter page sits in the upper part of it, roughly pixels x 50 to 950 and y 50 to 1215, so the printed content gets about 106 pixels per inch of the original paper. |
| How are the files named? | `<assignment title> - <student name>.pdf`. The title itself contains ` - `, so the name is the part after the last one. The `Student:` line in the sidebar is the better source. |

Three details matter to the code.

The slide image has a soft mask. The canvas outside the printed page is transparent, and it comes out black if the raw image stream is read without the mask. It must be composited over white.

A long assignment title in the sidebar runs past x = 200 pt and is drawn across the top-left of the slide. Cropping a page render to the slide box therefore picks up sidebar text. The slide has to be taken from the image object itself, with its mask.

The sidebar is the only place the student name is printed by the export. An image built from the slide alone does not contain it.

## Where students put their work

Seen on one nine-slide assignment with six students:

- Typed text beside the questions, and typed labels on free-body diagrams.
- Stylus handwriting on the printed page.
- A highlight or a stylus circle on a multiple-choice option.
- Multiple-choice letters written in the left margin, outside the printed page.
- Handwriting continued below the printed page, in the empty lower half of the slide.
- A photo of each printed page pasted over the whole slide, skewed and unevenly lit.
- Slides left untouched.

Work outside the printed page (left margin, lower half of the slide) is common enough that no stage may assume the response lies within the printed area.

## Blank assignments

15 of the 25 blanks are vector PDFs from LaTeX or an office suite, with a full text layer. The other 10 are themselves Classkick exports: same sidebar (without the student and class lines), same 1004 × 2008 slide image, no text layer for the content. For those, a blank slide and a submission slide are already aligned pixel for pixel.

## Other PDFs

Five submissions, all for one exam, are not Classkick exports. They come from phone scanning apps and PDF tools, and their page counts (3, 4, 4, 5 and 8) differ from the blank's 5. Their pages hold one to three images each at up to 928 pixels per inch. These take the general path: no slide number, no fixed layout.

## Consequences for the build

These are decisions taken by the build overseer on the strength of the measurements above. Each departs from an assumption in the design or the plan.

1. **Fixtures copy the real layout.** The synthetic fixtures of plan task 0.3 are built as Classkick exports as described here: a 612 × 792 pt page, the sidebar text, and one 1004 × 2008 image with a soft mask at (200, 0, 592, 783). One fixture class is added for a non-Classkick PDF whose page count differs from the blank.
2. **Ingest reads the slide, not the page.** For a Classkick export the working image of a page is the slide image at its native 1004 × 2008 pixels, composited over white, taken from the image object. Student name and slide number are read from the sidebar text, with the file name as fallback. Any other PDF is rendered at 200 DPI as the design says. The same applies to a blank that is a Classkick export.
3. **Coordinates on a Classkick page are fractions of the slide.** Boxes in maps and readings refer to the slide image. The transform between a vector blank's page and the slide is measured by the fingerprint stage (plan task 4.1). Until then the band in the spec is only a hint passed to the mapper.
4. **Native resolution is the ceiling everywhere.** Design section 7.7 expects that typed text and stylus strokes survive as vector content with no ceiling. They do not. Rendering a Classkick page above 184 pixels per inch adds nothing, so zoom means cropping and tiling the slide image, plus the optional enlargement of up to 2x. The `low_resolution` flag should be expected on photographed pages with small writing.
5. **Page matching is known, not inferred.** On a Classkick export, page *k* is blank page *k*. Feature matching is still needed to register a vector blank onto the slide and for non-Classkick PDFs.
6. **Real files never enter the repository.** Tests that need them read the folder named by the environment variable `REDPEN_REAL_DATA` and are skipped when it is unset. Nothing derived from them (images, recorded model responses, transcripts, names) is committed.
