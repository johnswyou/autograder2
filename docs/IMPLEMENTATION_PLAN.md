# redpen: implementation plan

How to build the system described in `SYSTEM_DESIGN.md`, in what order, and how to know each step is finished.

| | |
|---|---|
| Status | Draft for review |
| Date | 11 November 2026 |
| Revision | 3. Adds zoom handling for small writing (tasks 2.1, 2.4, 3.4 and 4.2). Revision 2 moved real data from Phase 0 to Phase 3 and fixed page scores to the printed-page rule. |
| Companion document | `SYSTEM_DESIGN.md` (section numbers below refer to it) |

## 1. Approach

The plan builds a thin version of the whole pipeline first, measures it against human grades, and only then adds the checks and second opinions from the design. There are two reasons for this order. The riskiest parts of the system are reading handwriting and mapping free-form pages, and neither risk can be judged until real submissions flow through real code. And most of the design's verification machinery costs money on every run, so each piece should have to show a measured gain before it stays.

| Phase | Outcome | Working days |
|---|---|---|
| 0. Groundwork | Repository scaffolded, schemas and synthetic fixtures in place | 3 |
| 1. Gateway | Cached, schema-checked, resumable calls to OpenRouter | 3 |
| 2. Walking skeleton | `redpen run` works end to end in its simplest form, on fixtures | 7 |
| 3. Measurement | Real exports collected, and baseline accuracy against human grades broken down by stage | 5 |
| 4. Accuracy | The design's checks and verification, each kept only if it earns its place | 11.5 |
| 5. Operations | Re-grading, overrides, review, privacy, packaging | 5 |
| 6. Pilot | Shadow grading on live assignments and a decision on how to use the tool | calendar time |

The estimates total about 35 working days for one developer who knows Python. They are rough. Phase 4 is the least certain, since its contents depend on what Phase 3 finds.

A hand-graded past assignment is not available yet, and the plan is arranged around that. Nothing before Phase 3 needs one: Phases 0 to 2 are built and tested on synthetic fixtures. Phase 3 cannot start without real data, so the date that data arrives sets the earliest finish for everything after it. A handful of ungraded exports would be enough to answer the format questions in task 3.1, and would be useful at any point before then.

## 2. Technology choices

| Need | Choice | Reason |
|---|---|---|
| Language | Python 3.12 or later | Requirement |
| Packaging | uv, `src/` layout, `pyproject.toml` | Fast, reproducible installs; `uv tool install` gives users a one-line setup |
| CLI | Typer | Subcommands and help text from type hints |
| Schemas | Pydantic v2 | One definition gives local validation and the JSON Schema sent to OpenRouter |
| HTTP | httpx, async | Direct control over OpenRouter-specific fields (`provider`, `session_id`) with no SDK in between |
| PDF rendering | pypdfium2 | Permissive licence. PyMuPDF is AGPL, which would constrain how the tool can be shared. |
| Imaging | Pillow, OpenCV (headless), NumPy | Rendering, feature matching, subtraction |
| Answer checks | SymPy, Pint | Expression equivalence and unit conversion |
| Key file | ruamel.yaml | Keeps comments and ordering when a hand-edited `key.yaml` is rewritten |
| Terminal output | Rich | Progress and tables |
| Storage | Plain files and the standard-library `sqlite3` | Nothing to install or administer |
| Quality | pytest, Hypothesis, ruff, pyright | Tests, property tests for point arithmetic, lint, types |

## 3. Repository layout

```
redpen/
  pyproject.toml
  README.md
  redpen.toml.example
  docs/
    SYSTEM_DESIGN.md
    IMPLEMENTATION_PLAN.md
    export-notes.md            findings from task 3.1
    baseline.md                results from Phase 3
  src/redpen/
    cli.py                     Typer app, one function per command
    config.py                  redpen.toml and config.lock.toml
    workspace.py               paths, artifact read and write, hashing
    orchestrator.py            the loop over students and problems
    schemas/
      spec.py  key.py  submission.py  reading.py  grade.py  flags.py
    llm/
      client.py                requests, retries, repair
      cache.py                 SQLite call cache
      ledger.py                tokens and cost
      models.py                checks, alias resolution, pinning
    imaging/
      render.py                PDF to images
      prep.py                  rotation, contrast, crops
      fingerprint.py           template match and subtraction
    prepare/
      parse.py  solve.py  rubric.py  allocate.py  selftest.py
    grade/
      ingest.py  mapper.py  reader.py  grader.py
      checks.py                numbers, units, multiple choice
      expr.py                  whitelist expression parser
      audit.py
    report/
      pages.py  tables.py  feedback.py  review.py
    eval/
      gold.py  metrics.py
    prompts/
      parse/  solve/  rubric/  map/  read/  grade/  audit/
  scripts/
    make_fixtures.py
    label.py
  tests/
    unit/  replay/  fixtures/  recordings/
```

Each stage in the design maps to one module. Model-facing text lives only under `prompts/`, as versioned files, so that a change to wording is visible in review and invalidates the right cache entries.

## 4. Phases

### Phase 0: Groundwork (3 days)

| # | Task | Days | Done when |
|---|---|---|---|
| 0.1 | Scaffold the project: package, CLI with every command stubbed, configuration loading, workspace paths, logging, and CI running ruff, pyright and pytest. | 1 | `redpen --help` lists every command in design section 12, and CI passes. |
| 0.2 | Write the Pydantic models for the spec, key, fingerprint, map, reading, grade and flags. Export JSON Schemas and lint them against strict-mode limits (every field required, no extra properties, no unions of objects). | 1 | Round-trip tests pass, and a test confirms every schema passes the lint. |
| 0.3 | Write the synthetic fixture generator. It builds a four-page assignment (three multiple-choice questions and two free-response problems, one with a free-body diagram) and renders submissions for each layout class in the table below, with ground truth alongside. | 1 | Fixtures build deterministically and are committed. |

Layout classes for the fixtures:

| Class | Description |
|---|---|
| typed-in-place | Typed text beside each problem on the blank page |
| ink-in-place | Handwriting-style strokes beside each problem |
| photo-per-page | Each page is a skewed, unevenly lit photo of the printed page |
| all-on-foreign-pages | Multiple choice in place; all other work on two photo pages; later pages untouched |
| rotated | A photo page turned 90 degrees |
| partly-blank | Half the problems unanswered |
| blank | Nothing added anywhere |
| letters-listed | Multiple-choice answers written as a list on a separate page |
| small-writing | A full sheet of small handwriting photographed and fitted onto one page |
| low-resolution | The same sheet embedded at a resolution too low to read |

Until real exports are available, the fixtures stand in for them. They encode assumptions about what a Classkick export looks like, for example that every slide appears as a page and that a page is the slide image under a header. Each assumption is written at the top of `make_fixtures.py`, so that task 3.1 can check them one by one against real files.

### Phase 1: Gateway (3 days)

Everything that talks to OpenRouter is built and tested before any stage uses it (design section 10).

| # | Task | Days | Done when |
|---|---|---|---|
| 1.1 | Client: async requests, message builder for text and image parts, JSON Schema response format, provider preferences, parameters filtered by what the model supports, error classes for retryable, fatal and validation failures. | 1 | A call with an image returns a validated object. |
| 1.2 | Call cache and usage ledger in SQLite. | 0.5 | A repeated call is served from the cache and the ledger records no new spend. |
| 1.3 | Retries with backoff, one repair attempt on a validation failure, concurrency limit, per-model rate limit. | 0.5 | A response that violates the schema is repaired once and then raised. A simulated rate limit is retried and succeeds. |
| 1.4 | `redpen models check`, alias resolution, and pinning to `config.lock.toml`. | 0.5 | A text-only model is rejected with a clear message. An alias resolves to a concrete slug, and a response from any other slug raises an error. |
| 1.5 | Record and replay for tests. | 0.5 | The test suite runs from recorded responses with the network disabled. Killing a batch midway and restarting finishes it without repeating completed calls. |

### Phase 2: Walking skeleton (7 days)

The simplest version of every stage: one model call each, no second opinions, no fingerprint, no audit. It is built and tested on the synthetic fixtures.

| # | Task | Days | Done when |
|---|---|---|---|
| 2.1 | Ingest and image preparation: read a folder or ZIP, take student names from file names (with an optional roster), render PDFs to images, and handle rotation and contrast correction. Render any region of a page straight from the PDF at a chosen scale, and record the native resolution of each embedded photo (design 7.7). Each transform is recorded so that boxes can be mapped back to the original page. | 1 | Unit tests pass on the fixtures, including the rotated class. A ZIP and a folder of the same files give identical results. A region rendered from the PDF at three times scale is measurably sharper than the same region enlarged from the 200 DPI page. |
| 2.2 | Prepare, first version: parse per page and stitch, one solver, rubric with one point per criterion, `key.yaml` written, `solutions.md` and `rubric.md` rendered, `redpen key check`. | 2 | On the fixture assignment the spec lists every part with the right type, and the key's answers match ground truth. |
| 2.3 | Map, first version: per-page inventory and a resolve call, with the completeness check only. | 1 | Every layout class yields a map that points each answered part at the right page. |
| 2.4 | Read, first version: one read per unit (multiple choice by page, free response by problem). Each mapped region is rendered from the PDF to fill the model's image budget and sent with the full page as an overview. No tiling yet. | 1 | Multiple-choice selections are read correctly in every layout class. |
| 2.5 | Grade, first version: multiple choice and blanks scored in code, one grader call per problem, bounds and sums enforced in code. | 1 | Fixture scores match ground truth. An award outside the listed levels is rejected. |
| 2.6 | Report, first version: the three outputs, with each part's points placed on the page where the part is printed (design 8.2), and a run report with cost. | 0.5 | Page totals equal problem totals for every student. A part answered on a foreign page still scores under its printed page. |
| 2.7 | `redpen run`: wiring, progress display, `--dry-run`, resumption. | 0.5 | The run completes on every fixture class and writes every output. The dry-run estimate is within 30% of the actual spend. |

### Phase 3: Measurement (5 days)

This is where real data enters. The phase cannot start without the material in task 3.1.

| # | Task | Days | Done when |
|---|---|---|---|
| 3.1 | Collect one assignment that has been graded by hand: the blank PDF, every student's PDF, and the page scores. Make a copy with names replaced by codes and keep it outside the repository. Read five of the exports by hand and record what they contain. Correct the fixture assumptions and the ingest stage to match, then run the skeleton on the full set. | 2 | `docs/export-notes.md` answers the questions below. An anonymised set of at least 25 submissions with human page scores exists. `redpen run` completes on it and writes every output. |
| 3.2 | `redpen eval`: gold CSV loader (by page or by problem), the metrics in design section 15, and a listing of the largest disagreements with links to artifacts. | 1 | The command runs on the real set and prints the table. |
| 3.3 | Labelling script. It pre-fills a map and final answers from the tool's own output, and a person corrects them. This produces stage-level gold for about twenty submissions. | 1 | Twenty submissions are labelled. |
| 3.4 | Stage metrics: mapping accuracy, reading accuracy on final answers, and grading accuracy when given gold readings. Then a comparison of three or four candidate models for the reader role, and the same for the grader. Reading accuracy is also measured against line height in pixels, which sets the zoom target and confirms each model's image budget. | 1 | `docs/baseline.md` holds one table per stage and records which models were chosen, with the numbers behind the choice. The zoom target and each model's image budget are recorded there too. |

Questions for `export-notes.md`:

- Does the PDF include pages the student never opened?
- Where are the student name and slide number placed, and how do they shift the page content?
- Are typed answers real text in the PDF, or flattened into the image?
- At what resolution are pasted photos embedded?
- Do page size and aspect ratio match the blank export?
- How are the files named?

Phase 3 ends with a short review. The baseline shows which stage loses the most points, and the order of work in Phase 4 is adjusted to match.

### Phase 4: Accuracy (11.5 days)

The tasks are listed in the order expected to matter most. Each must show an improvement on a metric from Phase 3 or catch a seeded error. A task that does neither is dropped or moved behind the `thorough` profile.

| # | Task | Days | Done when |
|---|---|---|---|
| 4.1 | Fingerprint (design 7.2): feature matching, homography, subtraction, boxes of added regions, untouched detection, and label anchors from the blank's text layer. | 2 | Every untouched fixture page is detected, and no page with any student mark, including a single highlight, is called untouched. |
| 4.2 | Adaptive zoom (design 7.7): line-height estimate in code, tiling with overlap for map and read, reading again at higher zoom when the reader is unsure, the native-resolution ceiling, and the `low_resolution` flag. | 1.5 | The small-writing fixture is read correctly. The low-resolution fixture is flagged `low_resolution` instead of being guessed at. Reading accuracy on small writing in the labelled set is higher than baseline. |
| 4.3 | Map checks (design 7.3): coverage, second look, duplicates, weak attribution, and re-mapping at the grader's request. | 1.5 | A response hidden in the corner of a foreign page is found or flagged. Mapping accuracy on the labelled set is higher than baseline. |
| 4.4 | Answer checks in code (design 7.5): numbers with units and tolerance, accepted alternatives, the whitelist expression parser, symbolic equivalence, and the rule for two marked options. | 1.5 | A table of equivalent and non-equivalent pairs passes. The parser rejects hostile input (attribute access, names with double underscores, oversized exponents). |
| 4.5 | Key verification (design 6.2 to 6.5): second solver, comparison, referee, arithmetic check, rubric self-test, and the point allocator with `--total` and `--page-points`. | 2 | Property tests show allocated points always sum exactly to the constraint. Key accuracy is measured on problems with published answers. |
| 4.6 | Selective verification (design 10.4): triggers for second reads and second grades under each profile, the referee, and the transcript dispute loop. | 1.5 | The three profiles behave as the design's table says. The cost and accuracy of `economy` and `balanced` are both measured. |
| 4.7 | Class audit (design 7.6): same answer same score, outlier pass, and the check of the key against the class. | 1 | Two identical responses given different awards are reconciled. A key answer altered by hand raises `key_suspect` on the real set. |
| 4.8 | Flags and the review list (design 9 and 8.4): severity, provisional status, `review.md`. | 0.5 | The share of large disagreements that carry no flag is measured and reported. |

The untouched-page threshold in 4.1 is deliberately conservative. Calling a touched page untouched would silently drop a student's work, while the opposite error only costs one model call.

For 4.5, released AP Physics 1 free-response questions and their published scoring guidelines are a ready source of problems with known answers and known rubrics.

### Phase 5: Operations (5 days)

| # | Task | Days | Done when |
|---|---|---|---|
| 5.1 | Re-grading by part hash with dependency closure, and overrides with staleness (design 6.6 and 11). | 1.5 | Editing one criterion re-grades that part and its dependents with no map or read calls. An override survives the re-grade. |
| 5.2 | `redpen review`: step through flags, show the transcript and the image path, accept or override. | 1 | A full review list can be cleared from the terminal. |
| 5.3 | Privacy (design 14.1): provider preferences on every request, name masking, `models check` under the privacy settings, and a `.gitignore` written into each workspace. | 1 | A test asserts that no request body contains a student name. Masked regions are visibly blanked in the images that are sent. |
| 5.4 | Hardening: behaviour under rate limits, a kill-and-resume test, a 150-submission run from recordings, and clear messages for common mistakes (missing API key, wrong path, encrypted PDF, empty ZIP). | 1 | All of these are covered by tests. |
| 5.5 | Packaging and documentation: install with `uv tool install`, a README quick start, and a short user guide. | 0.5 | Someone who has not seen the project can install it and grade the fixture class from the README alone. |

### Phase 6: Pilot

For two or three live assignments the instructor grades by hand as usual and the tool runs alongside. `redpen eval` compares the two each time.

The pilot ends in a decision against the targets in design section 15, with three possible outcomes:

1. The targets are met. The tool is used for real, and the instructor checks the review list and a random 10% of scores on each assignment.
2. The targets are met for some part types only, most likely multiple choice and typed or clearly written numeric work. The tool grades those, and the rest are routed to the review list by default.
3. The targets are not met. The baseline tables show which stage is responsible, and work returns to Phase 4.

After the pilot, the anonymised sets from these assignments become the regression suite. Any change to a prompt or a pinned model is run against that suite before it is used on live work.

## 5. Testing strategy

The tests are layered so that most of them need no network and no spend.

Unit tests cover all code that makes no model call: the point allocator, page roll-up, answer checks, the expression parser, fingerprinting on synthetic images, and the report builders. The allocator and the point sums also get property tests.

Replay tests run the full pipeline against recorded model responses. They catch breakage in the plumbing between stages and run in CI.

The layout matrix requires every fixture class in Phase 0 to produce a correct map. When a real submission arrives in a layout the tool handles badly, a fixture for that layout is added before the fix is written.

Seeded-error tests plant a known problem and require the named check to catch it: a wrong key answer, a response hidden on a foreign page, writing too small to read at page scale, an instruction written to the grader, identical answers graded differently, and a blank page. Every flag in design section 9 has at least one such test.

A live smoke test sends one small submission through the real API. It is opt-in and run before releases.

The regression evaluation runs the anonymised real sets and compares against human grades. It stays outside the repository because it contains student work.

## 6. Risks

| Risk | Likelihood | Response |
|---|---|---|
| The hand-graded assignment arrives late | Likely, since none is available now | Phases 0 to 2 do not need it. Everything from Phase 3 on waits for it. A few ungraded exports in the meantime would settle the format questions early. |
| Handwriting is not read accurately enough | Medium | Measured in Phase 3, before the accuracy work is chosen. Model comparison for the reader role, then second reads. If it still falls short, outcome 2 of the pilot applies. |
| Diagram and graph grading is unreliable | High | Expected. These part types always get a second grade, and can be routed to review by default. |
| The export format holds a surprise (flattened pages, missing pages, headers that break matching) | Medium | Fixtures are built on written-down assumptions until task 3.1 checks them against real files. Fingerprint, the stage most sensitive to format, is built after that. No later stage depends on it. |
| The export stores pasted photos at low resolution | Unknown until task 3.1 | Zoom cannot recover detail the file does not hold. Affected work is flagged `low_resolution`. If it is common, the remedy is on the submission side: fewer problems per photo, or the original photos collected alongside the export. |
| The generated key is wrong for a problem | Medium | Second solver, self-test, the class consensus check, and a re-grade that costs no map or read calls. |
| Cost is higher than estimated | Medium | Dry-run estimates from live prices, the `economy` profile, a cheaper mapper model, and prompt caching. |
| Structured output behaves differently across providers | Medium | Local validation on every response, `require_parameters`, `models check`, and the repair retry. |
| A model update changes grading behaviour | Certain over time | Concrete slugs are pinned per assignment. The regression suite is run before re-pinning. |

## 7. Definition of done for version 1

- `redpen run blank.pdf submissions.zip` completes unattended on the real set and writes every file listed in design section 8.
- The targets in design section 15 are met on at least two assignments, or the gaps are written down together with a decision on how the tool will be used.
- Every flag in design section 9 has a test that raises it.
- A re-grade after a key edit makes no map or read calls.
- A fresh machine can install and run the tool from the README alone.

## 8. What is needed from the instructor

1. One past assignment that was graded by hand: the blank PDF, all student PDFs, and the page scores. It is needed before Phase 3 and not before.
2. If possible sooner, three to five real exports, graded or not, to settle the format questions in task 3.1.
3. Answers to the three remaining open questions in design section 17.
4. About two hours to correct the labels for twenty submissions in Phase 3.
5. For the pilot, two or three assignments graded by hand as usual while the tool runs alongside.
