# redpen user guide

This guide is for the instructor who runs redpen. The README shows how to install it and grade the practice class; this guide says what each step does, what the files are, what to check, and what the tool does not catch.

A few words used throughout:

- **Model**: an AI model reached through OpenRouter. Each call to one costs money.
- **Role**: a job a model does. The parser reads the blank; the solver solves each problem and writes the rubric; the checker, a model of another company, solves each problem again; the referee settles a difference between the two solutions; the mapper finds where each answer is on a student's pages; the reader copies down what the student wrote; the second reader, of another company than the reader's, reads again what the reader was unsure of; the grader awards points.
- **Part**: a piece of a problem with its own points, such as 4(b). A problem with no sub-parts is one part. Parts are named Q1, Q4.b and so on.
- **Criterion**: one thing a part's answer must show, worth some points. Criteria are named Q4.b.c1, Q4.b.c2 and so on.
- **Key**: the answers, solutions, criteria and points of every part.
- **Workspace**: the folder that holds everything about one assignment.
- **Flag**: a mark the tool puts on a student's part, on a whole submission, or on a part of the key, when a score needs your eyes.

## The two steps

`redpen prepare` writes a key from the blank. `redpen grade` grades a class against that key. `redpen run` does the one and then the other. For your own classes, run them apart and read the key in between.

### Step 1: prepare the key

```
redpen prepare practice-set-3.pdf
```

The parser reads each page of the blank and lists its problems, their parts, and what kind of answer each asks for. The solver solves each problem. The checker solves it again on its own, and code compares the two answers; where code cannot compare them, the referee does. Then the solver writes the rubric, and the rubric's self-test grades the key's own solution as if it were a student's: it must earn every point.

The workspace is `practice-set-3.redpen`, beside the PDF, unless you give `--workspace DIR`. Points:

- Given neither option below, each part is worth the points of its criteria.
- `--total POINTS` shares a total among all parts, in proportion to their criteria's points.
- `--page-points P1,P2,...` gives each page of the blank its maximum, shared among the parts printed on it.

Under either option, points are shared in whole steps of `point_step`, so two parts of equal weight can differ by one step, and a part given a single step scores nothing below half of its criteria's points and all of it from half up.

Until you set them, with one of these options or by hand, the points of each part are the model's choice. On a workspace that has a key already, `--total` or `--page-points` sets its points and changes nothing else, with no model call. `prepare` has no dry run: `--dry-run` is refused. `redpen run BLANK.pdf SUBMISSIONS --dry-run` estimates both steps. A `prepare` that was stopped is run again: the key's Markdown files are written at its end.

### Between the steps: read and correct the key

Read these files in the workspace's `key/` folder before you grade:

- `verification.md`: how each part was checked, both answers, and the referee's reason. Its first lines name the parts that need a look.
- `rubric.md`: each part's criteria and points. A rubric can split one answer into several criteria, so that a correct answer put in other words loses a point. Read it before grading.
- `solutions.md`: the key's solution of each part.
- `key.yaml`: the key itself, the one file you edit.

Each part in `key.yaml` has a `state`:

- `verified`: the two solvers agreed, by code or by the referee, and the rubric passed its self-test.
- `refereed`: the referee had to choose between two different solutions, or found an arithmetic slip in one.
- `unverified`: the referee could not settle it, a call failed, or the rubric failed its self-test. Every student's score on that part is flagged `key_unverified`, so no student can be `final` until you settle it.

To correct the key, edit `key.yaml` in a text editor: an answer, a criterion's wording or points, partial credit, or the policies at the top of the file (`rel_tol`, how far a number may be from the key's, as a share of it: 0.02 is 2%; `require_units`; `penalise_sig_figs`; `carry_forward`, credit for using a wrong earlier result correctly; `point_step`, the smallest step of points). Set the policies before the first grade: a change of any of them grades everything again. Keep `rel_tol` zero or more. Leave an answer's `unit` empty for a pure number: a unit written "none" or "dimensionless" makes every answer given without a unit fail the check. Write expressions in plain keyboard characters (`pi`, `theta`, `sqrt(3)`): the check cannot read `π` or `θ`. Then check and render it:

```
redpen key check --workspace practice-set-3.redpen
redpen key render --workspace practice-set-3.redpen
```

`redpen key check` lists every problem it finds in the file, and notes each part changed since it was verified. `redpen key render` writes `solutions.md` and `rubric.md` again from `key.yaml`. `redpen key verify --workspace practice-set-3.redpen` has the checker, the referee and the self-test check the key as it now stands, and sets each part's `state`; it changes nothing else and costs money. If you have checked a part yourself, you may set its `state` to `verified` by hand: the state is not part of what the grader sees, so nobody is graded again.

Once a class is graded, edit the key in this order: edit `key.yaml`, run `redpen key check`, then `redpen regrade --dry-run`, then `redpen regrade` (see "Reviewing and correcting"). Until the re-grade, `redpen report` shows the edited part with no score and its students `incomplete`: that is expected, not damage.

### Step 2: grade the class

```
redpen grade submissions --workspace practice-set-3.redpen
```

SUBMISSIONS is a folder (its subfolders too) or a ZIP of PDFs, one per student. Files that are not PDFs are ignored. A student's name is the "Student:" line of a Classkick export, else the part of the file name after its last " - ", else the whole file name. A ZIP holds the class's PDFs only. With `--roster FILE.csv` (names in its first column, and no header row: a header is read as a name and listed as missing), a name that matches a roster name, ignoring capitals and spaces, takes the roster's spelling. The run report will list the roster names with no PDF, the PDFs that match no roster name, and the files that were skipped. If a student has two PDFs, the newer one is graded. SUBMISSIONS is always the whole class: grading a folder of late work alone makes the class those students only. Each student gets a student key, a folder name made of the name and six letters and digits, such as `Avery_Example-63834b`; it stays the same from run to run.

What grading does, in order, with the name each stage has on the progress bars and in the dry run's estimate:

1. **Ingest**: copies each PDF into the workspace and turns its pages into images.
2. **`map`**: the mapper finds where each part's answer is on the student's pages.
3. **`check`**: code compares each page with the class's and looks for writing no part claims (no model, so not in the estimate).
4. **`read`**: the reader copies down each answer.
5. **`second`**: the second reader reads again each answer the reader was unsure of.
6. **`grade`**: code scores choice answers and blank ones; the grader awards points on the others, criterion by criterion, and code checks a number or an expression against the key.
7. **`second grade`** (profiles `balanced` and `thorough`): each problem the grader graded is graded once more by the same model, and a part whose awards differ is flagged.
8. **`audit`**, the class audit: code finds the same work given different points across the class, and the grader decides each such group again; code also checks whether the class's most common answer differs from the key's.
9. **Report**: the files in `out/`.

`--dry-run` prints a cost estimate and sends nothing.

`redpen run BLANK.pdf SUBMISSIONS` runs `prepare` and then `grade`. Run again on a workspace that has a key, it checks the key and leaves it as it is.

Run one command at a time on a workspace. Two at once can pay twice for the same answers and make each other's steps fail; nothing is lost for good, and running the command again finishes the work.

## The workspace

The workspace of the practice class after `redpen run`, one `redpen review` and one `redpen override`. "Read" means a file is for you; "edit" means you may change it; "leave" means leave it alone. A folder that holds a file per page, per problem or per part shows only the first.

```text
practice-set-3.redpen/
  .gitignore                keeps the folder out of git (leave)
  config.lock.toml          models and profile pinned for this assignment (leave)
  cache.sqlite              every model answer, and what each call cost (leave)
  ingest_summary.json       the students of the latest grade, what ingest found (leave)
  reviews.yaml              your reviews, written by redpen review (leave)
  overrides.yaml            your corrections (edit, or use redpen override)
  key/
    blank.pdf               a copy of the blank (leave)
    pages/
      p01.png               each page of the blank as an image (leave)
      p01.json              what the parser read on that page (leave)
    spec.json               the problems and parts the parser found (leave)
    key.yaml                the key: answers, criteria, points, policies (read, edit)
    solutions.md            the key's solutions, made from key.yaml (read)
    rubric.md               the key's criteria and points, made from key.yaml (read)
    verification.md         how each part was checked; what needs a look (read)
    verification.json       the same, for the tool (leave)
  students/
    <student>/              one folder per student, named by the student key
      source.pdf            a copy of the student's PDF (leave)
      ingest.json           the student's name and pages (leave)
      pages/
        p01.png             each page as an image (read)
        p01.json            what the mapper found on that page (leave)
      map.json              where each part's answer is (leave)
      coverage.json         where writing lies that no part claims (leave)
      unclaimed/
        aside-1.png         a block the mapper set aside, such as a name (read)
      crops/
        Q1-1.png            each answer region as sent to the models, by part (read)
      readings/
        Q1.json             the reader's copy of each problem's answers (leave)
      grades/
        Q1.json             the grader's awards and reasons, by problem (leave)
        second.json         the second grade, compared with the first (leave)
  audit/
    Q1.json                 the class audit of each part (leave)
  out/
    scores_by_problem.csv   a row per student, a column per problem (read)
    scores_by_page.csv      a row per student, a column per page (read)
    scores_detail.csv       a row per student and criterion, with reasons (read)
    feedback/
      <student>.md          the student's feedback (read, then send)
    review.md               the review list (read)
    run_report.md           the run as a whole (read)
```

A student's folder can lack `crops/` and `grades/second.json` (a blank submission) or `unclaimed/` (nothing set aside). Never delete `key/spec.json` or `config.lock.toml` in the middle of a class: a fresh `prepare` asks everything again and pays for it.

Other files appear only when they are needed:

- `students/<student>/readings/second.json`: the second reading, when the reader was unsure of a part.
- `students/<student>/unclaimed/p01-1.png`: a patch of writing on page 1 that no part claims.
- `key/spec.rejected.json`: a parse that failed its checks. Correct it by hand and save it as `key/spec.json` to go on.
- `run.log`: the details of each failure, for whoever looks into it.
- `out/eval.md` and `out/eval.json`: written by `redpen eval`.

## The outputs

All in the workspace's `out/` folder, rebuilt after every grade, review or correction.

- **`scores_by_problem.csv`**: a row per student. Columns: `student`; one per problem (Q1, Q2, ...); `total`; `maximum`; `status`; `open_flags`, the number of open flags on the student, counting those on the key and low ones. An empty cell means a part of that problem has no score; `total` adds only what was scored.
- **`scores_by_page.csv`**: `student`, then one column per page of the blank (`page 1`, `page 2`, ...). A part's points count on the page where it is printed, wherever the student wrote the answer.
- **`scores_detail.csv`**: a row per student and criterion: `student`, `part`, `criterion`, `awarded`, `points`, and `rationale`, the grader's reason or why the part has no score. Always in the criteria's own points, even under `--total` or `--page-points`.
- The three CSV files begin with a byte-order mark, so that a spreadsheet reads them as UTF-8.
- **`feedback/<student>.md`**: the student's total, each problem and part with its points, and under each part the grader's feedback on each criterion that lost points, or why the part has no score yet. The title is the workspace's folder name without `.redpen`, so name the workspace after the assignment. The feedback is the grader's own words: read it before you send it.
- **`review.md`**: the review list. First the parts of the key with an open flag; then each student's flagged parts, high severity first and then by the points at stake; for each, its flags, the proposed score, the reader's copy of the answer, and links to the crops and stored files. Then corrections not applied, reviews out of date, entries closed by review, and the blocks the mapper set aside.
- **`run_report.md`**: the key's parts with their pages, points and states; the class audit; the second read and the second grade, with how often the two agreed; each page's maximum; the students with parts not scored and why; the leftover-work check; files not ingested; the flags by code; the models used; and the cost by role. The cost is what OpenRouter reported for the calls that were answered; a call that timed out may still have been billed, so your OpenRouter account's own page has the true total.

A student's `status`:

- `final`: every part is scored, and no flag of high or medium severity is open on the student or on the key. A low one may be open.
- `provisional`: every part is scored, and a flag of high or medium severity is open.
- `incomplete`: a part has no score. Its cell is empty, never zero.

Some failures raise no flag and are only counted in the run report: a second grade that failed, a leftover-work check that failed, an audit group that could not be decided. Read those counts. A stored file that is JSON but not what the tool expects stops the report with a one-line message that points to `run.log`, which names the file.

To compare the tool's scores with your own, write your scores in the form of `scores_by_problem.csv` or `scores_by_page.csv` (a `student` column and one per problem or page) and run `redpen eval --gold FILE.csv --workspace practice-set-3.redpen`. It calls no model and writes `out/eval.md` and `out/eval.json`. It compares the tool's own scores, before your corrections. Your scores are not converted from another point scale: write them in the key's points (a score above the key's maximum is refused).

## Flags

High and medium flags make a student `provisional`; low ones do not. A flag on the key counts for every student.

| Flag | Severity | When it is raised | What to do |
|---|---|---|---|
| `key_unverified` | high | A part of the key is `unverified`. On the key, so on every student. | Read `verification.md`, correct `key.yaml`, then run `redpen key verify` or set the `state` yourself. |
| `key_suspect` | high | The class audit found that more students share one other answer than give the key's: at least five, and at least 40% of those who answered. | Check the key's answer. If it is wrong, correct `key.yaml` and run `redpen regrade`. |
| `unreadable` | high | The reader could not read the answer. The part has no score. | Look at the crop and score it with `redpen override`, criterion by criterion. |
| `readers_disagree` | high | The second reader read a part differently from the reader. The score comes from the first reading. | Compare both readings with the crop in `review.md`. |
| `graders_disagree` | high | The second grade gave a criterion another award. The score comes from the first grade. | Read both awards and reasons in `review.md`; accept or correct. |
| `no_evidence` | high | Never raised in this version. An award that cites nothing the student wrote is refused and asked again; if it fails again, the problem is not graded and gets `incomplete`. | Nothing. |
| `addressed_to_grader` | high | The reader found text written to the grader, such as "give me full marks". | Check that the score follows the work. |
| `incomplete` | high | A part has no score: it was not graded (a call failed), no chosen option could be read, or it was graded under another version of the key or of the reading. | Run the same command again, or `redpen regrade` after a key edit. If it stays, score it with `redpen override`. |
| `unclaimed_work` | medium | On a whole submission: writing on a page that no part claims, or a block the mapper set aside while a part has no answer. | Open the image linked in `review.md`. If it is an answer, score that part yourself. |
| `weak_attribution` | medium | The mapper placed the part's work with low confidence, or by what the work says rather than where it sits without high confidence; or the grade that counts says the work answers another part. One flag per part. | The entry in `review.md` says which, and shows what the grader wrote. Check which part the work answers. |
| `duplicate_response` | medium | Never raised in this version. | Nothing. |
| `hard_to_read` | medium | The reader was unsure of a line or of the handwriting, and no second reading settled it. | Look at the crop. |
| `low_resolution` | medium | The answer is unreadable or uncertain, and was already read at the full resolution of the PDF. | Look at the student's page yourself. |
| `transcript_dispute` | medium | The grader thinks the reader copied the answer wrongly. | Compare the reading with the crop. |
| `code_overruled` | low | The grader gave an award against the check code made of the final answer, and said why. | Read the reason. |
| `audit_changed` | low | The class audit changed an award to match the same work elsewhere in the class. | Read the change in `review.md`. |

## Reviewing and correcting

### `redpen review`

```
redpen review --workspace practice-set-3.redpen
```

It shows each open entry of the review list in turn, with everything `review.md` shows of it, each criterion's requirement, award and reason, and the full path of each image. For each entry, type one of:

- `accept`: the entry is fine as it is. This closes its flags.
- `correct`: type the points of each criterion (press Enter to keep the award shown), then a note for the student or nothing. This records your corrections and closes the entry's flags.
- `skip`: leave it open for later.
- `quit`: stop. Everything done so far is kept; so it is if you press Ctrl-C.

An entry on a whole submission or on a part of the key offers only accept, skip and quit. Inside `correct`, on a part with no score, you must type a number for every criterion; only Ctrl-C leaves, and everything done before that entry is kept.

Each answer is stored as you give it, in `reviews.yaml` and `overrides.yaml`, and `out/` is rebuilt at the end. `redpen review --list` prints each open entry on one line and asks nothing. Neither costs anything.

A review closes an entry's flags only while the entry shows what you reviewed. It opens again when anything it shows changes: a flag, the reading, the bytes of a crop image, the second reading or grade, the award after the class audit, the reason a part has no score, your corrections, or the key's part. The review list lists such entries under "Reviews out of date". An update of the libraries that make the images can change every crop's bytes and so open every review again.

### `redpen override`

```
redpen override Avery_Example-63834b Q1.c1 0 --workspace practice-set-3.redpen --note "See me about this one."
redpen override Avery_Example-63834b Q1.c1 --remove --workspace practice-set-3.redpen
```

The student keys are the names of the folders under `students/`; `review.md` shows each beside the student's name. The first line records one correction: the student key, the criterion, and the points, from 0 to the criterion's full points in its own points. It replaces an earlier correction of the same student and criterion, and rebuilds `out/`. The second line, with `--remove`, deletes it. A correction does not close a flag, and it opens again a review of its entry. It applies to the version of the part it was made for: once you edit that part of the key, it is listed under "Corrections not applied" until you record it again. A part with no score gets one when every criterion of it is corrected.

You may also edit `overrides.yaml` by hand; the next command checks it and refuses an entry that names a criterion it does not know. When a student leaves the class, their reviews and corrections stay in the files and are listed as not applied; nothing is refused for them.

### `redpen regrade`

```
redpen regrade --workspace practice-set-3.redpen --dry-run
redpen regrade --workspace practice-set-3.redpen
```

After an edit of `key.yaml`, this grades the latest class again from the stored readings: no ingest, map or read. Then the second grade, the class audit and the report. Every request that did not change is answered from the cache. For an edited part, the request for its whole problem is sent and paid for again, and the grade of the edited part and of the parts that depend on it is replaced; the other parts of that problem keep the grade they had. With `--dry-run` it prints how many grade requests the cache would not answer and what they are estimated to cost, and sends nothing. An edit of the policies changes every part, so everything is graded again. A change of a part's `state` or of the points (`prepare --total` or `--page-points`) changes no grade; run `redpen report` instead.

### `redpen report`

```
redpen report --workspace practice-set-3.redpen
```

This writes `out/` again from what is stored, with your corrections and reviews. It calls no model.

## Stopping and resuming

You can stop every command with Ctrl-C, or lose the connection, and run the same command again. Every model answer is stored in `cache.sqlite` as it arrives, and a run asks the cache before it asks a model, so nothing answered is paid for twice. A request still waiting for its answer when you stop is sent again, and may be paid for twice. The files the tool reads back are written whole or not at all.

The estimate of `run --dry-run` and `grade --dry-run` covers the whole job as if nothing had been done yet: it does not look in the cache, so a resumed run costs less than it says. Under its table it prints what it assumes, such as no retry. `regrade --dry-run` does look in the cache.

## When something goes wrong

The tool says what went wrong in one line, and exits with a code other than 0. These are its fixed messages; the paths and counts in them are examples.

> OPENROUTER_API_KEY is not set. Export your OpenRouter API key in it.

Set the key as the README shows, in the terminal you run redpen from.

> OpenRouter refused the API key (HTTP 401). Set OPENROUTER_API_KEY to a valid key and run the same command again; what was answered is kept.

The key is wrong or was revoked. Make a new one on the OpenRouter site.

> The OpenRouter account has no credit left (HTTP 402). Add credit and run the same command again; what was answered is kept.

Add credit on the OpenRouter site and run the same command again.

> OpenRouter no longer serves a pinned model or endpoint (HTTP 404). `redpen models check --workspace DIR` names it; run the same command again once it is served, or start a new workspace. What was answered is kept.

A model pinned for this assignment is gone. Run the check it names.

> practice-set-3.pdf cannot be opened as a PDF: it does not exist. Give BLANK.pdf as a PDF file that opens without a password.

Check the path. In place of "does not exist" the line can say "is a folder", "is encrypted", "has no pages", or "is not a PDF, or is damaged or cut short". An encrypted PDF must be saved again without its password.

> submissions cannot be read as a folder or ZIP of PDFs, or the roster cannot be read (FileNotFoundError). Give SUBMISSIONS as a folder or ZIP of the students' PDFs.

The folder, the ZIP or the roster file is not where you said, or cannot be read. Check both paths.

> No student PDF was found in submissions. Give SUBMISSIONS as a folder or ZIP of the students' PDFs.

The folder or ZIP holds no file ending in .pdf.

> No student PDF could be ingested from submissions: 3 could not be read.

> Files not ingested: 1. practice-set-3.redpen/out/run_report.md lists them.

A student's PDF is encrypted or damaged. The run report names it; `ingest_summary.json` says why. Get the student a new PDF.

> Students with more than one PDF: 1. The newest of each was graded, and practice-set-3.redpen/ingest_summary.json names the others; remove those not meant to be graded.

Remove the PDFs that should not be graded, and run the same command again.

> practice-set-3.redpen holds no prepared key. Run `redpen prepare BLANK.pdf --workspace practice-set-3.redpen` first.

Check the `--workspace` path, or prepare the key first.

> practice-set-3.redpen holds the key of another blank. Use another --workspace.

Each workspace belongs to one blank. Give a new folder with `--workspace`.

> practice-set-3.redpen/key/key.yaml does not load. `redpen key check` lists its problems.

> practice-set-3.redpen/key/key.yaml has problems, so nothing was graded.

An edit of `key.yaml` broke it. The first line can also say "does not parse as YAML at line 12". Run `redpen key check`, which names each problem, and fix them.

> practice-set-3.redpen/key/spec.json is damaged (not JSON). Delete it and run `redpen prepare` again to have it made again.

> Stored files of students that are damaged (not JSON): 1. practice-set-3.redpen/run.log names them. Delete them and run `redpen grade` again to have them made again.

A stored file was cut short, by a full disk or by hand. Do what the line says: the file is made again, mostly from the cache.

> practice-set-3.redpen/config.lock.toml is damaged (not a lock). Delete it and run `redpen prepare` again to have the models pinned again from redpen.toml.

The lock was cut short or edited. Deleting it pins the models of `redpen.toml` afresh; if they are not the ones the key was made with, grading asks new questions and pays for them.

> Nothing was sent: the pinned endpoints are not verified to keep nothing.

The check before the first student page is sent failed: a model of this assignment no longer has a service that keeps nothing, or the check could not be completed. Run `redpen models check --workspace practice-set-3.redpen` to see which.

> grade: 2 of 11 students failed.

> Each traceback is in practice-set-3.redpen/run.log.

A call failed for some students after its retries. Run the same command again; what was answered is kept.

## Privacy

What is sent to the models, through OpenRouter:

- the blank assignment's pages, as images and text;
- each student's pages and the regions cut from them, as images. The mapper sees each page whole;
- the reader's copy of each answer, and the key, as text.

The workspace folder's name is sent with every request, as a label for the session: do not put a person's name in it. What is not sent: the PDFs themselves, the file names, the roster, and anything else in the workspace that is not part of a request. In text sent to a model, each student's whole name of two or more words is replaced by `[name]`: the name as the tool knows it, from the Classkick "Student:" line or the file name, or the roster's spelling. It is found in its own order or last name first with a comma, in any capitals. A file named without " - " can give a one-word name, which is not replaced: give a roster, or name the files "... - First Last.pdf".

`strict_privacy`, on by default, sends every request only to services that keep nothing: no data retention, no "Safety Retention", no use of the data. `redpen models check` says, for each role, whether its model has such a service. `redpen models check --workspace DIR` prints the same table, then a line for each model pinned in that workspace that has a problem; no such line means none. Before the first student page is sent, the tool checks again that the services pinned for the assignment still keep nothing, and sends nothing if it cannot confirm it. With `strict_privacy = false` this is not checked, and the run report says so for good.

`mask_names`, off by default, blanks each student's name in the images sent after the map: the mapper is asked where names are on each page, and sees each page before it is blanked. Once on for a workspace, it stays on for it. Both settings are read from `redpen.toml` in the folder you run redpen from.

The workspace holds copies of the students' PDFs, their pages, the readings and the scores. It gives itself a `.gitignore` so that git leaves it out. Keep it where you keep your gradebook. Nothing leaves your computer except the requests above, with the workspace folder's name.

## Settings

Settings live in `redpen.toml`, in the folder you run redpen from. Every setting is optional; with no file, the defaults below hold. The project's `redpen.toml.example` is a copy to start from. The API key is never a setting.

```toml
profile = "balanced"
image_budget = 1536
strict_privacy = true
mask_names = false

[models]
parser  = "~anthropic/claude-sonnet-latest"
solver  = "~anthropic/claude-opus-latest"
checker = "~openai/gpt-sol-latest"
mapper  = "~anthropic/claude-haiku-latest"
reader  = "~anthropic/claude-sonnet-latest"
grader  = "~anthropic/claude-sonnet-latest"
second  = "~openai/gpt-sol-latest"
referee = "~anthropic/claude-opus-latest"
```

- `profile`: how much checking is done; see below. `--profile` on `run` or `prepare` sets it for a new workspace.
- `image_budget`: the longest edge, in pixels, of an image sent to a model, from 256 to 4096. A change makes the images different, so a run asks its questions again and pays for them.
- `strict_privacy`: see "Privacy".
- `mask_names`: see "Privacy".
- `models`: the model behind each role: an OpenRouter model name, or a name such as `~anthropic/claude-sonnet-latest` that stands for that family's newest model and is fixed when a key is prepared. `checker` must be of another company than `solver`, and `second` of another company than `reader`, or the tool refuses to run: a model checking itself would agree with itself. The roles are `parser`, `solver`, `checker`, `referee`, `mapper`, `reader`, `second` and `grader`; the grader also does the rubric's self-test, the second grade and the class audit.

When a key is prepared, the models and the profile are pinned in the workspace's `config.lock.toml`. From then on that workspace keeps them, whatever `redpen.toml` or `--profile` says. A change of `redpen.toml` applies to new workspaces.

The profiles:

| Profile | What it adds |
|---|---|
| `economy` | Nothing beyond what every profile does: the second solver and the referee for the key, the rubric's self-test, the second read where the reader was unsure, the check for leftover work, and the class audit. |
| `balanced` | The default. The second grade: each problem the grader graded is graded once more by the same model, and a part whose awards differ is flagged `graders_disagree`. |
| `thorough` | The same as `balanced` in this version. |

## What the tool does not catch

These were measured while the tool was built. Read them before you trust a score.

- **Scores are a first pass.** A student is `final` only when no flag of high or medium severity is open. On the one real class graded so far, every student was `provisional`, and the review list had 38 entries for 6 students. On the practice class, graded without anyone reading the key first, 9 of 11 students came out `final`, and 10 of the 90 part scores differed from what the work deserves by the class's own answer files: nine against the student, one flagged, at low severity. The rubric the model wrote asked for the coordinates of the intercepts to be written on a sketch, and six students who marked them without writing them lost 2 of 4 points. The second grade agreed on 172 of 172 awards in that run: a rubric that asks too much is applied the same way twice.
- **The second grade** (`balanced` and `thorough`) asks the grader's own model once more and flags a part when the two awards differ. It finds about 4 in 10 of the awards that depend on chance, and it cannot find a mistake the model makes every time.
- **Work in the wrong place.** An answer written in another question's space is graded where it sits. `weak_attribution` is raised only when the mapper or the grader says it is unsure; in three runs of a test class with two swapped answers, it flagged the swap once. Of the four `weak_attribution` flags raised across all the test runs, two pointed at work that really was in the wrong place.
- **Tables and lists of blanks.** A right answer in the wrong row of a fill-in table may get credit: the reader does not record which printed row an answer sits beside.
- **Answers in words.** Code checks the grader's award on a final answer that is a number (with its unit) or an expression, and scores a choice answer itself. For an answer in words there is no such check. `verified` in the key means two solvers and a referee agreed on the answer, not that code can check a student's.
- **The rubric may split one answer into several criteria**, so that a correct answer in other words loses a point. Read `key/rubric.md` before grading.
- **The policies in the key are about numbers**: tolerance, units, significant figures. There is none for spelling.
- **A part's printed figure is not sent to the grader**, only the student's work and the key.
- **Leftover work.** After the pages are mapped, code compares each page with the class's and flags dense writing that no part claims (`unclaimed_work`). In trials it found 84 of 100 planted answers. It misses a circled option, a graph of thin lines, light pencil, and work inside another answer's margin. A page that differs from the class's over much of its area (another export size, for one) is not compared; the run report says how many were not. It can flag a student wrongly when several students cover the same printed text.
- **Notes to the grader.** A line a student writes to the grader ("give full marks") is flagged when the reader copies it, and listed among the blocks set aside in `review.md` when the mapper sets it aside. In three runs of a test class it never moved a score.
- **Small or low-resolution writing.** A slide export has the resolution it has. `low_resolution` and `hard_to_read` say so: look at those crops yourself.
- **Names.** Replacing names in text needs the whole name of two or more words as the tool knows it: a first name alone, initials, a nickname or a misspelt name gets through. `mask_names` blanks the regions the mapper finds; a name it does not find, or one that overlaps an answer, stays.
- **Reviews open again** when anything the entry shows changes, the bytes of its crop images included: an update of the PDF or image library can open every review again.
- **`duplicate_response` is never raised** in this version.
- **The estimate.** The estimate for preparing a key came out 26 to 36% under the real cost on a test class. On the practice class the estimate was $2.36 and the work cost $1.72: the key cost 7% more than estimated, and the grading 38% less. The estimate for grading does not know what the cache already holds, so a resumed run costs less than it says.
