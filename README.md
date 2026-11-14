# redpen

redpen grades one assignment for one class. You give it the blank assignment as a PDF and one PDF of work per student; it writes an answer key for you to check, then grades each student's work against that key with AI models. It writes scores, a feedback file for each student, and a review list of the scores it is unsure of, which you look at yourself before you use any score.

## What you need

- Python 3.12 or later, and [uv](https://docs.astral.sh/uv/), the program that installs redpen. uv can install Python for you.
- An account at [OpenRouter](https://openrouter.ai) with credit on it. OpenRouter passes each request to an AI model and bills your account for it.
- Your OpenRouter API key in the environment variable `OPENROUTER_API_KEY`. Never write the key in a file. To set it for the terminal you are in without it showing on the screen or going into your shell's history, run these two lines, paste the key after the first, and press Enter:

```
read -rs OPENROUTER_API_KEY
export OPENROUTER_API_KEY
```

## Install

In a terminal, in the project's folder (your copy of this repository):

```
uv tool install .
redpen --version
```

If the terminal says `redpen` is not found, run `uv tool update-shell`, then open a new terminal.

## Quick start: grade the practice class

The project holds a made-up class to practise on: a four-page assignment of physics and algebra and eleven students' PDFs. The names are invented. Copy them into a folder of their own, outside the project. From the project's folder:

```
mkdir -p ~/redpen-practice/submissions
cp tests/fixtures/synthetic/blank.pdf ~/redpen-practice/practice-set-3.pdf
cp tests/fixtures/synthetic/Practice*.pdf ~/redpen-practice/submissions/
cd ~/redpen-practice
```

1. **Check the models.** This costs nothing. Each row of the table it prints should end in `ok`; for a row that says `FAIL`, the lines under the table say why, and the guide's section "Settings" says how to choose another model.

   ```
   redpen models check
   ```

2. **See what the run will cost.** A dry run sends nothing to any model. It prints, for each stage of the work, the number of model calls, the tokens sent and returned, and the cost in US dollars, then the assumptions behind those numbers. The run itself can cost more or less: the guide says by how much it has been off.

   ```
   redpen run practice-set-3.pdf submissions --dry-run
   ```

3. **Grade.** This writes the answer key and then grades every student. It shows a progress bar for each stage with the money spent so far. Everything goes into the folder `practice-set-3.redpen` beside the PDF, called the workspace. You can stop it with Ctrl-C and run the same command again: nothing already answered is paid for twice. If its last lines say that some students failed, run it again.

   ```
   redpen run practice-set-3.pdf submissions
   ```

4. **Look at the results** in `practice-set-3.redpen/out/`:
   - `scores_by_problem.csv` and `scores_by_page.csv`: a row per student, for a spreadsheet. The `status` column says `final`, `provisional` (a flag on the student is open: look before you use it) or `incomplete` (a part has no score).
   - `feedback/`: one file per student, with the points of each part and the reasons points were lost. Read it before you send it.
   - `review.md`: the review list, most serious first.
   - `run_report.md`: the run as a whole: the key, what was flagged, the models used, and what it cost.

5. **Review.** This shows each entry of the review list in turn and asks you to accept, correct or skip it. It costs nothing.

   ```
   redpen review --workspace practice-set-3.redpen
   ```

For a class of your own, prepare the key first, read it and correct it, and only then grade: the guide says how.

## Where students' work goes

Each student's pages are sent through OpenRouter to the AI models, and only to services that keep nothing of what they are sent; this is checked before the first page goes. Everything else stays in the workspace folder on your computer. The guide's section "Privacy" says what is sent and what is not.

## More

The user guide, [docs/USER_GUIDE.md](docs/USER_GUIDE.md), covers the two steps, the workspace, the outputs, the flags, reviewing and correcting, settings, privacy, and what the tool does not catch.
