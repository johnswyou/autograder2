# redpen

redpen grades one assignment for one class. You give it the blank assignment as a PDF and one PDF of work per student; it writes an answer key for you to check, then grades each student's work against that key with AI models. It writes scores, a feedback file for each student, and a review list of the scores it is unsure of, which you look at yourself before you use any score.

## What you need

The commands below are for a terminal on macOS or Linux.

- [uv](https://docs.astral.sh/uv/), the program that installs redpen. To install it on macOS or Linux:

  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  On Windows, in PowerShell:

  ```
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

  redpen needs Python 3.12 or later. If your computer has none, uv fetches one by itself during the install.
- An account at [OpenRouter](https://openrouter.ai) with credit on it. OpenRouter passes each request to an AI model and bills your account for it.
- Your OpenRouter API key in the environment variable `OPENROUTER_API_KEY`. Never write the key in a file. To set it without it showing on the screen or going into your shell's history:
  1. Type `read -rs OPENROUTER_API_KEY` and press Enter. The terminal waits and shows nothing.
  2. Paste the key and press Enter. Nothing appears.
  3. Type `export OPENROUTER_API_KEY` and press Enter.

  Do not paste the two lines together. The key lasts for that terminal only: do this again in each new terminal.

## Install

In a terminal, in the project's folder (your copy of this repository):

```
uv tool install .
redpen --version
```

If the terminal says `redpen` is not found, run `uv tool update-shell`, then open a new terminal (and set the key again).

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

2. **See what it will cost.** A dry run sends nothing to any model. It estimates both steps, preparing the key and grading: for each stage, the number of model calls, the tokens sent and returned, and the cost in US dollars, then the assumptions behind those numbers. On the practice class it said $2.36, and the work cost $1.72.

   ```
   redpen run practice-set-3.pdf submissions --dry-run
   ```

3. **Prepare the key.** This writes the answer key into `practice-set-3.redpen/key/`, in the folder `practice-set-3.redpen` beside the PDF, called the workspace. It takes a minute or two and shows nothing until it is done; then it prints the problems and parts it found, their points, and what it spent.

   ```
   redpen prepare practice-set-3.pdf
   ```

4. **Read the key** before you grade. A model wrote it. Open `practice-set-3.redpen/key/rubric.md`: each part's criteria and their points. Look for a criterion that asks for more than you would, and for points weighted as you would not. Then open `practice-set-3.redpen/key/verification.md`: its first lines name the parts that need a look. To change the key, edit `key/key.yaml` as the guide's section "Between the steps" says.

5. **Grade.** This grades every student against the key. It shows a progress bar for each stage, with the money spent so far: `map` (finding each answer on the pages), `check` (looking for work no answer region claims), `read` (copying down each answer), `second` (reading again what was unclear), `grade`, and `second grade` (grading once more, to compare). Its last line says what it spent; `practice-set-3.redpen/out/run_report.md` has the cost by role, under "Cost". You can stop it with Ctrl-C and run the same command again: what was already answered is not paid for twice. Run again on a finished class, it costs nothing and takes about a minute. If its last lines say that some students failed, run it again.

   ```
   redpen grade submissions --workspace practice-set-3.redpen
   ```

6. **Look at the results** in `practice-set-3.redpen/out/`:
   - `scores_by_problem.csv` and `scores_by_page.csv`: a row per student, for a spreadsheet.
   - `scores_detail.csv`: a row per student and criterion, with the grader's reason for each award.
   - `feedback/`: one file per student, with the points of each part and the reasons points were lost. Read it before you send it.
   - `review.md`: the review list, most serious first.
   - `run_report.md`: the run as a whole: the key, what was flagged, the models used, and what it cost.

   The `status` column of `scores_by_problem.csv` says `final`, `provisional` or `incomplete` (a part has no score). `final` means only that no flag of high or medium severity is open on the student. It does not mean the score is right. When the practice class was graded without anyone reading the key first, 9 of 11 students came out `final`, and 10 of the 90 part scores differed from what the work deserves by the practice class's own answer files: nine of them against the student, and only one flagged, at low severity. The cause was the rubric the model wrote: it asked for the coordinates of the intercepts to be written on a sketch, and six students who marked them without writing them lost 2 of 4 points. The grader applied it the same way both times it graded.

7. **Review.** This shows each entry of the review list in turn and asks you to accept, correct or skip it. It costs nothing.

   ```
   redpen review --workspace practice-set-3.redpen
   ```

`redpen run practice-set-3.pdf submissions` does steps 3 and 5 in one go, without the pause to read the key.

## Where students' work goes

Each student's pages are sent through OpenRouter to the AI models, and only to services that keep nothing of what they are sent; this is checked before the first page goes. Everything else stays in the workspace folder on your computer. The guide's section "Privacy" says what is sent and what is not.

## More

The user guide, [docs/USER_GUIDE.md](docs/USER_GUIDE.md), covers the two steps, the workspace, the outputs, the flags, reviewing and correcting, settings, privacy, and what the tool does not catch.
