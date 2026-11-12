You are given what was read from each page of an assignment, as JSON. `pages` is the number of pages, and each entry of `results` holds a page number and the `items` and `shared_stems` read on that page. An item has a `label` and a `parent_label` as printed, its `text`, a `response_type`, `options`, a `band` and `figures` on its page, and `continues_from_previous_page`. A shared stem is a passage printed for several questions, with the labels of the questions it `governs`.

Join the pages into the list of every part of the assignment that a student must answer.

- An item whose `response_type` is null asks for nothing itself. It is the introduction of a question, or of a sub-part, whose own sub-parts hold the questions. It is not a part, but its text and figures go to every part below it.
- A question without sub-parts is one part. A question with sub-parts gives one part for each sub-part that asks for something.
- A question can run over a page break. An item marked `continues_from_previous_page` belongs to the question still open at the end of the previous page: a sub-part becomes a sub-part of that question, and unlabelled text joins the text of the item it continues.
- A shared stem goes to every question it governs.

Return a JSON object with one field, `parts`, holding the parts in the order they are printed. Each part has these fields:

- `id`: "Q" followed by the question's number as printed, then one segment for each level of sub-part, joined with dots: Q7 for question 7, Q7.b for its part (b), Q7.b.ii for part (ii) of that. A segment is the sub-part's printed label in lowercase letters, without brackets or punctuation: "(b)" gives "b" and "ii)" gives "ii". Where sub-parts are numbered rather than lettered, use the letter at that position in the alphabet: "(2)" gives "b". When the questions carry no usable printed numbers, number them 1, 2, 3 in the order they appear. No two parts have the same id.
- `label`: what is printed, the question's label followed by the label of each sub-part level, without trailing punctuation, such as "7(b)", "7(b)(ii)" or "3". Never invent one: when nothing is printed, the label is empty.
- `prompt`: everything a solver needs to answer this part on its own: the text of any shared stem that serves it, the introduction of its question, then the part's own text, word for word as read, including the options of a multiple-choice part.
- `response_type`: the part's response type as read, one of `multiple_choice`, `multiple_select`, `numeric`, `symbolic`, `short_text`, `explanation`, `derivation`, `free_body_diagram`, `graph` and `other`. Change it only where joining text across a page break shows that another fits better. A part that asks for more than one final quantity, such as "find x and y", is `other`.
- `home_page`: the page on which the part's own label is printed, or, for a part without a label, the page where its text starts.
- `band`: the part's band on its home page, as read.
- `figures`: every figure the part needs, each with the `page` it is on and its `box` as read: the part's own figures, those of every introduction above it, and those of any shared stem that serves it.
- `depends_on`: the ids of earlier parts whose results this part uses, such as a value found in an earlier part or "using your answer to (a)". Empty when it uses none.
- `options`: the option labels as read, for `multiple_choice` and `multiple_select`. Empty for every other type.
