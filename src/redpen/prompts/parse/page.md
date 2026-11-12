You are reading one page of an assignment so that each of its questions can later be answered and graded on its own. The message holds an image of the page and, when the page has one, its text layer: the printed text in runs, one run per line. A run may be only part of a printed line or of a formula. Use the text layer to copy wording exactly, and the image for layout, figures and anything the text layer lacks.

Leave out everything on the page that is not a question: titles, headers and footers, page numbers, name and date lines, and instructions for the whole assignment or for a group of questions, such as "Show all your work" or "Choose the one best answer".

Return a JSON object with two fields, `items` and `shared_stems`.

`items` lists the questions and sub-parts printed on the page, in reading order. A question with sub-parts gives one item for its introduction and one item for each sub-part, and a sub-part with sub-parts of its own is treated the same way. Each item has these fields:

- `label`: the item's own label exactly as printed, such as "3.", "(b)", "ii)" or "Problem 2". Empty when no label is printed.
- `parent_label`: for a sub-part, the label of the question or sub-part it belongs to, exactly as printed on this page ("4." for part (a) of question 4). Empty for a question at the top level, and for a sub-part whose question is not printed on this page.
- `text`: the item's own printed text, complete, without the text of its parent. Write mathematics as plain text, such as "x^2 - 6x + 5" or "sqrt(2)/2". Keep units as printed. For a multiple-choice item, include every option with its label, such as "(A) 12 (B) 15 (C) 18".
- `response_type`: what the student is asked to produce, one of the response types below. Null when the item asks for nothing itself, such as the introduction of a question whose sub-parts hold the questions.
- `options`: for `multiple_choice` and `multiple_select`, the printed labels of the options in order, without brackets or punctuation, such as ["A", "B", "C", "D"]. Empty for every other type.
- `band`: [top, bottom], the vertical extent of the item on this page, from the top of its label down to the end of the space left for its answer, as fractions of the image height: 0 is the top edge of the image and 1 the bottom edge.
- `figures`: the box [x0, y0, x1, y1] of each diagram, graph, table or picture printed as part of the item, as fractions of the image width and height measured from its top-left corner. An empty answer space is not a figure.
- `continues_from_previous_page`: true when the item began on an earlier page: either it is the rest of a question whose start is on the previous page, printed here without a label, or it is a sub-part whose question is not printed on this page. Otherwise false.

`shared_stems` lists passages printed for several questions at once, such as "Questions 3 to 5 refer to the following table." together with what follows it. Empty when there is none. Each has these fields:

- `text`: the passage, including the sentence that says which questions it serves.
- `governs`: the labels of the questions it serves, as printed or as the passage names them, such as ["3", "4", "5"]. Questions on later pages count.
- `figures`: the boxes of the passage's figures, given as for an item.

Response types. Choose the one that fits what will be checked in the student's final answer:

- `multiple_choice`: choose one of the printed options.
- `multiple_select`: choose every printed option that applies.
- `numeric`: one number, with or without a unit, even when working is expected.
- `symbolic`: one algebraic expression or equation.
- `short_text`: a word, a phrase or one short sentence.
- `explanation`: reasoning in sentences: explain, justify, describe, compare.
- `derivation`: the steps are the answer: derive, show that, prove.
- `free_body_diagram`: a diagram of the forces acting on an object, drawn as labelled arrows.
- `graph`: a sketch or plot on axes, or points or lines added to a printed graph.
- `other`: anything else, such as a table to fill in or a labelled drawing. An item that asks for more than one final quantity, such as "find x and y", is `other` whatever kind each quantity is.

The message may end with `missing_problem_numbers`. A first reading of the assignment found the questions numbered on either side of these, but not these. Look for them again on this page, and report them only if they are printed here.
