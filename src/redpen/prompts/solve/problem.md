You are solving one problem of an assignment, so that an answer key can be written for it. The assignment may be in any subject. Students' responses will be graded against your answers, so each answer must be correct and must be stated in the form the question asks for.

The message holds:

- An image of each page the problem is printed on, in page order. Figures, tables, graphs and given values may appear only on the images. Read them carefully.
- A JSON text with `pages`, the page number of each image in order, and `parts`, every part of the problem in order. Each part has `id`, `label` (as printed), `response_type` (what the student is asked to produce, see below), `prompt` (the full text of the part, including any introduction or shared passage it depends on) and `options` (the printed option labels of a choice question, otherwise empty).
- Possibly a JSON text with `problems`. Your previous answer to this same message had these problems. Answer again in full and correct them.

Solve the parts in order. A later part may use the result of an earlier one. Use the values, constants and conventions the problem gives. Where it gives none, use the standard ones.

Return a JSON object with one field, `parts`: one entry for each part in the message, in the same order, and no other entries. Each entry has five fields:

- `part_id`: the part's `id`, exactly as given.
- `solution`: the worked solution as a teacher would write it for the key. Name the principle, rule or method used, then give each step, the substitution of the given values, and the result with its unit. One step per line. Write mathematics as plain text: `*` for multiplication, `/` for division, `^` for powers, `sqrt()`, `pi`, and function names such as `sin()` or `ln()`. Where the solution uses the result of an earlier part, say which part. When an alternative assumption changes the result (see `alternatives`), say so at the end, with the result it gives.
- `answer`: the final answer, described below.
- `uses`: the ids of the earlier parts in this message whose results your solution of this part uses, whether or not the question says so. Empty when there are none. Never the part itself or a later part.
- `alternatives`: only for a numeric result, and only when the question leaves open an assumption that has more than one defensible value, such as a constant that is commonly used with two different values, or a quantity that can fairly be read two ways from a figure. Solve with the value the problem's own context makes standard, and list each other defensible choice here, with `assumption` (what is assumed instead, as a short phrase) and `value` (the final number that follows from it, in the same unit as the answer's `value`). Empty when the question settles every assumption, for example by stating the value to use.

The `answer` object has eight fields. Fill the ones that fit the part's `response_type`, as listed further down, and leave every other field null or empty:

- `choices`: the labels of the chosen options, exactly as they appear in `options`.
- `value`: the final number.
- `unit`: the unit of `value` in plain text, such as "m/s^2", "kg*m/s" or "cm^3", or null for a pure number. Give the unit the question expects. Do not round more than the given values justify.
- `rel_tol`: null, unless the question itself calls for a looser match than an exact calculation, as for an estimate or a value read off a graph or a figure. Then give the relative tolerance a fair grader would accept, such as 0.1.
- `accept_also`: always empty. Alternative answers go in `alternatives`.
- `expression`: the final expression in exact form, in the variables and constants the question names. Keep fractions, roots and `pi` exact rather than decimal, unless the question asks for a decimal. Write the right-hand side only: for "find v", the expression for v, not "v = ...". Use the same plain-text notation as in `solution`.
- `elements`: the elements the question requires a correct drawing to show, one entry each, with four fields: `label` (the label a correct drawing would carry, using the problem's own symbols where it has them), `kind` (what the element is, such as "force", "curve", "point", "line", "asymptote" or "axis label"), `direction` (which way it points, or the shape of a curve, such as "opens upward" or "increasing, linear"; null when that does not apply) and `position` (where it is, such as "(3, -4)" for a point, or where an arrow starts; null when that does not apply).
- `must_include`: each fact, claim, step or feature a full-credit response must contain, one short sentence each.

`elements` and `must_include` hold only what the question asks for. A value it does not ask for, a feature of a drawing or a graph it does not mention, or a consequence beyond what it asks is not required, however true, and stays out of both. It can go in `solution`.

The fields each response type uses:

- `multiple_choice`: `choices`, exactly one label.
- `multiple_select`: `choices`, every correct label.
- `numeric`: `value` and `unit`, and `rel_tol` as described.
- `symbolic`: `expression` only. No unit.
- `short_text`: `must_include`, the word, phrase or fact the response must state.
- `explanation`: `must_include`, the conclusion first, then each reason a correct explanation must give.
- `derivation`: `must_include` for the steps that must appear, and the result in `expression`, or in `value` and `unit`, where there is one.
- `free_body_diagram`: `elements`, one per force, and `must_include` for any other requirement of the diagram.
- `graph`: `elements` for each curve, line and labelled point a correct graph shows, with coordinates where they are determined, and `must_include` for any other required feature.
- `other`: whichever fields hold the answer. When the part asks for several results, give each in `must_include` with its value and any unit, such as "x = 4" and "y = -1".
