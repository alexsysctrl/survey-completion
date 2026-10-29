# Completing a survey from what a respondent already answered

A model for the missing half of a household survey.

Long surveys do not ask everyone everything. A question is only put to a respondent if an earlier
answer took certain values — a housing question after "do you own a home here?", a module on return
intentions only in the rounds that fielded it. So a dataset of a thousand households contains a lot of
cells that were never recorded, and some of them were never recorded for a *reason*. This model reads
the answers that exist and puts a probability on every answer that does not, including the
"this question was not asked" case.

It was built for the SimulacraBench competition (NeurIPS 2026 Competition Track), which scores models
on three real instruments: a household survey of Syrian refugees in four countries (UNHCR), a survey on
religious leaders and childhood immunisation (UNICEF), and an online skills assessment for job seekers
in southern Africa (World Bank).

## The part that decides the score

On the UNHCR instrument, 156 of the 213 questions to be predicted are gated, and 56% of the cells being
scored hold the never-asked value. Treating that as one more category to estimate is estimating a fact:
whether a branch was reached follows from an earlier answer. When that earlier answer is one the model
can see, the answer to "was it asked" is not a guess at all.

Everything else is a hierarchy of evidence about the respondent — where they came from, which round of
the survey this row is, who in the household answered, how many children under five are in the
household — combined with counts.

```
P(t | key) = ( count(key, t) + m · P(t | parent key) ) / ( count(key) + m )
```

A `key` is a combination of the respondent's visible answers; its parent is the same combination with
one feature dropped, and the ladder ends at the population's own answer rates. `m` is what makes the
model honest: a combination backed by three respondents is barely trusted, one backed by three hundred
is believed. A joint key that never appears in the data never claims confidence it does not have.

Underneath the pooled marginal sits a prior: for each question, what a small general language model
says a respondent would answer, read from the option letters' first-token probabilities and bundled as
a table (`model/priors/`). It enters as a handful of pseudo-respondents, so the first real counts
outvote it. It is included because on real survey text, "what share of displaced households own a home
they left behind" is a question a model can have an opinion about — and because a wrong opinion is
cheap when it is only worth five people.

## What it was measured on, and what those numbers can and cannot mean

The competition publishes a practice harness with generated answers: real questions, real options, real
gating, invented answer distributions. Their own documentation is blunt about it — the practice answers
"have no relationship to the world at all". So these numbers show that the structure is implemented
correctly and that the whole thing runs inside the time budget. They do not say where this model ranks
against the field, and nothing here claims otherwise.

Skill, as the competition defines it: 0 is the population's own answer rates, 1 is perfect, negative is
worse than guessing. Measured 29 October 2026 with their own grader, development phase. This is the
ablation, one codebase, each layer switched on in order — reproduce it with `python3 ablate.py --kit
<path to the starter kit>`:

| Model | UNHCR | World Bank | UNICEF |
|---|---|---|---|
| marginals only — the metric's own zero | 0.002 | 0.002 | −0.003 |
| + joint-key counts with support-weighted backoff | 0.177 | 0.238 | 0.379 |
| + the exact never-asked rule | 0.177 | 0.238 | 0.379 |
| + the bundled language-model prior | 0.177 | 0.238 | 0.379 |
| the organizers' own baseline, for reference (their code, its language model removed) | 0.142 | 0.238 | 0.379 |

Read that honestly, in both directions. The first row landing at zero is the check that the ruler is
being used correctly: a model that only knows population answer rates gets the score the metric says it
should. The jump to the second row is the whole of the gain, and it comes from conditioning on what
the respondent did answer. The third and fourth rows change nothing **on this data** — the generated
answers contain none of the world knowledge a prior is for, and their gating is already learnable from
the conditioning columns. Two of the three layers in this model are therefore unproven by the only data
publicly available, and are included because the instruments they are meant for are real surveys, which
is a judgment this data cannot check.

For reference, the current best scores on the competition's own development leaderboard (snapshot dated
27 October 2026) are 0.884 on UNHCR, 0.249 on World Bank and 0.350 on UNICEF. Those are measured on
the real instruments and should not be compared line-for-line with the table above. The distance on
UNHCR is the honest headline: on generated data, local methods top out near 0.18 there while the field
sits at 0.88 on real surveys of displaced households. Real gating patterns, real regional structure and
real-world plausibility are where that difference lives, and measuring them means submitting.

## Running it

Get the competition's starter kit: <https://github.com/SituatedEvals/public>. Put this repository's
`model/` directory beside the kit's own submissions and run their scorer:

```bash
python3 score.py --submission path/to/this/model --data _sandbox/unhcr --schema data/unhcr.json --phase 1
```

Run the self-test from this directory:

```bash
python3 selftest.py                 # structure and prior files, no data needed
python3 selftest.py --kit ../public # also checks the calibration against their grader
```

Only numpy and pandas. No model is loaded at run time — the language model's contribution is a table.

## Limits, stated plainly

It predicts from what a respondent's row already contains. It cannot know anything about a household
that was never asked. It treats each question independently once the conditioning is done, so it does
not model a household's answers as one consistent story. The exact-gate rule applies only where a
gate's parent is itself visible — which is 1 of 156 gated questions on UNHCR and 27 of 30 on World
Bank, and elsewhere the model has to learn the pattern from counts instead.

No licence has been chosen for this code yet.

Built by Alex at SQOPA as an entry to SimulacraBench. The competition, its data and its scoring belong
to Stanford HAI, UNHCR, UNICEF and the World Bank; their starter kit is the harness this code runs in.
