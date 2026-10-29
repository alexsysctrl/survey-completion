"""v8: counts that learn whether a question was asked, the instrument's determinism applied exactly,
and a language model's reading of each question underneath the counts.

Three layers, coarsest to finest, each one trusted only as far as its evidence reaches:

  1. A prior: for every scored question, what a general model says a respondent like this would
     answer, read from the option letters' first-token logits. Bundled as a table, computed offline
     from a small instruct model — the same role their `baseline/external_model` plays with a model
     loaded in the container, without needing a model or a GPU at scoring time. It enters as
     PRIOR_WEIGHT pseudo-respondents under the pooled marginal, so a wrong prior is outvoted by the
     first real counts rather than believed.
  2. The pooled marginal of the visible rows.
  3. Joint keys over conditioning columns the respondent actually answered, each level blended by
     support: counts over a combination carry weight in proportion to how many people stand behind
     them, and everything thinner backs off toward the level above.

On top: where the instrument is deterministic, it is followed exactly. A gated question whose parent
is a visible column and whose parent's answer is outside `observed_if` was not asked — that is not an
estimate, and the never-asked slot gets the mass.

Practice-data note, because it bounds what these numbers can mean: the delivered practice data is
generated — real questions, real gates, invented answer distributions. It can show the structure is
right and that the whole thing runs inside the time budget. Whether a prior that is right about the
world helps is a question the real instruments answer, on their side.
"""

import csv
import os
import numpy as np
import pandas as pd

# Every setting can be overridden from the environment, which is how the ablations in selftest.py
# run and how a reader can see what each layer is actually worth. Inside the competition container
# nothing is set, so the defaults below are what gets submitted.
def _setting(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return type(default)(raw)


MAX_FEATURES = _setting("SIM_MAX_FEATURES", 3)     # conditioning columns per item, joint prefixes only
M_SUPPORT = _setting("SIM_M_SUPPORT", 20.0)         # observations a combination needs to be believed
ALPHA = _setting("SIM_ALPHA", 0.5)                  # Laplace floor inside a level
MIN_SUPPORT = _setting("SIM_MIN_SUPPORT", 40)       # a level this thin is dropped, not believed
CLOSED_GATE = _setting("SIM_CLOSED_GATE", 1)        # follow the instrument where it is determinism
PRIOR_WEIGHT = _setting("SIM_PRIOR_WEIGHT", 5.0)    # what the language model is worth, in people
VALID_FRAC = 0.2
SEED = 20261029
HERE = os.path.dirname(os.path.abspath(__file__))


def _codes(series, levels):
    lookup = {level: i for i, level in enumerate(levels)}
    return series.map(lookup).fillna(-1).astype(np.int64).to_numpy()


def _key(codes_list):
    """Mixed-radix joint key over one table's rows; -1 where any part is unknown.

    A column's radix comes from its own maximum code across all rows, so the same combination of
    answers maps to the same key on fit rows and predicted rows alike.
    """
    rows = len(codes_list[0])
    known = np.ones(rows, dtype=bool)
    key = np.zeros(rows, dtype=np.int64)
    for codes in codes_list:
        span = int(codes.max()) + 2
        known &= codes >= 0
        key = key * span + np.where(codes >= 0, codes + 1, 0)
    return np.where(known, key, -1)


def _level(key, fit_rows, truth, width):
    seen = key[fit_rows]
    known = seen >= 0
    if not known.any():
        return None
    uniq, inv = np.unique(seen[known], return_inverse=True)
    t = truth[fit_rows][known]
    ok = t >= 0
    joint = np.zeros((len(uniq), width), dtype=float)
    np.add.at(joint, (inv[ok], t[ok]), 1.0)
    if joint.sum() < MIN_SUPPORT:
        return None
    return uniq, joint


def _apply(key, rows, level, parent, base):
    uniq, joint = level
    q = key[rows]
    idx = np.searchsorted(uniq, q)
    in_table = (idx < len(uniq)) & (np.take(uniq, np.clip(idx, 0, len(uniq) - 1)) == q) & (q >= 0)
    row_tot = joint.sum(axis=1, keepdims=True)
    cond = (joint + ALPHA) / (row_tot + ALPHA * joint.shape[1])
    safe = np.clip(idx, 0, len(uniq) - 1)
    support = joint[safe].sum(axis=1)
    w = np.where(in_table[:, None], (support / (support + M_SUPPORT))[:, None], 0.0)
    return w * cond[safe] + (1.0 - w) * parent


def _chain(features, ctx, fit_rows, predict_rows, truth, width, base):
    p = np.tile(base, (len(predict_rows), 1))
    used = []
    for depth in range(1, len(features) + 1):
        key = _key([ctx[c] for c in features[:depth]])
        level = _level(key, fit_rows, truth, width)
        if level is None:
            break
        p = _apply(key, predict_rows, level, p, base)
        used.append(features[depth - 1])
    return used, p


def load_priors(schema):
    """The bundled prior table that belongs to this instrument, chosen by item-name overlap.

    The schema is what identifies the instrument at scoring time — there is no name in it that a
    submission is told to trust — so the match is made on the questions themselves.
    """
    folder = os.path.join(HERE, "priors")
    if not os.path.isdir(folder):
        return {}
    items = set(schema["items"])
    best, best_overlap = {}, 0
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        table = {}
        with open(os.path.join(folder, name), newline="") as fh:
            for row in csv.DictReader(fh):
                table.setdefault(row["item"], {})[row["option"]] = float(row["p"])
        overlap = len(set(table) & items)
        if overlap > best_overlap:
            best, best_overlap = table, overlap
    return best if best_overlap >= max(3, int(0.5 * len(items))) else {}


def predict(frame, schema):
    gated_value = schema["gated_value"]
    items = [n for n, r in schema["items"].items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if schema["items"][n]["class"] == "GIVEN"]
    targets = [n for n in items if schema["items"][n]["class"] == "PREDICT"]
    values = {n: list(schema["items"][n]["values"]) for n in items}
    vocab = {n: values[n] + ([gated_value] if schema["items"][n].get("gate") else []) for n in items}
    sentinel = {n: (len(values[n]) if schema["items"][n].get("gate") else -1) for n in items}
    gate = {n: schema["items"][n].get("gate") for n in items}
    priors = load_priors(schema)

    n = len(frame)
    visible = frame.loc[~frame[targets].isna().any(axis=1)]
    if len(visible) < 50:
        visible = frame
    vis_idx = np.flatnonzero(frame.index.isin(visible.index))
    rng = np.random.default_rng(SEED)
    perm = vis_idx.copy()
    rng.shuffle(perm)
    cut = int(len(perm) * VALID_FRAC)
    select_rows, build_rows = perm[:cut], perm[cut:]

    ctx = {g: _codes(frame[g], vocab[g]) for g in given}
    tgt = {t: _codes(frame[t], vocab[t]) for t in targets}
    in_visible = np.zeros(n, dtype=bool)
    in_visible[vis_idx] = True

    dist = {}
    for item in targets:
        width = len(vocab[item])
        truth = tgt[item]
        fit_pool = np.flatnonzero(in_visible & (truth >= 0))
        counts = np.bincount(truth[fit_pool], minlength=width).astype(float)

        # Pseudocounts over the answer options only. Whether a branch was reached is not something
        # a general model can know about a particular respondent, so the never-asked slot is left out
        # of the prior and comes from the counts or the gate instead.
        prior = np.zeros(width)
        got = priors.get(item)
        asked_slots = [i for i in range(width) if i != sentinel[item]]
        for i in asked_slots:
            if got and vocab[item][i] in got:
                prior[i] = got[vocab[item][i]]
        total = prior.sum()
        if total > 0:
            prior = prior / total
        else:
            for i in asked_slots:
                prior[i] = 1.0 / max(len(asked_slots), 1)
        base = counts + ALPHA + PRIOR_WEIGHT * prior
        base = base / base.sum()

        selected, best_log = [], None
        while len(selected) < MAX_FEATURES:
            best = None
            for cand in given:
                if cand in selected:
                    continue
                used, p = _chain(selected + [cand], ctx, build_rows, select_rows, truth, width, base)
                if not used:
                    continue
                val = truth[select_rows]
                ok = val >= 0
                if ok.sum() < 20:
                    continue
                score = float(np.log(np.clip(p[np.arange(len(val))[ok], val[ok]], 1e-12, 1.0)).mean())
                if best is None or score > best[0]:
                    best = (score, cand)
            if best is None or (best_log is not None and best[0] <= best_log + 1e-4):
                break
            best_log, cand = best
            selected.append(cand)

        if selected:
            _used, p = _chain(selected, ctx, fit_pool, np.arange(n), truth, width, base)
        else:
            p = np.tile(base, (n, 1))

        g = gate[item]
        if CLOSED_GATE and g and schema["items"][g["parent"]]["class"] == "GIVEN":
            closed = ~np.isin(frame[g["parent"]].to_numpy(dtype=object), g["observed_if"])
            s = sentinel[item]
            if s >= 0 and closed.any():
                onehot = np.zeros(width)
                onehot[s] = 1.0
                p[closed] = onehot
        dist[item] = p

    grid = frame[items].to_numpy(dtype=object)
    return [dist[items[column]][row]
            for row in range(n)
            for column in range(len(items))
            if pd.isna(grid[row, column])]
