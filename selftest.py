"""Self-test for the survey-completion model. Two kinds of check, both two-sided.

1. Structure, on a fixture built here: a two-branch instrument with one gated question, so the
   claims that matter can be checked without any competition data — one vector per missing cell, in
   the order the grader reads, legal probabilities, and a closed branch carrying the never-asked mass.
2. Calibration, against the competition's own grader when it is available (`--kit ../public`):
   switched down to marginals only this model must land at zero, because zero is what the metric says
   a model that only knows population answer rates gets; and with its structure on it must do better
   than that. A check that only ever passes is not a check.

    python3 selftest.py
    python3 selftest.py --kit ../public
"""
import argparse, csv, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "model"))


def fixture():
    """A small instrument with a visible segment, an open question and a gated one."""
    gated_value = "NA_GATED"
    schema = {
        "gated_value": gated_value,
        "dataset": {"description": "fixture"},
        "items": {
            "segment": {"class": "GIVEN", "values": ["north", "south"], "question": "Region",
                        "gate": None},
            "ask_any": {"class": "PREDICT", "values": ["yes", "no"], "question": "A general question",
                        "gate": None},
            "ask_only_north": {"class": "PREDICT", "values": ["often", "sometimes", "never"],
                               "question": "A question asked only in the north",
                               "gate": {"parent": "segment", "observed_if": ["north"]}},
        },
    }
    rows = []
    for i in range(120):                      # visible rows: complete answers
        north = i % 3 != 0
        seg = "north" if north else "south"
        rows.append({"respondent_id": f"v{i}", "role": "TRAIN", "segment": seg,
                     "ask_any": "yes" if i % 2 else "no",
                     "ask_only_north": ("often" if i % 5 else "never") if north else gated_value})
    for i in range(40):                       # scored rows: what is missing is missing
        seg = "north" if i % 2 else "south"
        rows.append({"respondent_id": f"h{i}", "role": "DEV", "segment": seg,
                     "ask_any": np.nan, "ask_only_north": np.nan})
    return schema, pd.DataFrame(rows)


def check_structure():
    import importlib
    model = importlib.import_module("main")
    schema, frame = fixture()
    items = list(schema["items"])
    grid = frame[items].to_numpy(dtype=object)
    wanted = [(row, col) for row in range(len(frame)) for col in range(len(items))
              if pd.isna(grid[row, col])]
    vectors = model.predict(frame, schema)

    problems = []
    if len(vectors) != len(wanted):
        problems.append(f"returned {len(vectors)} vectors for {len(wanted)} missing cells")
    for (row, col), vec in zip(wanted, vectors):
        item = items[col]
        width = len(schema["items"][item]["values"])
        if schema["items"][item].get("gate"):
            width += 1
        vec = np.asarray(vec, dtype=float)
        if vec.shape != (width,):
            problems.append(f"{item}: vector of length {vec.shape} for {width} options")
            continue
        if (vec < 0).any():
            problems.append(f"{item}: negative probability")
        if abs(vec.sum() - 1.0) > 1e-6:
            problems.append(f"{item}: row sums to {vec.sum():.6f}")
    # The deterministic rule: a southern household cannot have been asked the northern question.
    south = [i for i, (row, col) in enumerate(wanted)
             if items[col] == "ask_only_north" and frame["segment"].iat[row] == "south"]
    if not south:
        problems.append("fixture produced no closed-branch cells to check")
    else:
        sentinel = len(schema["items"]["ask_only_north"]["values"])
        worst = min(float(np.asarray(vectors[i], dtype=float)[sentinel]) for i in south)
        if worst < 0.99:
            problems.append(f"closed branch put only {worst:.3f} on the never-asked slot")
    return problems, len(wanted), len(south)


def check_priors():
    problems = []
    folder = os.path.join(HERE, "model", "priors")
    if not os.path.isdir(folder):
        return ["no bundled priors directory"], 0
    total = 0
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        by_item = {}
        with open(os.path.join(folder, name), newline="") as fh:
            for row in csv.DictReader(fh):
                by_item.setdefault(row["item"], []).append((row["p"], row["letter"]))
        total += len(by_item)
        for item, rows in by_item.items():
            probs = [float(p) for p, _ in rows]
            letters = [l for _, l in rows]
            if len(set(letters)) != len(letters):
                problems.append(f"{name}:{item}: repeated option letters")
            if abs(sum(probs) - 1.0) > 1e-3:
                problems.append(f"{name}:{item}: prior sums to {sum(probs):.4f}")
            if any(p < 0 for p in probs):
                problems.append(f"{name}:{item}: negative prior weight")
    return problems, total


def check_kit_calibration(kit, instruments):
    """Off, this model must score the grader's zero. On, it must do better than off."""
    model_dir = os.path.join(HERE, "model")
    results = []
    for inst in instruments:
        data = os.path.join(kit, "_sandbox", inst)
        schema = os.path.join(kit, "data", f"{inst}.json")
        if not (os.path.isdir(data) and os.path.isfile(schema)):
            results.append((inst, None, None, f"kit data missing for {inst}"))
            continue
        env = dict(os.environ, SIM_MAX_FEATURES="0", SIM_PRIOR_WEIGHT="0", SIM_CLOSED_GATE="0")
        flat = subprocess.run([sys.executable, os.path.join(kit, "score.py"), "--submission", model_dir,
                              "--data", data, "--schema", schema, "--phase", "1"],
                             capture_output=True, text=True, env=env, cwd=kit, timeout=1200)
        on = subprocess.run([sys.executable, os.path.join(kit, "score.py"), "--submission", model_dir,
                            "--data", data, "--schema", schema, "--phase", "1"],
                            capture_output=True, text=True, cwd=kit, timeout=1200)
        flat_skill = _skill(flat.stdout + flat.stderr)
        on_skill = _skill(on.stdout + on.stderr)
        results.append((inst, flat_skill, on_skill, (flat.stdout + flat.stderr)[-500:]))
    return results


def _skill(text):
    for line in text.splitlines():
        if "PASS" in line:
            parts = line.split()
            try:
                return float(parts[1])
            except (IndexError, ValueError):
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", help="path to the competition starter kit, for the calibration check")
    args = ap.parse_args()

    problems, cells, closed = check_structure()
    print(f"structure: {cells} missing cells in the fixture, {closed} of them a closed branch")
    print("structure:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("   ", p)

    priors_problems, n_items = check_priors()
    print(f"bundled priors: {n_items} questions checked;",
          "PASS" if not priors_problems else "FAIL")
    for p in priors_problems:
        print("   ", p)

    if args.kit:
        results = check_kit_calibration(args.kit, ["unhcr", "world_bank", "unicef"])
        calibration_ok = True
        for inst, flat, on, tail in results:
            if flat is None or on is None:
                print(f"calibration {inst}: could not read a score from the grader")
                print("   the marginals-only run said:", tail)
                calibration_ok = False
                continue
            ok = abs(flat) < 0.02 and on > flat
            calibration_ok &= ok
            print(f"calibration {inst}: marginals-only {flat:.4f} (want ~0), with structure {on:.4f}",
                  "PASS" if ok else "FAIL")
    else:
        print("calibration: not run (pass --kit <starter kit> to run it)")

    ok = not problems and not priors_problems
    print("overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
