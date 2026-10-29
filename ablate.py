"""Ablation: run the model with each layer switched off, through the competition's own grader.

Four configurations, the same code, differences only in the environment variables the model reads:

  marginals      no conditioning, no gate rule, no prior — the grader's own zero, and it must land there
  + counts       joint-key counts with support-weighted backoff
  + gate rule    the never-asked slot taken from the instrument where the instrument states it
  full           with the bundled language-model prior under the pooled marginal

Anything that does not change the score on the practice data is not proven by this data. The practice
answers are generated, so this table measures what the practice harness can see, which is structure —
not whether a prior that is right about the world earns its place.

    python3 ablate.py --kit ../public
"""
import argparse, datetime, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    ("marginals", dict(SIM_MAX_FEATURES="0", SIM_CLOSED_GATE="0", SIM_PRIOR_WEIGHT="0")),
    ("+ counts", dict(SIM_MAX_FEATURES="3", SIM_CLOSED_GATE="0", SIM_PRIOR_WEIGHT="0")),
    ("+ gate rule", dict(SIM_MAX_FEATURES="3", SIM_CLOSED_GATE="1", SIM_PRIOR_WEIGHT="0")),
    ("full model", dict(SIM_MAX_FEATURES="3", SIM_CLOSED_GATE="1", SIM_PRIOR_WEIGHT="5")),
]
INSTRUMENTS = ["unhcr", "world_bank", "unicef"]


def skill(kit, instrument, overrides):
    env = dict(os.environ, **overrides)
    run = subprocess.run(
        [sys.executable, os.path.join(kit, "score.py"), "--submission", os.path.join(HERE, "model"),
         "--data", os.path.join(kit, "_sandbox", instrument),
         "--schema", os.path.join(kit, "data", f"{instrument}.json"), "--phase", "1"],
        capture_output=True, text=True, env=env, cwd=kit, timeout=1800)
    for line in (run.stdout + run.stderr).splitlines():
        if "PASS" in line or "FAIL" in line:
            parts = line.split()
            try:
                return float(parts[1])
            except (IndexError, ValueError):
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, help="path to the competition starter kit")
    ap.add_argument("--out", default=os.path.join(HERE, "measurements", "ablation.json"))
    args = ap.parse_args()

    table = {inst: {} for inst in INSTRUMENTS}
    for inst in INSTRUMENTS:
        for name, overrides in CONFIGS:
            table[inst][name] = skill(args.kit, inst, overrides)
            print(f"{inst:12s} {name:14s} {table[inst][name] if table[inst][name] is not None else 'no score'}")

    print(f"\n{'':16s}" + "".join(f"{i:>13s}" for i in INSTRUMENTS))
    for name, _ in CONFIGS:
        print(f"{name:16s}" + "".join(f"{table[i][name]:13.4f}" for i in INSTRUMENTS))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"measured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "phase": 1, "practice_data": "generated",
                   "results": table}, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
