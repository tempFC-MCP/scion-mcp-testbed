"""Check the scheduling run against its own inputs, schedules and solver bounds."""
import csv
import json
from math import ceil, isclose
from pathlib import Path
from statistics import mean

import numpy as np

from scheduler import metrics, weighted_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_scores(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def check_geographic_optima(adoption):
    # Adoption records identify the 24 main cases separately from the sensitivity runs.
    cases = {row["instance"]: {} for row in adoption
             if row["group"] == "geographic" and row["candidates"] == ["full"]}
    for row in json.loads((RESULTS / "experiments.json").read_text())["results"]:
        if row["instance"] in cases and row["method"] in ("full", "MILP"):
            cases[row["instance"]][row["method"]] = row

    gaps = []
    certified = exact = 0
    for case, runs in cases.items():
        instance = json.loads((RESULTS / f"instance-{case:02d}.json").read_text())
        for key in ("q", "regions", "F", "budget"):
            instance[key] = np.asarray(instance[key])
        rounds = instance["T"]
        regional_turns = np.zeros(instance["nr"], dtype=int)
        np.add.at(regional_turns, instance["regions"], instance["q"])
        missing = max(0, instance["g"] * rounds - np.minimum(regional_turns, rounds).sum())
        # Across all rounds, each outage can absorb at most rounds * budget exposed turns.
        excess = np.maximum(0, instance["F"] @ instance["q"] - rounds * instance["budget"]).sum()
        lower = weighted_score(dict(geo_deficit=int(missing), excess=int(excess), window_misses=0))
        swap = runs["full"]
        swap_score = weighted_score(metrics(swap["schedule"], instance))
        solver = runs["MILP"]
        best = swap_score
        if solver.get("schedule") is not None:
            solver_score = weighted_score(metrics(solver["schedule"], instance))
            best = min(best, solver_score)
        if solver['dual_bound'] is not None:
            lower = max(lower, ceil(solver['dual_bound'] - 1e-6))
        if best < lower:
            raise ValueError(f"Instance {case}: valid schedule scores below lower bound {lower}")
        optimum = None
        if solver["status"] == 0:
            if (solver.get("schedule") is None or solver_score != best
                    or not isclose(solver["objective"], solver_score, abs_tol=1e-6)):
                raise ValueError(f"Instance {case}: MILP optimum disagrees with checked schedules")
            optimum = best
        elif best == lower:
            optimum = best
        if optimum is not None:
            certified += 1
            exact += int(swap_score == optimum)
            if optimum > 0:
                gaps.append(100 * (swap_score - optimum) / optimum)
    print(f"Geographic optima certified: {certified}/{len(cases)}; swaps attained {exact}.")
    if gaps:
        print(f"Over positive certified optima, mean gap {mean(gaps):.1f}%, maximum {max(gaps):.1f}%.")


def main():
    scores = read_scores(RESULTS / "figure2-exposure.csv")
    adoption = json.loads((RESULTS / "adoption.json").read_text())
    check_geographic_optima(adoption)
    design = {(n, N, seed) for n in (50, 100, 300, 600) for N in (4, 8) for seed in range(3)}
    for regime in ("Geographic", "Local"):
        cases = [row for row in scores if row["regime"] == regime]
        keys = {(int(row["validators"]), int(row["proposers"]), int(row["seed"])) for row in cases}
        if keys != design or len(cases) != len(design):
            raise ValueError(f"Incomplete Figure 2 results for {regime}")
        swaps = [row for row in adoption if row["group"] == regime.lower()
                 and row["candidates"] == ["full"]]
        if len(swaps) != len(design) or any(r["adopted"] not in ("original", "full") for r in swaps):
            raise ValueError(f"Incomplete adoption records for {regime}")
        baseline = mean(int(row["baseline_excess"]) for row in cases)
        infra = mean(int(row["infrastructure_excess"]) for row in cases)
        adopted = sum(row["adopted"] == "full" for row in swaps)
        print(f"{regime}: mean exposure {baseline:.2f} -> {infra:.2f}; "
              f"{adopted}/{len(cases)} swap candidates adopted.")


if __name__ == "__main__":
    main()
