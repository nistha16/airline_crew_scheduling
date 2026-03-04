import math
import random
from dataclasses import dataclass

from utils import penalty_energy, greedy_constructor, prune_redundant_columns, greedy_fallback


@dataclass(frozen=True)
class SAParams:
    lam: float = 1500.0
    T0: float = 1500.0
    Tmin: float = 1e-3
    alpha: float = 0.9995
    L: int = 300
    max_evals: int = 500000


def run_sa(inst, seed, params=SAParams()):
    rng = random.Random(seed)

    state = greedy_constructor(inst, seed=seed)
    state = prune_redundant_columns(state, seed=seed)

    best_cost = state.cost_value if state.is_feasible() else None
    E = penalty_energy(state, params.lam)
    T = params.T0
    evals = 0

    while T > params.Tmin and evals < params.max_evals:
        for _ in range(params.L):
            if evals >= params.max_evals:
                break

            violated = [i for i, c in enumerate(state.cover) if c != 1]

            if violated:
                # target violated rows — cheap single flip
                i = rng.choice(violated)
                j = rng.choice(inst.rowCols[i])

                delta_cost, delta_viol, _ = state._flip_updates(j)
                E_new = E + delta_cost + params.lam * delta_viol
                dE = E_new - E

                if dE <= 0 or rng.random() < math.exp(-dE / T):
                    state.flip(j)
                    E = E_new
                    if state.is_feasible():
                        if best_cost is None or state.cost_value < best_cost:
                            best_cost = state.cost_value
                evals += 1

            else:
                # feasible: 30% compound move, 70% simple flip of a selected column
                selected = [j for j in range(inst.n) if state.x[j] == 1]
                if not selected:
                    evals += 1
                    continue

                if rng.random() < 0.3:
                    # compound move — remove + greedy repair
                    candidate = state.copy()
                    candidate.flip(rng.choice(selected))

                    for row in [i for i in range(inst.m) if candidate.cover[i] == 0]:
                        best_j, best_score = None, float("inf")
                        for j2 in inst.rowCols[row]:
                            if candidate.x[j2] == 1:
                                continue
                            newly = sum(1 for r in inst.colRows[j2] if candidate.cover[r] == 0)
                            if newly <= 0:
                                continue
                            score = inst.cost[j2] / newly
                            if score < best_score:
                                best_score = score
                                best_j = j2
                        if best_j is not None:
                            candidate.flip(best_j)

                    E_new = penalty_energy(candidate, params.lam)
                    dE = E_new - E

                    if dE <= 0 or rng.random() < math.exp(-dE / T):
                        state = candidate
                        E = E_new
                        if state.is_feasible():
                            if best_cost is None or state.cost_value < best_cost:
                                best_cost = state.cost_value
                else:
                    # simple flip — remove a selected column (will become infeasible)
                    j = rng.choice(selected)
                    delta_cost, delta_viol, _ = state._flip_updates(j)
                    E_new = E + delta_cost + params.lam * delta_viol
                    dE = E_new - E

                    if dE <= 0 or rng.random() < math.exp(-dE / T):
                        state.flip(j)
                        E = E_new

                evals += 1

        T *= params.alpha

    if best_cost is None:
        best_cost = greedy_fallback(inst, seed)
    return best_cost
