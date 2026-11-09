import random
from dataclasses import dataclass

from utils import SolutionState, penalty_energy, greedy_constructor, prune_redundant_columns, repair_search, greedy_fallback


@dataclass(frozen=True)
class BGAParams:
    pop_size: int = 60
    generations: int = 300
    pc: float = 0.9
    pm: float = 0.005
    tournament_k: int = 3
    elite: int = 2
    lam: float = 50000.0
    greedy_frac: float = 1.0


def one_point_crossover(a, b, rng):
    point = rng.randrange(1, len(a))
    return a[:point] + b[point:], b[:point] + a[point:]


def mutate_bits(x, pm, rng):
    for j in range(len(x)):
        if rng.random() < pm:
            x[j] = 1 - x[j]


def tournament_select(pop, rng, k, lam):
    best, best_E = None, float("inf")
    for _ in range(k):
        cand = pop[rng.randrange(len(pop))]
        E = penalty_energy(cand, lam)
        if E < best_E:
            best_E = E
            best = cand
    return best.copy()


def run_standard_bga(inst, seed, params=BGAParams()):
    rng = random.Random(seed)

    # initialization
    pop = []
    greedy_count = int(params.pop_size * params.greedy_frac)

    for i in range(params.pop_size):
        if i < greedy_count:
            st = greedy_constructor(inst, seed=rng.randrange(10**9))
            st = prune_redundant_columns(st, seed=rng.randrange(10**9))
        else:
            x = [1 if rng.random() < 0.05 else 0 for _ in range(inst.n)]
            st = SolutionState.from_x(inst, x)
        pop.append(st)

    best_cost = None
    best_near = None

    def update_best(st):
        nonlocal best_cost, best_near
        if st.is_feasible():
            if best_cost is None or st.cost_value < best_cost:
                best_cost = st.cost_value
        if best_near is None or (st.viol_value, st.cost_value) < (best_near.viol_value, best_near.cost_value):
            best_near = st.copy()

    for st in pop:
        update_best(st)

    # evolution loop
    for _ in range(params.generations):
        pop_sorted = sorted(pop, key=lambda s: penalty_energy(s, params.lam))
        elites = [p.copy() for p in pop_sorted[:params.elite]]

        children = []
        while len(children) < (params.pop_size - params.elite):
            p1 = tournament_select(pop, rng, params.tournament_k, params.lam)
            p2 = tournament_select(pop, rng, params.tournament_k, params.lam)

            if rng.random() < params.pc:
                c1, c2 = one_point_crossover(p1.x, p2.x, rng)
            else:
                c1, c2 = p1.x.copy(), p2.x.copy()

            mutate_bits(c1, params.pm, rng)
            mutate_bits(c2, params.pm, rng)

            s1 = SolutionState.from_x(inst, c1)
            s1 = prune_redundant_columns(s1, seed=rng.randrange(10**9))
            if not s1.is_feasible():
                s1 = repair_search(inst, s1, rng, steps=50)
            s2 = SolutionState.from_x(inst, c2)
            s2 = prune_redundant_columns(s2, seed=rng.randrange(10**9))
            if not s2.is_feasible():
                s2 = repair_search(inst, s2, rng, steps=50)

            children.append(s1)
            if len(children) < (params.pop_size - params.elite):
                children.append(s2)

        pop = elites + children
        for st in pop:
            update_best(st)

    # post-evolution repair on best near-feasible
    if best_cost is None and best_near is not None:
        repaired = repair_search(inst, best_near.copy(), rng, steps=500)
        if repaired.is_feasible():
            best_cost = repaired.cost_value
    if best_cost is None:
        best_cost = greedy_fallback(inst, seed)
    return best_cost
