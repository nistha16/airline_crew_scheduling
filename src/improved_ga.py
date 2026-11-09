import random
from dataclasses import dataclass

from utils import SolutionState, repair_search, greedy_fallback


@dataclass(frozen=True)
class Params:
    pop_size: int = 60
    generations: int = 300
    pc: float = 0.9
    pm: float = 0.003
    pf: float = 0.45
    improve_iters: int = 300
    init_steps: int = 600
    init_p_random: float = 0.2


def stochastic_ranking(pop, pf, rng):
    idx = list(range(len(pop)))
    for _ in range(len(pop)):
        swapped = False
        for j in range(len(idx) - 1):
            a, b = pop[idx[j]], pop[idx[j + 1]]
            use_obj = (rng.random() < pf) or (a.viol_value == 0 and b.viol_value == 0)
            if use_obj:
                if a.cost_value > b.cost_value:
                    idx[j], idx[j + 1] = idx[j + 1], idx[j]
                    swapped = True
            else:
                if a.viol_value > b.viol_value:
                    idx[j], idx[j + 1] = idx[j + 1], idx[j]
                    swapped = True
        if not swapped:
            break
    return [pop[i] for i in idx]


def one_point_crossover(a, b, rng):
    p = rng.randrange(1, len(a))
    return a[:p] + b[p:], b[:p] + a[p:]


def mutate(x, pm, rng):
    for j in range(len(x)):
        if rng.random() < pm:
            x[j] = 1 - x[j]


def prune(st, rng, passes=10):
    inst = st.inst
    for _ in range(passes):
        selected = [j for j in range(inst.n) if st.x[j] == 1]
        rng.shuffle(selected)
        changed = False
        for j in selected:
            if all(st.cover[i] >= 2 for i in inst.colRows[j]):
                st.flip(j)
                changed = True
        if not changed:
            break
    return st


def pseudo_random_init(inst, rng, steps, p_random):
    st = SolutionState.empty(inst)
    for _ in range(steps):
        uncovered = [i for i in range(inst.m) if st.cover[i] == 0]
        if not uncovered:
            break
        row = rng.choice(uncovered)
        if rng.random() < p_random:
            j = rng.choice(inst.rowCols[row])
        else:
            best_j, best_score = None, float("inf")
            for j2 in inst.rowCols[row]:
                newly = sum(1 for r in inst.colRows[j2] if st.cover[r] == 0)
                if newly <= 0:
                    continue
                score = inst.cost[j2] / newly
                if score < best_score:
                    best_score = score
                    best_j = j2
            if best_j is None:
                best_j = rng.choice(inst.rowCols[row])
            j = best_j
        st.flip(j)
    return prune(st, rng)


def improve(inst, st, rng, iters):
    prev_viol = None
    for _ in range(iters):
        uncovered = [i for i in range(inst.m) if st.cover[i] == 0]
        if uncovered:
            row = rng.choice(uncovered)
            best_j, best_score = None, float("inf")
            for j in inst.rowCols[row]:
                if st.x[j] == 1:
                    continue
                overlap = sum(1 for r in inst.colRows[j] if st.cover[r] >= 1)
                score = inst.cost[j] + 500 * overlap
                if score < best_score:
                    best_score = score
                    best_j = j
            if best_j is None:
                best_j = rng.choice(inst.rowCols[row])
            st.flip(best_j)
            continue

        prune(st, rng)

        if st.viol_value > 0:
            if prev_viol is not None and st.viol_value >= prev_viol:
                break
            prev_viol = st.viol_value
            over = [i for i in range(inst.m) if st.cover[i] > 1]
            row = rng.choice(over)
            cols = [j for j in inst.rowCols[row] if st.x[j] == 1]
            j = max(cols, key=lambda jj: inst.cost[jj])
            st.flip(j)
            continue

        break
    return st


def run_improved_bga(inst, seed, params=Params()):
    rng = random.Random(seed)

    pop = []
    for _ in range(params.pop_size):
        st = pseudo_random_init(inst, rng, params.init_steps, params.init_p_random)
        st = improve(inst, st, rng, params.improve_iters // 3)
        pop.append(st)

    best_cost = None

    def update_best(st):
        nonlocal best_cost
        if st.is_feasible():
            if best_cost is None or st.cost_value < best_cost:
                best_cost = st.cost_value

    for st in pop:
        update_best(st)

    for _ in range(params.generations):
        ranked = stochastic_ranking(pop, params.pf, rng)

        children = []
        while len(children) < params.pop_size:
            p1 = rng.choice(ranked[:params.pop_size // 2])
            p2 = rng.choice(ranked[:params.pop_size // 2])

            if rng.random() < params.pc:
                c1x, c2x = one_point_crossover(p1.x, p2.x, rng)
            else:
                c1x, c2x = p1.x.copy(), p2.x.copy()

            mutate(c1x, params.pm, rng)
            mutate(c2x, params.pm, rng)

            c1 = SolutionState.from_x(inst, c1x)
            c2 = SolutionState.from_x(inst, c2x)

            prune(c1, rng)
            prune(c2, rng)

            if c1.viol_value > 0:
                c1 = improve(inst, c1, rng, params.improve_iters)
                if not c1.is_feasible():
                    c1 = repair_search(inst, c1, rng, steps=80)
            if c2.viol_value > 0:
                c2 = improve(inst, c2, rng, params.improve_iters)
                if not c2.is_feasible():
                    c2 = repair_search(inst, c2, rng, steps=80)

            children.append(c1)
            if len(children) < params.pop_size:
                children.append(c2)

        pop = stochastic_ranking(pop + children, params.pf, rng)[:params.pop_size]

        for st in pop:
            update_best(st)

    if best_cost is None:
        best_cost = greedy_fallback(inst, seed)
    return best_cost
