import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SPPInstance:
    name: str
    m: int              # number of rows
    n: int              # number of columns
    cost: list
    colRows: list       # for each column j: list of covered rows (0-based)
    rowCols: list       # for each row i: list of columns that cover it
    row_degree: list    # len(rowCols[i])


class SPPFormatError(ValueError):
    pass


def load_spp_instance(file_path, strict=True):
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = [int(t) for t in text.split()]
    idx = 0

    def next_int():
        nonlocal idx
        if idx >= len(tokens):
            raise SPPFormatError(f"{path}: unexpected end of file.")
        v = tokens[idx]
        idx += 1
        return v

    m = next_int()
    n = next_int()

    if m <= 0 or n <= 0:
        raise SPPFormatError(f"{path}: invalid header m={m}, n={n}.")

    cost = []
    colRows = []
    rowCols = [[] for _ in range(m)]

    for j in range(n):
        c = next_int()
        k = next_int()
        rows_1_based = [next_int() for _ in range(k)]

        seen = set()
        rows_0_based = []
        for r in rows_1_based:
            if r <= 0 or r > m:
                raise SPPFormatError(f"{path}: column {j+1}: row {r} out of range.")
            if r in seen:
                if strict:
                    raise SPPFormatError(f"{path}: column {j+1}: duplicate row {r}.")
                continue
            seen.add(r)
            rows_0_based.append(r - 1)

        cost.append(c)
        colRows.append(rows_0_based)
        for i in rows_0_based:
            rowCols[i].append(j)

    if strict and idx != len(tokens):
        raise SPPFormatError(f"{path}: {len(tokens) - idx} extra tokens after parsing.")

    row_degree = [len(cols) for cols in rowCols]
    uncovered = [i for i, deg in enumerate(row_degree) if deg == 0]
    if uncovered and strict:
        raise SPPFormatError(f"{path}: {len(uncovered)} rows not covered by any column.")

    return SPPInstance(
        name=path.stem, m=m, n=n,
        cost=cost, colRows=colRows, rowCols=rowCols, row_degree=row_degree,
    )


@dataclass
class SolutionState:
    inst: SPPInstance
    x: list
    cover: list
    cost_value: int
    viol_value: int     # sum_i |cover[i]-1|, 0 means feasible

    @staticmethod
    def empty(inst):
        return SolutionState(inst, [0] * inst.n, [0] * inst.m, 0, inst.m)

    @staticmethod
    def from_x(inst, x):
        cover = [0] * inst.m
        cost_value = 0
        for j, bit in enumerate(x):
            if bit:
                cost_value += inst.cost[j]
                for i in inst.colRows[j]:
                    cover[i] += 1
        viol = sum(abs(c - 1) for c in cover)
        return SolutionState(inst, list(x), cover, cost_value, viol)

    def is_feasible(self):
        return self.viol_value == 0

    def _flip_updates(self, j):
        turning_on = (self.x[j] == 0)
        delta_cost = self.inst.cost[j] if turning_on else -self.inst.cost[j]
        affected = self.inst.colRows[j]
        step = 1 if turning_on else -1
        delta_viol = sum(
            abs(self.cover[i] + step - 1) - abs(self.cover[i] - 1)
            for i in affected
        )
        return delta_cost, delta_viol, affected

    def flip(self, j):
        turning_on = (self.x[j] == 0)
        step = 1 if turning_on else -1
        delta_cost, delta_viol, affected = self._flip_updates(j)
        self.x[j] = 1 if turning_on else 0
        for i in affected:
            self.cover[i] += step
        self.cost_value += delta_cost
        self.viol_value += delta_viol

    def copy(self):
        return SolutionState(
            self.inst, self.x.copy(), self.cover.copy(),
            self.cost_value, self.viol_value,
        )


def penalty_energy(state, lam):
    return state.cost_value + lam * state.viol_value


def greedy_constructor(inst, seed=None):
    rng = random.Random(seed)
    st = SolutionState.empty(inst)

    while True:
        uncovered = [i for i in range(inst.m) if st.cover[i] == 0]
        if not uncovered:
            break
        row = rng.choice(uncovered)

        best_j, best_score = None, float("inf")
        for j in inst.rowCols[row]:
            if st.x[j] == 1:
                continue
            newly = sum(1 for i in inst.colRows[j] if st.cover[i] == 0)
            if newly <= 0:
                continue
            score = inst.cost[j] / newly
            if score < best_score:
                best_score = score
                best_j = j

        if best_j is None:
            best_j = rng.choice(inst.rowCols[row])

        st.flip(best_j)

    return st


def prune_redundant_columns(state, seed=None, max_passes=10):
    rng = random.Random(seed)
    inst = state.inst

    for _ in range(max_passes):
        selected = [j for j in range(inst.n) if state.x[j] == 1]
        rng.shuffle(selected)
        changed = False
        for j in selected:
            if all(state.cover[i] >= 2 for i in inst.colRows[j]):
                state.flip(j)
                changed = True
        if not changed:
            break

    return state


def repair_search(inst, st, rng, steps=200):
    best = st.copy()

    for _ in range(steps):
        if st.is_feasible():
            return st

        if (st.viol_value < best.viol_value) or (st.viol_value == best.viol_value and st.cost_value < best.cost_value):
            best = st.copy()

        uncovered = [i for i in range(inst.m) if st.cover[i] == 0]
        if uncovered:
            i = rng.choice(uncovered)
            cand = [j for j in inst.rowCols[i] if st.x[j] == 0]
            if not cand:
                continue
            j = min(cand, key=lambda jj: inst.cost[jj])
            st.flip(j)
            continue

        over = [i for i in range(inst.m) if st.cover[i] > 1]
        if over:
            i = rng.choice(over)
            cand = [j for j in inst.rowCols[i] if st.x[j] == 1]
            if not cand:
                continue
            j = max(cand, key=lambda jj: inst.cost[jj])
            st.flip(j)
            continue

        break
    return best


def greedy_fallback(inst, seed):
    rng = random.Random(seed)
    # try greedy + prune + repair search
    for s in range(seed, seed + 50):
        st = greedy_constructor(inst, seed=s)
        st = prune_redundant_columns(st, seed=s)
        if st.is_feasible():
            return st.cost_value
        st = repair_search(inst, st, rng)
        if st.is_feasible():
            return st.cost_value
    return None
