# Airline Crew Scheduling

Solving the Set Partitioning Problem (SPP) using three metaheuristic algorithms: Simulated Annealing (SA), Binary Genetic Algorithm (BGA), and Improved BGA (IBGA) with stochastic ranking.

Each row must be covered by exactly one column, minimising total cost.

## Files

- `src/utils.py` -- data structures, instance loader, greedy constructor, pruning, repair search
- `src/simulated_annealing.py` -- SA with penalty function and compound moves
- `src/ga_standard.py` -- standard BGA with tournament selection and elitism
- `src/improved_ga.py` -- improved BGA with stochastic ranking, pseudo-random initialisation, and heuristic improvement operator
- `src/test.py` -- runs 30 seeds per algorithm and saves results to CSV
- `data/` -- OR-Library instances (sppnw41, sppnw42, sppnw43)

## How to Run

```
python3 src/test.py data/sppnw41.txt sa
python3 src/test.py data/sppnw42.txt bga
python3 src/test.py data/sppnw43.txt ibga
python3 src/test.py data/sppnw41.txt all   # run all three algorithms
```

No external libraries needed. Python 3 only.
