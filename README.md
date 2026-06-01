# beltmatic_optimizer
Beltmatic arithmetic expression synthesizer.

[optimization2.py specific] This module searches for low-cost arithmetic
expressions for integer targets, using add/sub/mul/pow operations weighted
by COST. The search is heuristic (BFS-on-cost for small values, depth-limited
beam search for larger values).

Beltmatic (https://store.steampowered.com/app/2674500) is a game where
the player builds factories on a belt. Numbers must be synthesized from
BASE powers (2**n) using arithmetic operations with weighted costs.

No analytic closed-form solution exists for the weighted-cost optimal
expression problem, hence the DP + heuristic search approach.