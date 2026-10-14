"""Validate the distribution-aware DFS formula structure for the noisy MUX.

The exhaustive oracle validates DFSScore, ProbDFS, MinimumProbDFS, and MDFS.
This module checks that the corresponding runtime definitions compose and
parse without executing their deeply nested exhaustive searches.
"""

from qfoil import QFOIL


def test_probabilistic_dfs_runtime_definitions_parse():
    parser = QFOIL(instance_len=8)

    parser.define_function(
        "ALL_COMP_SR",
        parameters=["candidate", "c", "delta"],
        body=(
            "SIGMA completion. ("
            "COMP(candidate, completion) * "
            "NOT(EP(completion, c) >= delta)"
            ") == 0"
        ),
    )
    parser.define_function(
        "ALL_COMP_CE",
        parameters=["candidate", "c", "delta"],
        body=(
            "SIGMA completion. ("
            "COMP(candidate, completion) * "
            "NOT(CEP(completion, c) >= delta)"
            ") == 0"
        ),
    )
    parser.define_function(
        "THRESHOLD_DETERMINED",
        parameters=["candidate", "c", "delta"],
        body=(
            "NOT("
            "NOT(ALL_COMP_SR(candidate, c, delta)) * "
            "NOT(ALL_COMP_CE(candidate, c, delta))"
            ")"
        ),
    )
    parser.define_function(
        "DFS_SCORE",
        parameters=["candidate", "c", "delta"],
        body=(
            "SIGMA assignment. ("
            "SUF(candidate, assignment) * VAL(assignment) * "
            "THRESHOLD_DETERMINED(assignment, c, delta)"
            ")"
        ),
    )
    parser.define_function(
        "PROB_DFS",
        parameters=["candidate", "c", "delta", "rho"],
        body="DFS_SCORE(candidate, c, delta) >= rho",
    )
    parser.define_function(
        "MINIMUM_PROB_DFS",
        parameters=["candidate", "c", "delta", "rho"],
        body=(
            "PROB_DFS(candidate, c, delta, rho) * "
            "(SIGMA other. ("
            "PROB_DFS(other, c, delta, rho) * "
            "NOT(SIZE(candidate) <= SIZE(other))"
            ") == 0)"
        ),
    )
    parser.define_function(
        "MDFS",
        parameters=["candidate", "c", "delta", "rho"],
        body=(
            "MINIMUM_PROB_DFS(candidate, c, delta, rho) * "
            "(SIGMA other. ("
            "MINIMUM_PROB_DFS(other, c, delta, rho) * "
            "NOT(DFS_SCORE(other, c, delta) <= "
            "DFS_SCORE(candidate, c, delta))"
            ") == 0)"
        ),
    )

    parser.parse("DFS_SCORE(x, c, delta)")
    parser.parse("PROB_DFS(x, c, delta, rho)")
    parser.parse("MINIMUM_PROB_DFS(x, c, delta, rho)")
    parser.parse("MDFS(x, c, delta, rho)")
