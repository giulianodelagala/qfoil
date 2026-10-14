"""Validate the minimal DFS formula structure for the noisy MUX.

The exhaustive oracle proves the expected result in ``test_global_oracle``.
Directly evaluating the rank-3 formula would nest exact searches over the
6561-element partial domain, so the normal suite validates construction and
parsing without pretending that this brute-force execution is inexpensive.
"""

from qfoil import QFOIL


def test_minimal_dfs_runtime_definition_parses():
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
        "DFS",
        parameters=["candidate", "c", "delta"],
        body=(
            "SIGMA assignment. ("
            "SUF(candidate, assignment) * "
            "NOT(THRESHOLD_DETERMINED(assignment, c, delta))"
            ") == 0"
        ),
    )
    parser.define_function(
        "MINIMAL_DFS",
        parameters=["candidate", "c", "delta"],
        body=(
            "DFS(candidate, c, delta) * "
            "(SIGMA smaller. ("
            "(smaller << candidate) * DFS(smaller, c, delta)"
            ") == 0)"
        ),
    )

    parser.parse("MINIMAL_DFS(x, c, delta)")
