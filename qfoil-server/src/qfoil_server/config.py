"""Central defaults and limits for server evaluation and witness search.

These constants are read at startup/import time. Search defaults can be
overridden by HTTP request options within the maximum limits below.
"""

# HTTP evaluation defaults to heuristic SIGMA; requests may select exact mode.
DEFAULT_SIGMA_MODE = "heuristic"

# Total evaluator node visits allowed per operation, shared by all subqueries.
DEFAULT_EVALUATION_BUDGET = 20_000_000
MAX_EVALUATION_BUDGET = 20_000_000
MAX_SIGMA_SEARCH_BUDGET = 20_000_000

# Local candidate-evaluation limit for SIGMA searches during evaluation.
DEFAULT_SIGMA_SEARCH_BUDGET = 64

# Defaults and HTTP validation limits for explicit witness search.
DEFAULT_SEARCH_BUDGET = 1_000
MAX_SEARCH_BUDGET = 100_000
DEFAULT_INNER_BUDGET = 8
MAX_INNER_BUDGET = 1_000
DEFAULT_POOL_SIZE = 30
MAX_POOL_SIZE = 1_000
