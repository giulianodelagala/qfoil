# Article MUX validation report

The run on 23 October 2026 completed with **32 passed, 0 failed, 0 errors,
and 0 skipped**, in approximately 16.99 seconds. The 16 configurations each
have one independent reference-set check and one PyJuice evaluator check.
Each evaluator check exercises both `restricted` and `exact` enumeration,
comparing counts, existential decisions, and witness validity.

- [pytest.log](pytest.log): complete verbose output with test identifiers.
- [junit.xml](junit.xml): machine-readable status and duration for each test.
- [gpu.txt](gpu.txt): NVIDIA device and driver information.
- [commit.txt](commit.txt): repository HEAD when the run started.
- [worktree.txt](worktree.txt): local changes present when the run started.

The recorded HEAD predates this archive commit. The run used the untracked
`tests/mux/test_article_validation.py` now committed alongside this report.
The original execution metadata is preserved rather than rewritten to refer
to a later commit. The pending README and licensing changes are also included;
no runtime implementation changes are introduced by this archive commit.

Run the same tests from the repository root in a CUDA-capable environment:

```bash
python -m pytest tests/mux/test_article_validation.py -v -ra --color=no --junitxml=junit.xml
```

The report records test outcomes, not a table of numeric witness counts or
maximum probability errors. Passing integration tests confirms the asserted
agreement for these cases; it is not a general arithmetic certification.
