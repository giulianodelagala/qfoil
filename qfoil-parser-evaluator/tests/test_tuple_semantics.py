"""Tests for backend-independent QFOIL tuple semantics."""

import pytest

from qfoil.language import QFOILParser
from qfoil.semantics import STANDARD_DEFINITIONS, TupleSemantics


@pytest.mark.parametrize("instance_len", [-1, 1.5, True])
def test_tuple_semantics_rejects_invalid_instance_length(instance_len):
    with pytest.raises(ValueError, match="non-negative integer"):
        TupleSemantics(instance_len)


def test_handlers_cover_the_native_tuple_vocabulary():
    semantics = TupleSemantics()

    assert set(semantics.handlers) == {
        "BOTTOM",
        "FULL",
        "L0",
        "L1",
        "FEAT",
        "SIZE",
        "FREE",
        "MEET",
        "JOIN",
        "PARTITION",
        "OPPOSITE",
        "SUF",
        "COMP",
    }


def test_lattice_order_and_completion_are_backend_independent():
    semantics = TupleSemantics()
    partial = (1, None, 0)
    complete = (1, 1, 0)

    assert semantics.subseteq(partial, complete) == 1
    assert semantics.subset(partial, complete) == 1
    assert semantics.subset(complete, complete) == 0
    assert semantics.comp([partial, complete]) == 1
    assert semantics.comp([partial, (0, 1, 0)]) == 0


def test_meet_and_join_follow_partial_tuple_lattice():
    semantics = TupleSemantics()

    assert semantics.meet([(1, None), (1, 0), (1, None)]) == 1
    assert semantics.join([(1, None), (None, 0), (1, 0)]) == 1
    assert semantics.join([(1, None), (0, None), (1, None)]) == 0


def test_tuple_validation_errors_remain_explicit():
    semantics = TupleSemantics()

    with pytest.raises(ValueError, match="equal-length"):
        semantics.subseteq((1,), (1, 0))


def test_every_standard_definition_is_parseable():
    parser = QFOILParser()

    assert len({definition.name for definition in STANDARD_DEFINITIONS}) == len(
        STANDARD_DEFINITIONS
    )
    for definition in STANDARD_DEFINITIONS:
        parser.parse(definition.body)
