"""Exact probabilistic models for a noisy binary 4-to-1 multiplexer.

The eight input variables are ordered as
``S0, S1, D0, D1, D2, D3, N0, N1``.  The selected data input is::

    index = 2 * S1 + S0
    Y = D[index]

``N0`` and ``N1`` are independent noise features and do not affect ``Y``.
``MuxDistribution`` describes all eight independent Bernoulli inputs and the
deterministic MUX label. ``build_mux_model`` converts that distribution into
two exact class-conditional PyJuice circuits, one for each value of ``Y``.
No parameter learning is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Iterator, Optional, Sequence

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import torch

from qfoil.backends.pyjuice import PyJuiceModel

FEATURE_NAMES = ("S0", "S1", "D0", "D1", "D2", "D3", "N0", "N1")
NUM_FEATURES = len(FEATURE_NAMES)

# A deliberately asymmetric distribution for examples and differential tests.
BERNOULLI_SELECTOR_PROBS = (0.35, 0.65)
BERNOULLI_DATA_PROBS = (0.10, 0.30, 0.70, 0.90)
BERNOULLI_NOISE_PROBS = (0.20, 0.80)

Assignment = tuple[int, ...]
PartialAssignment = tuple[Optional[int], ...]


def _validate_binary_values(values: Sequence[int], name: str) -> None:
    if len(values) != NUM_FEATURES:
        raise ValueError(f"{name} must contain {NUM_FEATURES} values")
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(f"{name} values must be integers in {{0, 1}}")


def _validate_partial_values(values: Sequence[Optional[int]], name: str) -> None:
    if len(values) != NUM_FEATURES:
        raise ValueError(f"{name} must contain {NUM_FEATURES} values")
    if any(
        value is not None and (type(value) is not int or value not in (0, 1))
        for value in values
    ):
        raise ValueError(f"{name} values must be 0, 1, or None")


def mux_output(assignment: Sequence[int]) -> int:
    """Return the deterministic MUX output for one complete assignment."""
    _validate_binary_values(assignment, "assignment")
    selector = 2 * assignment[1] + assignment[0]
    return assignment[2 + selector]


def enumerate_assignments() -> Iterator[Assignment]:
    """Iterate over all 256 complete noisy-MUX inputs."""
    yield from product((0, 1), repeat=NUM_FEATURES)


def enumerate_partial_assignments() -> Iterator[PartialAssignment]:
    """Iterate over all 6561 elements of ``{0, 1, None}^8``."""
    yield from product((0, 1, None), repeat=NUM_FEATURES)


@dataclass(frozen=True)
class MuxDistribution:
    """Independent Bernoulli inputs with a deterministic MUX label.

    Probabilities are expressed as ``P(variable=1)``.  Values strictly
    between zero and one ensure that both class-conditional circuits are
    well-defined and that every complete input has positive probability.
    """

    selector_probs: tuple[float, float]
    data_probs: tuple[float, float, float, float]
    noise_probs: tuple[float, float]

    def __post_init__(self) -> None:
        if len(self.selector_probs) != 2:
            raise ValueError("selector_probs must contain probabilities for S0 and S1")
        if len(self.data_probs) != 4:
            raise ValueError("data_probs must contain probabilities for D0..D3")
        if len(self.noise_probs) != 2:
            raise ValueError("noise_probs must contain probabilities for N0 and N1")
        for probability in (
            *self.selector_probs,
            *self.data_probs,
            *self.noise_probs,
        ):
            if not 0.0 < probability < 1.0:
                raise ValueError(
                    "all Bernoulli probabilities must lie strictly in (0, 1)"
                )

    @classmethod
    def uniform(cls) -> "MuxDistribution":
        """Return the distribution with eight independent fair bits."""
        return cls(
            selector_probs=(0.5, 0.5),
            data_probs=(0.5,) * 4,
            noise_probs=(0.5, 0.5),
        )

    @classmethod
    def bernoulli(cls) -> "MuxDistribution":
        """Return the standard asymmetric Bernoulli example distribution."""
        return cls(
            selector_probs=BERNOULLI_SELECTOR_PROBS,
            data_probs=BERNOULLI_DATA_PROBS,
            noise_probs=BERNOULLI_NOISE_PROBS,
        )

    @property
    def feature_probs(self) -> tuple[float, ...]:
        return (*self.selector_probs, *self.data_probs, *self.noise_probs)

    def assignment_probability(self, assignment: Sequence[int]) -> float:
        """Return ``P(X=assignment)`` for a complete input."""
        _validate_binary_values(assignment, "assignment")
        return prod(
            probability if value else 1.0 - probability
            for value, probability in zip(assignment, self.feature_probs)
        )

    def joint_probability(self, assignment: Sequence[int], class_id: int) -> float:
        """Return ``P(X=assignment, Y=class_id)``."""
        self._validate_class_id(class_id)
        if mux_output(assignment) != class_id:
            return 0.0
        return self.assignment_probability(assignment)

    def class_probability(self, class_id: int) -> float:
        """Return the exact prior ``P(Y=class_id)``."""
        self._validate_class_id(class_id)
        probability_one = 0.0
        for s0, s1 in product((0, 1), repeat=2):
            selector_probability = self._selector_probability(s0, s1)
            selected = 2 * s1 + s0
            probability_one += selector_probability * self.data_probs[selected]
        return probability_one if class_id == 1 else 1.0 - probability_one

    @property
    def priors(self) -> dict[int, float]:
        return {class_id: self.class_probability(class_id) for class_id in (0, 1)}

    def evidence_probability(
        self,
        evidence: Sequence[Optional[int]],
        class_id: Optional[int] = None,
    ) -> float:
        """Return ``P(evidence)`` or ``P(evidence, Y=class_id)``.

        ``None`` marks an unobserved input.  Exhaustive enumeration is
        intentional here: with only 256 complete inputs this is a compact,
        transparent oracle for notebooks and tests.
        """
        _validate_partial_values(evidence, "evidence")
        if class_id is not None:
            self._validate_class_id(class_id)

        total = 0.0
        for assignment in enumerate_assignments():
            if not self.matches(evidence, assignment):
                continue
            if class_id is None:
                total += self.assignment_probability(assignment)
            else:
                total += self.joint_probability(assignment, class_id)
        return total

    def posterior_class(
        self, evidence: Sequence[Optional[int]], class_id: int
    ) -> float:
        """Return ``P(Y=class_id | evidence)``."""
        numerator = self.evidence_probability(evidence, class_id=class_id)
        denominator = self.evidence_probability(evidence)
        return numerator / denominator

    def conditional_probability(
        self,
        event: Sequence[Optional[int]],
        given: Sequence[Optional[int]],
        class_id: Optional[int] = None,
    ) -> float:
        """Return ``P(event | given)`` or ``P(event | given, Y=class_id)``."""
        _validate_partial_values(event, "event")
        _validate_partial_values(given, "given")
        merged: list[Optional[int]] = []
        for event_value, given_value in zip(event, given):
            if (
                event_value is not None
                and given_value is not None
                and event_value != given_value
            ):
                return 0.0
            merged.append(given_value if given_value is not None else event_value)

        if class_id is None:
            numerator = self.evidence_probability(tuple(merged))
            denominator = self.evidence_probability(given)
        else:
            self._validate_class_id(class_id)
            numerator = self.evidence_probability(tuple(merged), class_id)
            denominator = self.evidence_probability(given, class_id)
        return numerator / denominator

    @staticmethod
    def matches(
        evidence: Sequence[Optional[int]], assignment: Sequence[int]
    ) -> bool:
        _validate_partial_values(evidence, "evidence")
        _validate_binary_values(assignment, "assignment")
        return all(
            expected is None or expected == actual
            for expected, actual in zip(evidence, assignment)
        )

    def sample(
        self, num_samples: int, *, seed: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw complete inputs and their deterministic MUX labels on CPU."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        probabilities = torch.tensor(self.feature_probs, dtype=torch.float32)
        random_values = torch.rand((num_samples, NUM_FEATURES), generator=generator)
        data = (random_values < probabilities).to(dtype=torch.long)
        selector = 2 * data[:, 1] + data[:, 0]
        labels = data[torch.arange(num_samples), 2 + selector]
        return data, labels

    def _selector_probability(self, s0: int, s1: int) -> float:
        return prod(
            probability if value else 1.0 - probability
            for value, probability in zip((s0, s1), self.selector_probs)
        )

    @staticmethod
    def _validate_class_id(class_id: int) -> None:
        if type(class_id) is not int or class_id not in (0, 1):
            raise ValueError("class_id must be 0 or 1")


def _categorical_leaf(variable: int, probability_one: float):
    parameters = torch.tensor(
        [1.0 - probability_one, probability_one], dtype=torch.float32
    )
    return juice.inputs(
        variable,
        num_nodes=1,
        dist=dists.Categorical(num_cats=2),
        params=parameters,
    )


def build_mux_pc_node(distribution: MuxDistribution, class_id: int):
    """Build the exact, uncompiled node graph for ``P(X | Y=class_id)``."""
    distribution._validate_class_id(class_id)
    class_prior = distribution.class_probability(class_id)
    branches = []
    branch_weights = []

    for s0, s1 in product((0, 1), repeat=2):
        selected = 2 * s1 + s0
        selected_probability = distribution.data_probs[selected]
        probability_of_class = (
            selected_probability if class_id == 1 else 1.0 - selected_probability
        )
        branch_weights.append(
            distribution._selector_probability(s0, s1)
            * probability_of_class
            / class_prior
        )

        leaves = [
            _categorical_leaf(0, float(s0)),
            _categorical_leaf(1, float(s1)),
        ]
        for data_index, probability_one in enumerate(distribution.data_probs):
            if data_index == selected:
                probability_one = float(class_id)
            leaves.append(_categorical_leaf(2 + data_index, probability_one))
        for noise_index, probability_one in enumerate(distribution.noise_probs):
            leaves.append(_categorical_leaf(6 + noise_index, probability_one))
        branches.append(juice.multiply(*leaves))

    return juice.summate(
        *branches,
        num_nodes=1,
        params=torch.tensor(branch_weights, dtype=torch.float32),
    )


def build_mux_model(
    distribution: MuxDistribution,
    device: Optional[torch.device | str] = None,
) -> PyJuiceModel:
    """Compile an exact two-class MUX model for the PyJuice backend.

    PyJuice 2.3.x requires CUDA even while compiling its tensor circuit.  A
    clear error is preferable to the lower-level ``torch.cuda`` failure users
    would otherwise receive.
    """
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected_device = torch.device(device if device is not None else default_device)
    if selected_device.type == "cuda" and selected_device.index is None:
        selected_device = torch.device("cuda:0")
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("building and executing the MUX PyJuice model requires CUDA")

    pc_per_class = {}
    for class_id in (0, 1):
        root = build_mux_pc_node(distribution, class_id)
        pc_per_class[class_id] = juice.compile(
            root, device=selected_device, verbose=False
        )

    metadata = {
        "num_features": NUM_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "model_naming": "noisy_mux_4_to_1",
        "selector_convention": "index = 2 * S1 + S0",
        "selector_probs": list(distribution.selector_probs),
        "data_probs": list(distribution.data_probs),
        "noise_probs": list(distribution.noise_probs),
        "noise_features": ["N0", "N1"],
        "exact": True,
    }
    return PyJuiceModel(
        pc_per_class=pc_per_class,
        priors=distribution.priors,
        metadata=metadata,
        device=selected_device,
    )
