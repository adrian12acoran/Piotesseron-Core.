import pytest

from piotesseron_core import (
    BinaryPolicy,
    InternalState,
    MasterClosure,
    PiSubstrate,
    Piotesseron,
    PiotesseronConfig,
    structural_self_check,
)


def base_metrics(**overrides):
    metrics = {
        "support": 0.50,
        "coherence": 0.50,
        "reliability": 0.50,
        "completeness": 0.50,
        "contradiction": 0.10,
        "risk": 0.10,
        "novelty": 0.50,
        "actionability": 0.50,
        "irreversibility": 0.10,
        "noise": 0.20,
        "limit_pressure": 0.10,
        "uncertainty": 0.20,
        "balance": 0.80,
    }
    metrics.update(overrides)
    return metrics


def test_structural_self_check():
    checks = structural_self_check()

    assert checks
    assert all(checks.values()), checks


def test_a3_is_inviolable():
    substrate = PiSubstrate(anchor=3)

    assert substrate.A3 == 3

    with pytest.raises(ValueError):
        PiSubstrate(anchor=4)


def test_four_internal_states_exist():
    assert InternalState.ONE.value == "1"
    assert InternalState.ZERO.value == "0"
    assert InternalState.MAYBE.value == "MAYBE"
    assert InternalState.SILENCE.value == "SILENCE"

    assert InternalState.MAYBE.conceptual_label == "Tal_Vez"
    assert InternalState.SILENCE.conceptual_label == "Silencio"


def test_state_machine_can_represent_all_four_states():
    core = Piotesseron()

    one_metrics = base_metrics(
        support=0.85,
        coherence=0.90,
        reliability=0.90,
        actionability=0.90,
    )

    zero_metrics = base_metrics(
        support=0.15,
        coherence=0.20,
        reliability=0.20,
        novelty=0.20,
        actionability=0.10,
    )

    maybe_metrics = base_metrics(
        support=0.50,
        coherence=0.50,
        reliability=0.50,
        novelty=0.80,
        actionability=0.50,
    )

    silence_metrics = base_metrics(
        support=0.80,
        coherence=0.80,
        reliability=0.80,
        risk=0.90,
        actionability=0.80,
    )

    assert (
        core.alcyone._determine_state(one_metrics)
        is InternalState.ONE
    )

    assert (
        core.alcyone._determine_state(zero_metrics)
        is InternalState.ZERO
    )

    assert (
        core.alcyone._determine_state(maybe_metrics)
        is InternalState.MAYBE
    )

    assert (
        core.alcyone._determine_state(silence_metrics)
        is InternalState.SILENCE
    )


def test_master_closures_follow_internal_states():
    core = Piotesseron()

    cases = [
        (
            InternalState.ONE,
            base_metrics(
                support=0.85,
                coherence=0.90,
                reliability=0.90,
                actionability=0.90,
            ),
            MasterClosure.ACTIVATE,
        ),
        (
            InternalState.ZERO,
            base_metrics(
                support=0.15,
                actionability=0.10,
            ),
            MasterClosure.DISCARD,
        ),
        (
            InternalState.MAYBE,
            base_metrics(),
            MasterClosure.SUSPEND,
        ),
        (
            InternalState.SILENCE,
            base_metrics(
                risk=0.90,
            ),
            MasterClosure.CONTAIN,
        ),
    ]

    for state, metrics, expected_closure in cases:
        task = core.alcyone._form_task(
            state,
            metrics,
        )

        closure = core.alcyone._master_closure(
            state,
            task,
        )

        assert closure is expected_closure


def test_binary_projection_preserves_four_state_interface():
    config = PiotesseronConfig(
        binary_policy=BinaryPolicy.THRESHOLD,
        maybe_binary_threshold=0.56,
    )

    core = Piotesseron(config)

    metrics = base_metrics(
        novelty=0.50,
        risk=0.10,
        contradiction=0.10,
    )

    assert (
        core.alcyone._project_binary(
            InternalState.ONE,
            0.10,
            metrics,
        )
        == 1
    )

    assert (
        core.alcyone._project_binary(
            InternalState.ZERO,
            0.90,
            metrics,
        )
        == 0
    )

    assert (
        core.alcyone._project_binary(
            InternalState.SILENCE,
            0.90,
            metrics,
        )
        == 0
    )

    assert (
        core.alcyone._project_binary(
            InternalState.MAYBE,
            0.55,
            metrics,
        )
        == 0
    )

    assert (
        core.alcyone._project_binary(
            InternalState.MAYBE,
            0.60,
            metrics,
        )
        == 1
    )


def test_real_evaluate_path_and_trajectory():
    core = Piotesseron()

    sample = {
        "signal": {
            "content": "Coherent controlled evidence.",
            "modality": "textual",
            "weight": 1.0,
            "coherence": 0.90,
            "reliability": 0.90,
            "completeness": 0.90,
            "contradiction": 0.05,
            "risk": 0.05,
            "novelty": 0.50,
            "actionability": 0.80,
            "irreversibility": 0.05,
            "noise": 0.05,
        }
    }

    decision = core.evaluate(
        sample,
        identifier="FUNCTIONAL-EVALUATE",
    )

    trajectory = decision.trajectory

    assert trajectory["uee_id"] == "FUNCTIONAL-EVALUATE"

    assert trajectory["A3_primary_anchor"] == 3

    assert trajectory["major_hypercube"]["name"] == "Alcyone"
    assert trajectory["major_hypercube"]["active"] is True
    assert trajectory["major_hypercube"]["unique"] is True
    assert trajectory["major_hypercube"]["final_authority"] is True

    assert (
        trajectory["sophiana_position"]["belongs_to_Alcyone"]
        is True
    )

    assert (
        trajectory["sophiana_position"]["owns_minor_hypercube"]
        is False
    )

    assert (
        trajectory["internal_state"]
        == decision.internal_state.value
    )

    assert (
        trajectory["master_closure"]
        == decision.master_closure.value
    )

    assert (
        trajectory["binary_projection"]
        == decision.binary_output
    )


def test_reentry_creates_new_external_trajectory():
    core = Piotesseron()

    original = core.evaluate(
        {
            "signal": {
                "content": "Initial evidence.",
                "coherence": 0.55,
                "reliability": 0.55,
                "completeness": 0.50,
                "contradiction": 0.10,
                "risk": 0.10,
                "novelty": 0.70,
                "actionability": 0.50,
                "irreversibility": 0.10,
                "noise": 0.25,
            }
        },
        identifier="REENTRY-ORIGIN",
    )

    history_before = len(core.history)

    new_evidence = {
        "content": "New evidence for structural reevaluation.",
        "modality": "textual",
        "weight": 1.0,
        "coherence": 0.90,
        "reliability": 0.90,
        "completeness": 0.90,
        "contradiction": 0.05,
        "risk": 0.05,
        "novelty": 0.60,
        "actionability": 0.85,
        "irreversibility": 0.05,
        "noise": 0.05,
    }

    reentered = core.reenter(
        original,
        new_evidence,
    )

    assert len(core.history) == history_before + 1

    assert "REENTRY" in reentered.trajectory["uee_id"]

    assert reentered.trajectory["external_input"] is True

    assert reentered.internal_state in {
        InternalState.ONE,
        InternalState.ZERO,
        InternalState.MAYBE,
        InternalState.SILENCE,
    }
