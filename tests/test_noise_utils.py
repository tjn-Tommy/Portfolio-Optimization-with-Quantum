import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from qiskit_aer.noise import NoiseModel

from optimizer.utils.noise_utils import build_aer_simulator, build_noise_model


def _errors_by_type(noise_dict, error_type):
    return [err for err in noise_dict.get("errors", []) if err["type"] == error_type]


def _errors_by_op(noise_dict, error_type="qerror"):
    errors = {}
    for err in _errors_by_type(noise_dict, error_type):
        operations = err.get("operations", [])
        if operations:
            errors[operations[0]] = err
    return errors


def _assert_depolarizing_probabilities(probs, p, num_qubits):
    if num_qubits == 1:
        expected_len = 4
        expected_id = 1 - 3 * p / 4
        expected_other = p / 4
    elif num_qubits == 2:
        expected_len = 16
        expected_id = 1 - 15 * p / 16
        expected_other = p / 16
    else:
        raise ValueError("Only 1q and 2q depolarizing errors are supported in tests.")

    assert len(probs) == expected_len
    assert probs[0] == pytest.approx(expected_id, rel=0, abs=1e-12)
    for prob in probs[1:]:
        assert prob == pytest.approx(expected_other, rel=0, abs=1e-12)


def test_build_noise_model_none():
    assert build_noise_model(None) is None
    assert build_noise_model({}) is None
    sim = build_aer_simulator(None)
    assert sim.options.noise_model is None


def test_build_noise_model_depolarizing_probabilities():
    config = {
        "type": "depolarizing",
        "p1": 0.12,
        "p2": 0.3,
        "gates_1q": ["x", "h"],
        "gates_2q": ["cx"],
    }
    noise_model = build_noise_model(config)
    errors_by_op = _errors_by_op(noise_model.to_dict(), "qerror")

    assert set(errors_by_op) == {"x", "h", "cx"}
    _assert_depolarizing_probabilities(errors_by_op["x"]["probabilities"], 0.12, 1)
    _assert_depolarizing_probabilities(errors_by_op["h"]["probabilities"], 0.12, 1)
    _assert_depolarizing_probabilities(errors_by_op["cx"]["probabilities"], 0.3, 2)


def test_build_noise_model_adds_readout_error():
    config = {"type": "depolarizing", "p": 0.01, "readout": 0.2}
    noise_model = build_noise_model(config)
    ro_errors = _errors_by_type(noise_model.to_dict(), "roerror")

    assert len(ro_errors) == 1
    assert ro_errors[0]["operations"] == ["measure"]
    np.testing.assert_allclose(
        ro_errors[0]["probabilities"],
        [[0.8, 0.2], [0.2, 0.8]],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "noise_type, extra_config",
    [
        ("amplitude_damping", {"gamma": 0.2}),
        ("phase_damping", {"gamma": 0.3}),
    ],
)
def test_build_noise_model_kraus_1q(noise_type, extra_config):
    config = {
        "type": noise_type,
        "gates_1q": ["x", "h"],
        **extra_config,
    }
    noise_model = build_noise_model(config)
    errors_by_op = _errors_by_op(noise_model.to_dict(), "qerror")

    assert set(errors_by_op) == {"x", "h"}
    for err in errors_by_op.values():
        assert err["instructions"][0][0]["name"] == "kraus"


def test_build_noise_model_thermal_relaxation_kraus():
    config = {
        "type": "thermal_relaxation",
        "t1": 50.0,
        "t2": 70.0,
        "time_1q": 20.0,
        "time_2q": 60.0,
        "gates_1q": ["x"],
        "gates_2q": ["cx", "cz"],
    }
    noise_model = build_noise_model(config)
    errors_by_op = _errors_by_op(noise_model.to_dict(), "qerror")

    assert set(errors_by_op) == {"x", "cx", "cz"}
    for err in errors_by_op.values():
        assert err["instructions"][0][0]["name"] == "kraus"


def test_build_noise_model_thermal_requires_t1_t2():
    with pytest.raises(KeyError):
        build_noise_model({"type": "thermal_relaxation", "t1": 50.0})


def test_build_noise_model_invalid_type():
    with pytest.raises(ValueError):
        build_noise_model({"type": "not-a-noise"})


def test_build_aer_simulator_with_noise_model():
    sim = build_aer_simulator({"type": "depolarizing", "p": 0.02})
    assert isinstance(sim.options.noise_model, NoiseModel)
