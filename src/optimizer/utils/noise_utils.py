from typing import Any, Dict, Optional, Sequence

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
)


def build_noise_model(config: Optional[Dict[str, Any]]) -> Optional[NoiseModel]:
    if not config:
        return None

    noise_type = config.get("type", "depolarizing")
    gates_1q: Sequence[str] = config.get("gates_1q", ("rx", "rz", "sx", "x", "h"))
    gates_2q: Sequence[str] = config.get("gates_2q", ("cx",))

    noise_model = NoiseModel()

    if noise_type == "depolarizing":
        p1 = float(config.get("p1", config.get("p", 0.001)))
        p2 = float(config.get("p2", p1))
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), gates_1q)
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), gates_2q)
    elif noise_type == "amplitude_damping":
        gamma = float(config.get("gamma", 0.001))
        noise_model.add_all_qubit_quantum_error(amplitude_damping_error(gamma), gates_1q)
    elif noise_type == "phase_damping":
        gamma = float(config.get("gamma", 0.001))
        noise_model.add_all_qubit_quantum_error(phase_damping_error(gamma), gates_1q)
    elif noise_type == "thermal_relaxation":
        t1 = float(config["t1"])
        t2 = float(config["t2"])
        time_1q = float(config.get("time_1q", 50.0))
        time_2q = float(config.get("time_2q", 150.0))
        noise_model.add_all_qubit_quantum_error(
            thermal_relaxation_error(t1, t2, time_1q), gates_1q
        )
        noise_model.add_all_qubit_quantum_error(
            thermal_relaxation_error(t1, t2, time_2q), gates_2q
        )
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    readout_prob = config.get("readout")
    if readout_prob is not None:
        prob = float(readout_prob)
        noise_model.add_all_qubit_readout_error(
            ReadoutError([[1 - prob, prob], [prob, 1 - prob]])
        )

    return noise_model


def build_aer_simulator(config: Optional[Dict[str, Any]]) -> AerSimulator:
    noise_model = build_noise_model(config)
    if noise_model is None:
        return AerSimulator()
    return AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
