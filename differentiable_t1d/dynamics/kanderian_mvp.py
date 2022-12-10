from typing import NamedTuple
import jax
import jax.numpy as jnp
import torch
import numpy as np
import pkg_resources
import pandas as pd


DYNAMICS_PARAMS_PATH = pkg_resources.resource_filename(
    "differentiable_t1d", "files/kanderian_dynamics_params.csv"
)

MEAL_PARAMS_PATH = pkg_resources.resource_filename(
    "differentiable_t1d", "files/kanderian_meal_params.csv"
)


class KanderianMvpParams(NamedTuple):
    BW: float  # kg
    total_daily_insulin: float  # U/day

    ci: float  # mL/min
    tau1: float  # min
    tau2: float  # min
    tausen: float  # min
    vg: float  # dL
    p2: float  # 1/min
    egp: float  # mg/dL/min
    gezi: float  # 1 / min
    si: float  # mL/muU


class KanderianMvp:
    @staticmethod
    def dynamics(
        params: KanderianMvpParams,
        t,
        state: jnp.ndarray,
        carbs: jnp.ndarray,
        insulin: jnp.ndarray,
    ):
        """
        Modified version of the Metronics Virtual Patient model from Kanderian 2009

        We use the 2 compartment insulin model from Kanderian et al. 2009, since they only report inferred parameters for this setting.

        Reference:
            Kanderian et al., 2009: Identification of Intraday Metabolic Profiles during Closed-Loop Glucose Control in Individuals with Type 1 Diabetes

        Units (same as Bergman's minimal model):
            G: mg/dL
            Ieff: 1/min
            Isc: muU/mL
            Ip: muU/mL
            Gisf: mg/dL

        We assume that the control inputs already have appropriate units.
            `carbs` is R_A(t) in the paper and has units of mg/dL/min. C_H has units of mg (representing the amount of carbs eaten)
            `insulin` is ID(t)/tau1/ci in the paper and has units of muU/mL/min.

        Args:
            params: Params of patient
            t: time; not used
            state: state vector
            carbs: rate of carbs glucose ingested per "body volume" (mg/dL/min) at time t
            insulin: insulin concentration rate (muU/min) at time t
        Returns:
            dstate_dt: state time derivative
        """
        state = jnp.clip(state, a_min=0.0)  # clamp to avoid negative values
        G, Ieff, Isc, Ip, Gisf = jnp.split(state, 5)

        # Glucose dynamics
        dG = -(params.gezi + Ieff) * G + params.egp + carbs  # Eq 4
        dIeff = -params.p2 * Ieff + params.p2 * params.si * Ip  # Eq 3

        # Insulin dynamics (2 compartments)
        dIsc = -Isc / params.tau1 + insulin  # Eq 1
        dIp = -Ip / params.tau2 + Isc / params.tau2  # Eq 2

        # Interstitial glucose dynamics
        dGisf = -Gisf / params.tausen + G / params.tausen  # Eq 10

        dstate = jnp.concatenate([dG, dIeff, dIsc, dIp, dGisf])
        return dstate, None

    @staticmethod
    def observe_blood_glucose(params: KanderianMvpParams, state: jnp.ndarray):
        return state[..., 0]

    @staticmethod
    def observe_subcutaneous_glucose(params: KanderianMvpParams, state: jnp.ndarray):
        return state[..., 4]


def load_patient(patient_id: int):
    df = pd.read_csv(DYNAMICS_PARAMS_PATH, index_col=0)
    patient_series = df.loc[patient_id]
    return patient_series


def load_meals():
    df = pd.read_csv(MEAL_PARAMS_PATH)
    return df


def initialize_patient(patient_id: int, to_tensors=False):
    """Load patient parameters and initial state"""

    all_params = load_patient(patient_id)
    params = KanderianMvpParams(
        **{
            k: np.array(float(v))
            for k, v in all_params.items()
            if k in KanderianMvpParams._fields
        }
    )
    init_state = np.array([140.0, 1e-2, 10.0, 10.0, 140.0])  # heuristic initial state

    if to_tensors:
        params = KanderianMvpParams(
            *[torch.as_tensor(p, dtype=torch.float) for p in params]
        )
        init_state = torch.as_tensor(init_state, dtype=torch.float)

    return params, init_state


if __name__ == "__main__":
    params, init_state = initialize_patient(1)
