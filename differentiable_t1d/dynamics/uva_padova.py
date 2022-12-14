import jax
import jax.numpy as jnp
import numpy as np
import torch
import pkg_resources
import pandas as pd

from typing import NamedTuple, Union

VPATIENT_PARAMS_PATH = pkg_resources.resource_filename(
    "differentiable_t1d", "files/vpatient_params.csv"
)
VPATIENT_CONTROL_PARMS_PATH = pkg_resources.resource_filename(
    "differentiable_t1d", "files/vpatient_control_params.csv"
)


class UvaPadovaParams(NamedTuple):
    b: float  # s; the second flex point of gastric emptying rate
    d: float  # s; the first flex point of gastric emptying rate (called c in the original paper on oral glucose)
    kmin: float  # s
    kmax: float  # s
    kabs: float  # s
    kp1: float
    kp2: float  # s
    kp3: float
    Fsnc: float
    ke1: float
    ke2: float
    f: float
    Vm0: float
    Vmx: float
    Km0: float
    k1: float  # s
    k2: float  # s
    m1: float  # s
    m2: float  # s
    m30: float  # s
    m4: float  # s
    ka1: float
    ka2: float
    Vi: float  # s
    p2u: float
    Ib: float  # s
    ki: float  # s
    kd: float
    ksc: float
    BW: float  # s
    Vg: float  # s


class UvaPadova:
    @staticmethod
    def dynamics(
        params: UvaPadovaParams,
        t,
        state: jnp.ndarray,
        carbs: jnp.ndarray,
        insulin: jnp.ndarray,
    ):
        """
        Time independent dynamics of the UVA Padova 2008 T1D model.
            See Kovatchev et al., 2009: In silico preclinical trials: a proof of concept in closed-loop control of type 1 diabetes.

        References for the equations in this function:
            Dalla Man et al., 2006: A System Model of Oral Glucose Absorption: Validation on Gold Standard Data
            Dalla Man et al., 2007a: GIM, Simulation Software of Meal Glucose–Insulin Model
            Dalla Man et al., 2007b: Meal simulation model of the glucose-insulin system
            Dalla Man et al., 2014: The UVA/PADOVA Type 1 Diabetes Simulator: New Features

        Args:
            params: Params of patient
            t: time; not used
            state: state vector
            carbs: carbs amount of glucose ingested per minute (mg/min) at time t
            insulin: insulin concentration rate in pmol per kg of body weight per min (pmol/kg/min) at time t
        Returns:
            dstate_dt: state time derivative
        """
        state = jnp.clip(state, a_min=0.0)  # clamp to avoid negative values
        (
            q_sto1,
            q_sto2,
            q_gut,
            G_p,
            G_t,
            I_p,
            X,
            I_1,
            I_d,
            I_l,
            I_sc1,
            I_sc2,
            Gs,
        ) = jnp.split(state, 13)

        #### Oral glucose subsystem
        # eps = 1e-8
        # Qsto = q_sto1 + q_sto2
        ## Equation 10 of Dalla Man et al., 2006
        # alpha = 5 / 2 / (1 - params.b) / (carbs + eps)
        ## Equation 11 of Dalla Man et al., 2006
        # beta = 5 / 2 / params.d / (carbs + eps)

        ## Equation 12 of Dalla Man et al., 2006
        # kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
        #    torch.tanh(alpha * (Qsto - params.b * carbs)) - torch.tanh(beta * (Qsto - params.d * carbs)) + 2
        # )
        # kgut = torch.where(carbs > eps, kgut, params.kmax)
        # NOTE: Eq 12 leads to zero gradients of kgut with respect to carbs if tanh saturates
        # This happens if carbs is too small, causing alpha and beta to be too large
        # Indeed, often kgut ends up just being constant in the simulation
        # Thus instead, we use a constant value
        kgut = params.kmax

        # Equation 7 of Dalla Man et al., 2006
        # Divide by BW to get mg/kg/min from mg/min
        Ra = params.f * params.kabs * q_gut / params.BW
        # Equation 7 of Dalla Man et al., 2006
        dq_sto1 = -params.kmax * q_sto1 + carbs
        dq_sto2 = params.kmax * q_sto1 - kgut * q_sto2
        dq_gut = kgut * q_sto2 - params.kabs * q_gut

        #### Glucose kinetics
        # Equation 10 in Dalla Man et al., 2007b
        EGP = params.kp1 - params.kp2 * G_p - params.kp3 * I_d
        EGP = jnp.clip(EGP, a_min=0.0)
        # Equation 14 of Dalla Man et al., 2007b
        Uii = params.Fsnc
        # Equation 27 of Dalla Man et al., 2007b
        E = jnp.clip(params.ke1 * (G_p - params.ke2), a_min=0.0)
        # Equation 16 of Dalla Man et al., 2007b
        Vmt = params.Vm0 + params.Vmx * X
        # Equation 17 of Dalla Man et al., 2007b
        Kmt = params.Km0
        # Equation 15 of Dalla Man et al., 2007b
        # TODO: update to new version of Uid that uses risk?
        Uid = (
            Vmt * G_t / (Kmt + G_t)
        )  # nonlinear Michaelis-Menten term for insulin dependent glucose utilization
        # Equation 1 top of Dalla Man et al., 2007b
        dG_p = EGP + Ra - Uii - E - params.k1 * G_p + params.k2 * G_t
        # Equation 1 middle of Dalla Man et al., 2007b
        dG_t = -Uid + params.k1 * G_p - params.k2 * G_t

        #### Insulin kinetics
        # Equation 3 middle of Dalla Man et al., 2007b
        # Extra terms come from subcutaneous insulin (Ra_I): Equation A15 of Dalla Man et al., 2014
        Ra_I = params.m1 * I_l + params.ka1 * I_sc1 + params.ka2 * I_sc2
        dI_p = -(params.m2 + params.m4) * I_p + Ra_I
        # Equation 3 top of Dalla Man et al., 2007b
        dI_l = -(params.m1 + params.m30) * I_l + params.m2 * I_p
        # Equation 3 bottom of Dalla Man et al., 2007b
        I = I_p / params.Vi
        # Equation 18 of Dalla Man et al., 2007b
        dX = -params.p2u * X + params.p2u * (
            I - params.Ib
        )  # torch.clamp(I - params.Ib, min=0.)
        # Equation 11 of Dalla Man et al., 2007b
        dI_1 = -params.ki * (I_1 - I)
        dI_d = -params.ki * (I_d - I_1)

        #### subcutaneous insulin kinetics
        # Equation 1 of Dalla Man et al., 2007a
        dI_sc1 = insulin - (params.ka1 + params.kd) * I_sc1

        dI_sc2 = params.kd * I_sc1 - params.ka2 * I_sc2

        #### subcutaneous glucose
        # Equation 7 of Dalla Man et al., 2007a
        # There is a typo in Equation 7, it should be G_p(t) instead of G(t) see https://github.com/jxx123/simglucose/issues/20
        dGs = -params.ksc * Gs + params.ksc * G_p

        dstate_dt = jnp.concatenate(
            [
                dq_sto1,
                dq_sto2,
                dq_gut,
                dG_p,
                dG_t,
                dI_p,
                dX,
                dI_1,
                dI_d,
                dI_l,
                dI_sc1,
                dI_sc2,
                dGs,
            ]
        )
        return dstate_dt, dict(
            Ra=Ra, Ra_I=Ra_I, EGP=EGP, E=E, Uid=Uid, Vmt=Vmt, Kmt=Kmt
        )

    @staticmethod
    def observe_blood_glucose(params: UvaPadovaParams, state: jnp.ndarray):
        G = state[..., 3] / params.Vg
        return G

    @staticmethod
    def observe_subcutaneous_glucose(params: UvaPadovaParams, state: jnp.ndarray):
        Gs = state[..., 12] / params.Vg
        return Gs


def load_patient(patient_id: Union[str, int]):
    """
    Construct patient by patient_id
    patient_id can be a string or an integer from 1 to 30.
    If patient_id is an int, it will be converted into:
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
    """
    if isinstance(patient_id, int):
        patient_id = patient_id - 1
        kind, id = patient_id // 10, patient_id % 10 + 1
        if kind == 0:
            kind = "adolescent"
        elif kind == 1:
            kind = "adult"
        elif kind == 2:
            kind = "child"
        else:
            raise ValueError("patient_id should be between 1 and 30")
        patient_id = f"{kind}#{id:03d}"
    print("Loading patient", patient_id)
    df = pd.read_csv(VPATIENT_PARAMS_PATH, index_col=0)
    df2 = pd.read_csv(VPATIENT_CONTROL_PARMS_PATH, index_col=0)
    df = pd.concat([df, df2], axis=1)
    patient_series = df.loc[patient_id]
    return patient_series


def initialize_patient(patient_id: Union[str, int], to_tensors=False):
    """Load patient parameters and initial state"""

    all_params = load_patient(patient_id)
    params = UvaPadovaParams(
        **{
            k: np.array(float(v))
            for k, v in all_params.items()
            if k in UvaPadovaParams._fields
        }
    )
    init_state = np.array([all_params[f"x0_{i}"] for i in range(1, 14)])
    unused_params = {
        k: v
        for k, v in all_params.items()
        if (k not in UvaPadovaParams._fields) and (k[:2] != "x0")
    }

    if to_tensors:
        params = UvaPadovaParams(
            *[torch.as_tensor(p, dtype=torch.float) for p in params]
        )
        unused_params = {
            k: torch.as_tensor(v, dtype=torch.float) for k, v in unused_params.items()
        }
        init_state = torch.as_tensor(init_state, dtype=torch.float)

    return params, unused_params, init_state


if __name__ == "__main__":
    params, unused_params, init_state = initialize_patient(1, to_tensors=False)
    state = init_state.copy()
    t = 0
    carbs, insulin = jnp.array(1000.0), jnp.array(0.0)
    dstate_dt = UvaPadova.dynamics(params, t, state, carbs, insulin)

    print(
        "Blood glucose", UvaPadova.observe_blood_glucose(params, state).item(), "mg/dL"
    )
    print(
        "Subcutaneous glucose",
        UvaPadova.observe_subcutaneous_glucose(params, state).item(),
        "mg/dL",
    )
    # check differentiability
    params_jac = jax.jacfwd(UvaPadova.dynamics)(params, t, state, carbs, insulin)
    print("Params jacobian", params_jac)
    state_jac = jax.jacfwd(UvaPadova.dynamics, argnums=2)(
        params, t, state, carbs, insulin
    )
    print("State jacobian", state_jac)
    carbs_jac = jax.jacfwd(UvaPadova.dynamics, argnums=3)(
        params, t, state, carbs, insulin
    )
    print("Carbs jacobian", carbs_jac)
    insulin_jac = jax.jacfwd(UvaPadova.dynamics, argnums=4)(
        params, t, state, carbs, insulin
    )
    print("Insulin jacobian", insulin_jac)
