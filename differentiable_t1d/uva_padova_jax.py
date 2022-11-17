from differentiable_t1d.utils import Params

import jax
import jax.numpy as jnp
import numpy as np


def uva_padova_2008_dynamics(params: Params, t, state: np.ndarray, carbs: np.ndarray, insulin: np.ndarray):
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
    state = jnp.clip(state, a_min=0.)  # clamp to avoid negative values
    q_sto1, q_sto2, q_gut, G_p, G_t, I_p, X, I_1, I_d, I_l, I_sc1, I_sc2, Gs = jnp.split(state, 13)

    #### Oral glucose subsystem
    #eps = 1e-8
    #Qsto = q_sto1 + q_sto2
    ## Equation 10 of Dalla Man et al., 2006
    #alpha = 5 / 2 / (1 - params.b) / (carbs + eps)
    ## Equation 11 of Dalla Man et al., 2006
    #beta = 5 / 2 / params.d / (carbs + eps)

    ## Equation 12 of Dalla Man et al., 2006
    #kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
    #    torch.tanh(alpha * (Qsto - params.b * carbs)) - torch.tanh(beta * (Qsto - params.d * carbs)) + 2
    #)
    #kgut = torch.where(carbs > eps, kgut, params.kmax)
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
    dq_sto2  = params.kmax * q_sto1 - kgut * q_sto2
    dq_gut = kgut * q_sto2 - params.kabs * q_gut

    #### Glucose kinetics
    # Equation 10 in Dalla Man et al., 2007b
    EGP = params.kp1 - params.kp2 * G_p - params.kp3 * I_d
    EGP = jnp.clip(EGP, a_min=0.)
    # Equation 14 of Dalla Man et al., 2007b
    Uii = params.Fsnc
    # Equation 27 of Dalla Man et al., 2007b
    E = jnp.clip(params.ke1 * (G_p - params.ke2), a_min=0.)
    # Equation 16 of Dalla Man et al., 2007b
    Vmt = params.Vm0 + params.Vmx * X
    # Equation 17 of Dalla Man et al., 2007b
    Kmt = params.Km0
    # Equation 15 of Dalla Man et al., 2007b
    # TODO: update to new version of Uid that uses risk?
    Uid = Vmt * G_t / (Kmt + G_t) # nonlinear Michaelis-Menten term for insulin dependent glucose utilization
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
    dI_l= -(params.m1 + params.m30) * I_l + params.m2 * I_p
    # Equation 3 bottom of Dalla Man et al., 2007b
    I = I_p / params.Vi
    # Equation 18 of Dalla Man et al., 2007b
    dX = -params.p2u * X + params.p2u * (I - params.Ib) #torch.clamp(I - params.Ib, min=0.)
    # Equation 11 of Dalla Man et al., 2007b
    dI_1= -params.ki * (I_1 - I)
    dI_d= -params.ki * (I_d - I_1)

    #### subcutaneous insulin kinetics
    # Equation 1 of Dalla Man et al., 2007a
    dI_sc1= insulin - (params.ka1 + params.kd) * I_sc1

    dI_sc2= params.kd * I_sc1 - params.ka2 * I_sc2

    #### subcutaneous glucose
    # Equation 7 of Dalla Man et al., 2007a 
    # There is a typo in Equation 7, it should be G_p(t) instead of G(t) see https://github.com/jxx123/simglucose/issues/20
    dGs= -params.ksc * Gs + params.ksc * G_p

    dstate_dt = jnp.concatenate([dq_sto1, dq_sto2, dq_gut, dG_p, dG_t, dI_p, dX, dI_1, dI_d, dI_l, dI_sc1, dI_sc2, dGs])
    return dstate_dt, dict(Ra=Ra, Ra_I=Ra_I, EGP=EGP, E=E, Uid=Uid, Vmt=Vmt, Kmt=Kmt)

def observe_blood_glucose(params:Params, state):
    G = state[..., 3] / params.Vg
    return G

def observe_subcutaneous_glucose(params:Params, state):
    Gs = state[..., 12] / params.Vg
    return Gs


def square_controls(widths, amplitudes, timestamps, t):
    # t is a scalar
    assert widths.ndim == 1
    assert timestamps.ndim == 1
    assert amplitudes.ndim == 1
    assert t.ndim == 0

    t_since = t - timestamps # (T_e,)
    relevant = t_since >= 0 # (T_e,)
    t_since = jnp.where(relevant, t_since, jnp.zeros_like(t_since)) # (T_e,)
    u = amplitudes * relevant * (t_since < widths) # (T_e,)
    u = u.sum(0)
    return u


if __name__ == "__main__":
    from differentiable_t1d.utils import initialize_patient
    params, unused_params, init_state = initialize_patient(1, to_tensors=False)
    state = init_state.copy()
    t = 0
    carbs, insulin = jnp.array(1000.), jnp.array(0.)
    dstate_dt = uva_padova_2008_dynamics(params, t, state, carbs, insulin)

    print("Blood glucose", observe_blood_glucose(params, state).item(), "mg/dL")
    print("Subcutaneous glucose", observe_subcutaneous_glucose(params, state).item(), "mg/dL")
    # check differentiability
    params_jac = jax.jacfwd(uva_padova_2008_dynamics)(params, t, state, carbs, insulin)
    print("Params jacobian", params_jac)
    state_jac = jax.jacfwd(uva_padova_2008_dynamics, argnums=2)(params, t, state, carbs, insulin)
    print("State jacobian", state_jac)
    carbs_jac = jax.jacfwd(uva_padova_2008_dynamics, argnums=3)(params, t, state, carbs, insulin)
    print("Carbs jacobian", carbs_jac)
    insulin_jac = jax.jacfwd(uva_padova_2008_dynamics, argnums=4)(params, t, state, carbs, insulin)
    print("Insulin jacobian", insulin_jac)