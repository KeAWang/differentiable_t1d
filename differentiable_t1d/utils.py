import numpy as np
import torch

import pandas as pd
from collections import namedtuple
import pkg_resources
from scipy.stats import truncnorm
from typing import Union


VPATIENT_PARAMS_PATH = pkg_resources.resource_filename("differentiable_t1d", "files/vpatient_params.csv")
VPATIENT_CONTROL_PARMS_PATH = pkg_resources.resource_filename("differentiable_t1d", "files/vpatient_control_params.csv")

Params = namedtuple(
    "Params",
    [
        "b",  # the second flex point of gastric emptying rate
        "d",  # the first flex point of gastric emptying rate (called c in the original paper on oral glucose)
        "kmin",
        "kmax",
        "kabs",
        "kp1",
        "kp2",
        "kp3",
        "Fsnc",
        "ke1",
        "ke2",
        "f",
        "Vm0",
        "Vmx",
        "Km0",
        "k1",
        "k2",
        "m1",
        "m2",
        "m30",
        "m4",
        "ka1",
        "ka2",
        "Vi",
        "p2u",
        "Ib",
        "ki",
        "kd",
        "ksc",
        "BW",
        "Vg",
    ],
)


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
    """ Load patient parameters and initial state"""    

    all_params = load_patient(patient_id)
    params = Params( **{k: np.array(float(v)) for k, v in all_params.items() if k in Params._fields})
    init_state = np.array([all_params[f"x0_{i}"] for i in range(1, 14)])
    unused_params = { k: v for k, v in all_params.items() if (k not in Params._fields) and (k[:2] != "x0")}

    if to_tensors:
        params = Params(*[torch.as_tensor(p, dtype=torch.float) for p in params])
        unused_params = {k: torch.as_tensor(v, dtype=torch.float) for k, v in unused_params.items()}
        init_state = torch.as_tensor(init_state, dtype=torch.float)

    return params, unused_params, init_state


def generate_meals(seed=0, num_days=1, bw=60.0):
    # bodyweight bw in kg
    # each row of meals is [time, amount] in minutes and grams
    rng = np.random.RandomState(seed)
    meals = []
    for i in range(num_days):
        new_meals = generate_meals_one_day(rng, bw)
        new_meals[:, 0] += i * 24 * 60
        meals.append(new_meals)
    return np.concatenate(meals)

def generate_meals_one_day(rng, bw: float):
    meals = []

    # Probability of taking each meal
    # [breakfast, snack1, lunch, snack2, dinner, snack3]
    prob = np.array([1.0, 0.25, 1.0, 0.25, 1.0, 0.5])
    time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60.
    time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60.
    time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60.
    time_sigma = np.array([60, 30, 60, 30, 60, 30])
    amount_mu = np.array([0.75, 0.5, 1.0, 0.5, 1.25, 0.5]) * bw
    amount_sigma = np.array([0.15, 0.05, 0.15, 0.05, 0.15, 0.05]) * bw

    for p, tlb, tub, tbar, tsd, mbar, msd in zip(prob, time_lb, time_ub,
                                                 time_mu, time_sigma,
                                                 amount_mu, amount_sigma):
        if rng.rand() < p:
            tmeal = np.round(
                truncnorm.rvs(a=(tlb - tbar) / tsd,
                              b=(tub - tbar) / tsd,
                              loc=tbar,
                              scale=tsd,
                              random_state=rng),
                )
            amount = round(max(rng.normal(mbar, msd), 0.))
            meal = np.array([tmeal, amount])  # time in minutes, amount in grams
            meals.append(meal)
    meals = np.stack(meals)
    return meals

def generate_insulin_events(meals, icr, num_days):
    # Assumes meals are in grams, basal is in units, icr is in grams per unit, and bw is in kg
    bolus_times = meals[:, 0] - 5. # bolus a few minutes before meals
    bolus_dose = meals[:, 1] / icr  # units: IU
    insulin_times = np.concatenate([bolus_times])
    insulin_dose = np.concatenate([bolus_dose])

    # sort by time
    idx = np.argsort(insulin_times)
    insulin_times = insulin_times[idx]
    insulin_dose = insulin_dose[idx]
    insulin_events = np.stack([insulin_times, insulin_dose], axis=1)
    return insulin_events

def generate_meals_and_insulin_events(icr, bw, num_days, seed):
    meals = generate_meals(seed, num_days, bw)
    insulin_events = generate_insulin_events(meals, icr=icr, num_days=num_days)
    return meals, insulin_events

if __name__ == "__main__":
    params, unused_params, init_state = initialize_patient(1)
    print(params)
    print(unused_params)
    print(init_state)