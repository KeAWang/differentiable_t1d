import numpy as np
import torch
import pandas as pd
from collections import namedtuple
from typing import Union


# TODO: make sure that all units make sense
# TODO: make vpatient_params.csv installable via pip

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
    df = pd.read_csv("vpatient_params.csv", index_col=0)
    print("Loading patient", patient_id)
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

if __name__ == "__main__":
    params, unused_params, init_state = initialize_patient(1)
    print(params)
    print(unused_params)
    print(init_state)