from typing import NamedTuple
import jax.numpy as jnp


class BolusParams(NamedTuple):
    insulin_carb_ratio: float
    correction_factor: float
    target_bg: float  # mg/dL
    prebolus_time: float  # in minutes


class BolusController:
    @staticmethod
    def apply(
        params: BolusParams,
        meal_times: jnp.ndarray,
        carb_amounts: jnp.ndarray,  # carbs in grams
        t: float,
        dt: float,
        bg: jnp.ndarray,
    ) -> jnp.ndarray:
        bolus_times = meal_times - params.prebolus_time
        in_bin = (bolus_times <= t) * (t < bolus_times + dt)
        # only dose if t is in same time bin as the bolus time of a meal
        # we sum to handle multiple meals sharing the same time bin
        carb_amount = jnp.sum(carb_amounts * in_bin)

        cr_dose = carb_amount / params.insulin_carb_ratio
        cf_dose = (
            (carb_amount > 0.0) * (bg - params.target_bg) / params.correction_factor
        )
        dose = cr_dose + cf_dose
        dose = jnp.clip(dose, a_min=0.0)
        return dose
