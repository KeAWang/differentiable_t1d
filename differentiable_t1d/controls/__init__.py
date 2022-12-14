from .bolus import BolusController, BolusParams
import jax
import jax.numpy as jnp


def square_profile(dose, duration, t):
    """Square wave profile

    Example usage:
        t = jnp.linspace(0, 100, 1000)
        ra = jax.vmap(square, (None, None, 0))(1, 1 / 0.1, t)
    """
    return dose * (t < duration) * (t >= 0) / duration


def two_compartment_profile(dose, k1, k2, t):
    """Two compartment chain model solution
    Consider
        dx1/dt = -k1 * x1
        dx2/dt = k1 * x1 - k2 * x2
        x1(0) = dose
        x2(0) = 0
    This function gives k2 * x2, the rate at which contents flow out of the system.
    Note that integrating this function from time=0 to infinity will give you `dose`.

    Example usage:
        t = jnp.linspace(0, 100, 1000)
        ra = jax.vmap(two_compartment_chain, (None, None, None, 0))(1, 0.1, 0.2, t)
    """
    t = t * (t >= 0.0)  # prevent exponential blowup
    return (
        dose
        * k1
        * k2
        * (jnp.exp(-k1 * t) - jnp.exp(-k2 * t))
        / (k2 - k1 + 1e-10)
        * (t >= 0.0)
    )


def two_equal_compartment_profile(dose, k, t):
    """two_compartment_chain when k1=k2 (take limit)
    Example usage:
        t = jnp.linspace(0, 100, 1000)
        ra = jax.vmap(two_equal_compartment_chain, (None, None, 0))(1, 0.1, t)

    """
    t = t * (t >= 0.0)  # prevent exponential blowup
    return dose * k * k * t * jnp.exp(-k * t) * (t >= 0.0)


def make_sequential_profile(profile_func, params, timestamps):
    # params is a tuple of 1D arrays, length equal to the number of parameter arguments of profile_func
    # profile_func: (*args, t) -> profile(t)
    # timestamps is a 1D array
    # length of each 1D array is equal to length of timestamps, equal to number of meals
    """
    Example usage where we have 2 meals:
        u = make_sequential_profile(two_equal_compartment_chain,
            (jnp.array([1., 1.]), jnp.array([0.1, 0.2])),
            jnp.array([0., 50.]),
        )
        t = jnp.linspace(0, 100, 1000)
        ra = jax.vmap(u)(t)
    """
    _profile_func = jax.vmap(profile_func, in_axes=(0,) * (len(params) + 1))

    def func(t):
        assert t.ndim == 0
        return _profile_func(*params, t - timestamps).sum(0)  # sum across timestamps

    return func


__all__ = [
    "BolusController",
    "BolusParams",
    "square_profile",
    "two_compartment_profile",
    "two_equal_compartment_profile",
    "make_sequential_profile",
]
