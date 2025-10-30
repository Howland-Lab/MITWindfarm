"""
IVP integration methods for Curled Wake Modeling

Kirby Heck
2025 May 05
"""

import numpy as np
from scipy import integrate  # import solve_ivp as solve_ivp_scipy


def rk4_step(t_n, u_n, dudt, dt):
    """
    Computes the next timestep of u_n given the finite difference function du/dt
    with a 4-stage, 4th order accurate Runge-Kutta method.

    Parameters
    ----------
    t_n : float
        time for time step n
    u_n : array-like
        condition at time step n
    dudt : function
        function du/dt(t, u)
    dt : float
        time step

    Returns u_(n+1)
    """
    k1 = dt * dudt(t_n, u_n)
    k2 = dt * dudt(t_n + dt / 2, u_n + k1 / 2)
    k3 = dt * dudt(t_n + dt / 2, u_n + k2 / 2)
    k4 = dt * dudt(t_n + dt, u_n + k3)

    u_n1 = u_n + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_n1


def EF_step(t_n, u_n, dudt, dt):
    """
    Simple forward Euler stepping scheme.

    Parameters
    ----------
    t_n : float
        time for time step n
    u_n : array-like
        condition at time step n
    dudt : function
        function du/dt(t, u)
    dt : float
        time step

    Returns u_(n+1)
    """
    u_n1 = u_n + dt * dudt(t_n, u_n)
    return u_n1


METHODS = {
    f"scipy_{key}".lower(): val for key, val in integrate._ivp.ivp.METHODS.items()
}
STEPS = {
    "rk4": rk4_step,
    "ef": EF_step,
}


def solve_ivp(dudt, T, u0, dt=0.1, f=rk4_step, end_exact=True, **kwargs):
    """
    General integration function which calls a step function multiple times depending
    on the parabolic integration strategy.

    Checks for two specific exceptions:
    - IntegrationException: raised when the integration fails at a specific time step.
    - DomainExpansionRequest: raised when the integration requests a domain expansion.

    Parameters
    ----------
    dudt : function
        Evolution function du/dt(t, u, ...)
    T : (2, )
        Time range
    u0 : array-like
        Initial condition of values
    dt : float
        Time step
    f : function
        Integration stepper function (e.g. RK4, EF, etc.)

    Returns
    -------
    t : (Nt, ) vector
        Time vector
    u(t) : (Nt, ...) array-like
        Solution to the parabolic ODE.
    """
    t = []
    ut = []

    u_n = u0  # initial condition
    t_n = T[0]

    ut.append(u_n)
    t.append(t_n)

    keep_going = True
    while keep_going:
        # update timestep
        t_n1 = t_n + dt
        if t_n1 > T[1]:
            if end_exact:
                dt = T[1] - t_n  # adjust the last step to end exactly at T[1]
                if dt == 0:
                    break  # avoid zero step size, not sure exactly how we get here
                t_n1 = T[1]
                keep_going = False
            else:
                break

        try:
            u_n1 = f(t_n, u_n, dudt, dt)
        except IntegrationException as e:
            # re-raise with additional state information
            raise IntegrationException(
                "Integration failed at time step.",
                partial_t=np.array(t),  # save integration up to this point
                partial_u=np.array(ut),
            ) from e
        except DomainExpansionRequest as e:
            # re-raise with additional state information
            e.partial_t = np.array(t)  # save integration up to this point
            e.partial_u = np.array(ut)
            raise e
            # raise DomainExpansionRequest(
            #     "Domain expansion requested during integration.",
            #     partial_t=np.array(t),
            #     partial_u=np.array(ut),
            #     expand_y=e.expand_y,
            #     expand_z=e.expand_z,
            # ) from e

        # save solution
        ut.append(u_n1)
        t.append(t_n1)

        # update:
        u_n = u_n1
        t_n = t_n1

    return np.array(t), np.array(ut)


def solve_ivp_interrupt(dudt, T, u0, f=METHODS["scipy_rk45"], **options):
    """
    Uses scipy's stepper functions but overrides the
    default scipy solve_ivp to allow for interruptions and
    custom Exception handling.
    """

    t0, tf = map(float, T)
    try:
        # for initializing the solver calls `dudt`, so this must be caught as well
        solver = f(dudt, t0, np.atleast_1d(u0), t_bound=tf, **options)
    except (DomainExpansionRequest, IntegrationException) as e:
        e.partial_t, e.partial_u = [T[0]], u0
        raise e

    t = [solver.t]
    ut = [solver.y]

    while solver.status == "running":
        try:
            message = solver.step()
        except (DomainExpansionRequest, IntegrationException) as e:
            e.partial_t = np.array(t)
            e.partial_u = np.array(ut)
            raise e
        except Exception as e:
            raise IntegrationException(
                f"Integration failed during step: {e}",
                partial_t=np.array(t),  # save integration up to this point
                partial_u=np.array(ut),
                extra=str(e),
            ) from e

        # append solution
        t.append(solver.t)
        ut.append(solver.y)

    return np.array(t), np.array(ut)


class Integrator:
    def __init__(self, scheme="rk4"):
        self.use_scipy = False
        self.use_scipy_method = False

        if callable(scheme):
            self.step_fn = scheme

        elif scheme.lower() in METHODS:
            self.use_scipy = True
            self.use_scipy_method = True
            self.step_fn = METHODS[scheme.lower()]

        elif scheme.lower() in STEPS:
            self.step_fn = STEPS[scheme.lower()]

        elif scheme.lower() == "scipy":
            self.use_scipy = True

        else:
            avail = ", ".join(sorted(METHODS.keys()) + list(STEPS.keys()) + ["scipy"])
            raise ValueError(
                f"Unknown integration scheme: {scheme}, choose from {avail}"
            )

    def __call__(self, dudt, T, u0, **kwargs):
        """Calls the integrator function"""

        if self.use_scipy_method:
            _kwargs = dict(rtol=1e-4, max_step=0.5)  # helps with stability
            _kwargs.update(kwargs)
            u0 = np.atleast_1d(u0)
            shape = u0.shape
            try:
                t, y = solve_ivp_interrupt(
                    dudt, T, u0.flatten(), f=self.step_fn, **kwargs
                )
                return t, y.reshape(len(t), *shape).squeeze()
            except (DomainExpansionRequest, IntegrationException) as e:
                e.partial_u = e.partial_u.reshape(len(e.partial_t), *shape).squeeze()
                raise e

        elif self.use_scipy:
            raise NotImplementedError(
                "scipy deprecated because it does not allow for exception handling"
            )

        else:
            return solve_ivp(dudt, T, u0, f=self.step_fn, **kwargs)


class IntegrationException(Exception):
    """Custom exception for integration failures."""

    def __init__(
        self,
        message,
        partial_t=None,
        partial_u=None,
        extra=None,
    ):
        super().__init__(message)
        self.partial_t = partial_t
        self.partial_u = partial_u
        self.message = message
        self.extra = extra

    def __str__(self):
        return self.message


class DomainExpansionRequest(Exception):
    """Custom exception to signal a request for domain expansion."""

    def __init__(
        self,
        message,
        partial_t=None,
        partial_u=None,
        expand_y=None,
        expand_z=None,
        extra=None,
    ):
        super().__init__(message)
        self.partial_t = partial_t
        self.partial_u = partial_u
        self.expand_y = expand_y
        self.expand_z = expand_z
        self.message = message
        self.extra = extra

    def __str__(self):
        return self.message


if __name__ == "__main__":
    # Example usage:
    # Define a simple ODE: du/dt = -u
    def dudt(t, u):
        # if u < 5e-2:
        #     raise IntegrationException("less than threshold")
        return -u

    integrator = Integrator("scipy_rk45")
    try:
        time, solution = integrator(dudt, (0, 10), 1, dt=0.1, end_exact=True)
    except IntegrationException as e:
        time, solution = e.partial_t, e.partial_u
    print("Time:", time[-1])
    print("Solution:", solution)
    print(solution.squeeze().shape)
    # Compare with analytical solution e^(-t)
    print("Analytical Solution:", np.exp(-time))
