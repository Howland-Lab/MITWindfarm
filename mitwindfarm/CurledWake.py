"""
Curled wake model solver in MITWindfarm.
(Now in a separate file)

Kirby Heck
2025 June 6
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from warnings import warn

from numpy.typing import ArrayLike
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import interpn, interp1d

from mitwindfarm.Windfield import Windfield
from mitwindfarm.Rotor import RotorSolution
from mitwindfarm.utils.integrate import (
    Integrator,
    IntegrationException,
    DomainExpansionRequest,
)
from mitwindfarm.utils.differentiate import second_der


class CurledWakeWindfield(Windfield):
    """
    Windfield for the curled wake model. This wind field HAS a base windfield
    which represents the base flow, and then adds turbines on top of the base
    windfield.

    The CurledWakeWindfield is also the flow solver/forward marching method
    for the Curled Wake Model. It does the following:
    - Applies initial conditions
    - Manages the domain size, expanding as necessary
    - Manages turbulence models
    - Marches the wind field (and possibly other fields: dk, dv, dw) forward
        in space
    """

    def __init__(
        self,
        base_windfield: Windfield,
        integrator: str = "scipy_rk45",
        ivp_kwargs: dict = None,
        dx: float = 0.1,
        dy: float = 0.1,
        dz: float = 0.1,
        ybuff: float = 3,
        zbuff: float = 2,
        N_vortex: int = 10,
        sigma_vortex: float = 0.2,
        smooth_fact: float = 1,
        k_model: Literal["const", "k-l"] = "const",
        k_kwargs: dict = None,
        ic_method: Literal["du", "fx"] = "du",
        clip_u: float = 0.1,
        use_r4: bool = True,
        auto_expand: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the wind field with specified parameters.

        Parameters:
        - base_windfield: The base wind field to be used.
        - integrator: IVP solver to be used for the wind field (default: "scipy_rk45").
            see integrate.Integrator for options.
        - ivp_kwargs: Additional arguments for the integrator (default: None).
        - dx: Grid spacing in the x-direction, non-dim (default: 0.2).
        - dy: Grid spacing in the y-direction, non-dim (default: 0.1).
        - dz: Grid spacing in the z-direction, non-dim (default: 0.1).
        - ybuff: Buffer in the y-direction (default: 3).
        - zbuff: Buffer in the z-direction (default: 2).
        - smooth_fact: Smoothing factor for the initial condition stencil (default: 1).
        - N_vortex: Number of vortices to use for the dv, dw initial conditions (default: 10).
        - sigma_vortex: radius for the vortex de-singularization (default: 0.2).
        - k_model: Turbulence model to use (default: "k-l").
        - k_kwargs: Additional arguments for the turbulence model (default: None).
        - ic_method: Method for initial condition stamping (default: "du").
            NOTE: "fx" is experimental and only solves for EF marching.
        - clip_u: Whether to clip the u-velocity to prevent negative values (default: 0.1).
            Set to <= 0 to disable clipping
        - use_r4: Whether to use the r4 rotor radius for initial conditions (default: True).
        - auto_expand: Whether to automatically expand the domain when needed (default: True).
        - verbose: Prints debug information if True (default: False).
        """
        self.base_windfield = base_windfield
        self.integrator = Integrator(integrator)
        self.ivp_name = integrator
        self.ivp_kwargs = ivp_kwargs if ivp_kwargs is not None else dict()
        self.dx, self.dy, self.dz = dx, dy, dz
        self.N_vortex = N_vortex
        self.sigma_vortex = sigma_vortex

        # The following will get initialized later in check_grid_init()
        self.grid = None  # list of [x, y, z] axes
        self.du = None  # solved du-field
        self.dv = None  # solved dv-field
        self.dw = None  # solved dw-field
        self.dk = None  # solved k_wake field

        if "scipy" not in self.ivp_name:
            self.ivp_kwargs.setdefault("dt", self.dx)

        self.ybuff = ybuff
        self.zbuff = zbuff

        self.extra_fx = None
        self.ic_method = ic_method  # "fx" DOES NOT WORK - ONLY USE "du"

        self.clip_u = clip_u
        self.use_r4 = use_r4
        self.auto_expand = auto_expand

        self.verbose = verbose

        # Turbulence modeling
        self.k_model = k_model  # turbulence model to use (default: "k-l")
        self.k_kwargs = k_kwargs if k_kwargs is not None else {}
        self.k_module = CurledTurbulenceModel.get_model(
            self.k_model, curledwake=self, **self.k_kwargs
        )

        self.smooth_fact = smooth_fact  # smoothing factor for the IC stencil
        self.turbines = []

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        self.march_to(x=x, y=y, z=z)  # check that the forward marching is sufficient

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        x, y, z = np.broadcast_arrays(x, y, z)

        wsp_base = self.base_windfield.wsp(x, y, z)
        wsp_wakes = interpn(
            (self.grid[0], self.grid[1], self.grid[2]),
            self.du,
            (x.ravel(), y.ravel(), z.ravel()),
            method="linear",
            bounds_error=False,
            fill_value=0,
        ).reshape(x.shape)
        wsp = wsp_base + wsp_wakes
        return wsp

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        self.march_to(x=x, y=y, z=z)  # check that the forward marching is sufficient

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        x, y, z = np.broadcast_arrays(x, y, z)

        ti_base = self.base_windfield.TI(x, y, z)
        wsp_base = self.base_windfield.wsp(x, y, z)

        k_wake = interpn(
            (self.grid[0], self.grid[1], self.grid[2]),
            self.dk,
            (x.ravel(), y.ravel(), z.ravel()),
            method="linear",
            bounds_error=False,
            fill_value=0,
        ).reshape(x.shape)
        wsp = self.wsp(x, y, z)
        ti = np.sqrt((wsp_base * ti_base) ** 2 + 2 * k_wake / 3) / wsp
        return ti

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        # TODO: FIX
        return self.base_windfield.wdir(x, y, z)

    def march_to(self, x: float, y: float, z: float) -> None:
        """
        March the wind field to the specified coordinates.

        Parameters:
        - x: x-coordinate.
        - y: y-coordinate.
        - z: z-coordinate.
        """
        self.check_grid_init(x=x, y=y, z=z)  # check if the grid is initialized
        self._march(xmax=np.max(x))

    def stamp_ic(
        self,
        rotor_solution: RotorSolution,
        xt,
        yt,
        zt,
        smooth_fact=None,
        D=1,
    ) -> None:
        """
        Stamp the initial condition of the rotor solution into the wind field.

        Parameters:
        - rotor_solution: The rotor solution to stamp into the wind field.
        - smooth_fact: Smoothing factor for the initial condition stencil.
        - D: Diameter of the rotor (default: 1).
        """
        # first, add the turbine to the list of turbines
        self.turbines.append(TurbineProperties(xt, yt, zt, D, rotor_solution))

        # adjust grid bounds if necessary
        self.adjust_grid_bounds(x=None, y=yt, z=zt, add_buffers=True)

        # streamwise velocity initial condition:
        smooth_fact = self.smooth_fact if smooth_fact is None else smooth_fact
        rotor = rotor_solution
        r4 = (
            np.sqrt((1 - rotor.extra.an) / rotor.extra.u4) * D / 2
            if self.use_r4
            else D / 2
        )
        ay = r4 * np.cos(rotor.yaw)
        az = r4  # TODO: could factor in rotor tilt later on
        shape = ic_stencil(
            self.grid[1],
            self.grid[2],
            yt,
            zt,
            smooth_fact=smooth_fact,
            ay=ay,
            az=az,
        )

        if self.ic_method == "fx":
            # NOTE: DO NOT USE
            thrust = -rotor.Ct * 0.5 * np.pi / 4
            self.extra_fx += (
                shape * thrust / (np.sum(shape) * self.dy * self.dz * self.dx)
            )
            warn(
                "`fx` is not a reliable method for stamping initial conditions. Use `du` instead."
            )
        else:
            # stamp the rotor solution into the wind field
            # TODO: check du is negative?
            delta_u = rotor.u4 - rotor.REWS  # delta_u, adjusted by REWS
            self.du[-1, ...] += shape * delta_u

        # dv, dw initial conditions:
        if rotor.yaw == 0:
            return  # no additional dv, dw to stamp in for this turbine

        # TODO: Put this in a separate module
        # self.N_vortex = 10  # make this a parameter
        # self.vortex_sigma = 0.2  # sigma/D, for de-singularization

        # r-axis: clip edges to prevent singularities
        r_i = np.linspace(-(D - self.dz) / 2, (D - self.dz) / 2, self.N_vortex)
        # NOTE: rotor.Ct differs from Shapiro et al. (2018) definition - includes cos^2(yaw)
        Gamma_0 = 0.5 * D * rotor.REWS * rotor.Ct * np.sin(rotor.yaw)
        Gamma_i = (
            Gamma_0 * 4 * r_i / (self.N_vortex * D**2 * np.sqrt(1 - (2 * r_i / D) ** 2))
        )

        # generally, vortices can decay, so sigma should be a function of x  # TODO
        sigma = self.sigma_vortex * D

        # now we build the main summation, which is 3D (y, z, i)
        yG, zG = np.meshgrid(self.grid[1], self.grid[2], indexing="ij")
        yG = yG[..., None]
        zG = zG[..., None]
        rsq = (yG - yt) ** 2 + (zG - zt - r_i[None, None, :]) ** 2  # 3D grid variable
        rsq = np.clip(rsq, 1e-8, None)  # avoid singularities

        # put pieces together:
        exponent = 1 - np.exp(-rsq / sigma**2)
        summation = exponent / (2 * np.pi * rsq) * Gamma_i[None, None, :]

        # sum all vortices along last dim
        v = np.sum(summation * (zG - zt - r_i[None, None, :]), axis=-1)
        w = np.sum(summation * -(yG - yt), axis=-1)
        self.dv[-1, ...] += v  # stamp in dv
        self.dw[-1, ...] += w  # stamp in dw

    def adjust_grid_bounds(
        self,
        x: ArrayLike = None,
        y: ArrayLike = None,
        z: ArrayLike = None,
        add_buffers: bool = True,
    ) -> None:
        """
        Expand the dimensions of the wind field to accommodate wake
        expansion and additional turbines.

        A buffer of xbuff, ybuff, zbuff will be applied to points checked.

        Parameters:
        - x: x-coordinates, optional
        - y: y-coordinates, optional
        - z: z-coordinates, optional
        - add_buffers: whether to add buffers to the grid (default: True)
        """
        if self.grid is None:
            raise AttributeError("Grid not initialized")

        # check and possibly expand grid with zero-padding
        yax, zax = self.grid[1], self.grid[2]
        ypad, zpad = (0, 0), (0, 0)
        if y is not None:
            y = np.atleast_1d(y)
            ymin = np.min(y) - self.ybuff * add_buffers
            ymax = np.max(y) + self.ybuff * add_buffers
            ypad_lower = np.arange(yax[0] - self.dy, ymin - self.dy, -self.dy)[::-1]
            ypad_upper = np.arange(yax[-1] + self.dy, ymax + self.dy, self.dy)
            # update y-grid
            self.grid[1] = np.concatenate([ypad_lower, yax, ypad_upper])
            ypad = (len(ypad_lower), len(ypad_upper))

        if z is not None:
            z = np.atleast_1d(z)
            zmin = np.min(z) - self.zbuff * add_buffers
            zmax = np.max(z) + self.zbuff * add_buffers
            zpad_lower = np.arange(zax[0] - self.dz, zmin - self.dz, -self.dz)[::-1]
            zpad_upper = np.arange(zax[-1] + self.dz, zmax + self.dz, self.dz)
            # update z-grid
            self.grid[2] = np.concatenate([zpad_lower, zax, zpad_upper])
            zpad = (len(zpad_lower), len(zpad_upper))

        # # now we need to pad the du, dv, dw fields
        self.du = np.pad(self.du, ((0, 0), ypad, zpad), mode="constant")
        self.dv = np.pad(self.dv, ((0, 0), ypad, zpad), mode="constant")
        self.dw = np.pad(self.dw, ((0, 0), ypad, zpad), mode="constant")
        self.dk = np.pad(self.dk, ((0, 0), ypad, zpad), mode="constant")
        self.extra_fx = np.pad(self.extra_fx, (ypad, zpad), mode="constant")

    def check_grid_init(
        self, x: ArrayLike = None, y: ArrayLike = None, z: ArrayLike = None
    ) -> None:
        """Initializes self.grid if it doesn't exist."""
        if self.grid is None:
            # Initialize the grid if it doesn't exist. Automatically add buffers
            self.grid = [
                np.atleast_1d(x),
                np.arange(-self.ybuff, self.ybuff + self.dy, self.dy) + y,
                np.arange(-self.zbuff, self.zbuff + self.dz, self.dz) + z,
            ]

            self.du = np.zeros(self.shape)
            self.dv = np.zeros(self.shape)
            self.dw = np.zeros(self.shape)
            self.dk = np.zeros(self.shape)
            self.extra_fx = np.zeros(self.shape[1:])

    def _march(self, xmax) -> None:
        """
        Forward marches delta_u and delta_k fields.
        Can also expand to march delta_v and delta_w fields,
        but for now these are constant in x (except when additional
        yawed wakes are stamped in with initial conditions).

        Returns
        - None (updates grid and self.du, self.dv, self.dw in place)
        """
        if xmax <= np.max(self.grid[0]):
            return  # nothing to compute!

        # for now, _v and _w (2D slices of dv, dw) do not evolve in space
        _v = self.dv[-1, ...]  # last slice of dv
        _w = self.dw[-1, ...]
        y = self.grid[1]
        z = self.grid[2]
        ybnd, zbnd = (0, 0), (0, 0)  # initialize variables for bound checking

        def _step(x, _state):
            """
            Step for all functions d()/dx.
            In the standard curled wake model, this is just Delta_u, but
            in more advanced modeling (Klemmer and Howland, 2025), this also
            marches the k_wake field forward simultaneously.

            Because some integrators (e.g., scipy) require a 1D array, we may
            to reshape arrays to compute derivatives, then pack them back
            into a flattened array.
            """
            if np.any(np.isnan(_state)):
                raise IntegrationException(
                    f"nan value encountered in state at x={x:.3f}",
                )

            # parse inputs from current state
            _u, _k = self.k_module.unpack_inputs(_state)
            _u = np.clip(_u, None, 0)  # no positive velocity deficits... for now

            # ======== check state bounds for domain expansion ========
            if self.auto_expand:
                check_yz = []
                check_yz.append(check_state_bounds(_u, thresh=1e-4))
                check_yz.append(check_state_bounds(_k, thresh=1e-6))
                if np.any([check_yz]):
                    # if any of the checks fail, we need to expand the domain along those dimensions
                    ybnd, zbnd = np.max(check_yz, axis=0)
                    raise DomainExpansionRequest(
                        f"Expanding domain at {x=:.2f}", expand_y=ybnd, expand_z=zbnd
                    )

            # ========= assemble variables and fields =========
            # Full velocity fields for advection:
            wsp = self.base_windfield.wsp(x, y[:, None], z[None, :])
            wdir = self.base_windfield.wdir(x, y[:, None], z[None, :])
            # compute k_base: assume TI = sqrt(2/3 k)/U
            kb = (self.base_windfield.TI(x, y[:, None], z[None, :]) * wsp) ** 2 * 3 / 2
            u = _u + wsp * np.cos(wdir)
            v = _v + wsp * np.sin(wdir)
            w = _w + 0
            k = _k + kb

            if (self.clip_u > 0) and np.any(u < self.clip_u):
                u = np.clip(u, self.clip_u, None)

            self.k_module.update_wake_fields(u, v, w, k, _u, _v, _w, _k)
            nu_T = self.k_module.nu_T(x)

            # ============== du/dx computation ==============
            # gradient fields of \Delta u:
            dudy = np.gradient(_u, y, axis=0)
            dudz = np.gradient(_u, z, axis=1)
            d2udy2 = second_der(_u, self.dy, axis=0)
            d2udz2 = second_der(_u, self.dz, axis=1)
            dudx = (-v * dudy - w * dudz + nu_T * (d2udy2 + d2udz2) + self.extra_fx) / u

            self.extra_fx *= 0  # reset extra forces after they are used - TODO: remove

            # ============== dk/dx computation ==============
            dkdx = self.k_module.compute_dkdx()

            ret = self.k_module.pack_outputs(dudx, dkdx)
            return ret

        # use last slice as initial condition, turbulence model may update IC
        ic = self.k_module.update_ic(self.du[-1, ...])

        try:
            x, ret = self.integrator(
                _step, [self.grid[0].max(), xmax], ic, **self.ivp_kwargs
            )
        except IntegrationException as e:
            x = e.partial_t
            ret = e.partial_u
            if self.verbose:
                print(
                    f"IntegrationException, exiting integration at x={max(x)}:\n\t",
                    e,
                )
        except DomainExpansionRequest as e:
            x = e.partial_t
            ret = e.partial_u
            ybnd = e.expand_y
            zbnd = e.expand_z
            # every time we get here, expand the expansion...
            if np.any(ybnd):
                self.ybuff += 1
            if np.any(zbnd):
                self.zbuff += 1

        if len(x) > 1:
            # append and concatenate progress
            du_new, dk_new = self.k_module.unpack_outputs(ret)
            dv_new = np.repeat(self.dv[-1][None, ...], len(x) - 1, axis=0)
            dw_new = np.repeat(self.dw[-1][None, ...], len(x) - 1, axis=0)

            self.du = np.concatenate([self.du, du_new[1:, ...]], axis=0)
            self.dv = np.concatenate([self.dv, dv_new], axis=0)
            self.dw = np.concatenate([self.dw, dw_new], axis=0)
            self.dk = np.concatenate([self.dk, dk_new[1:, ...]], axis=0)
            self.grid[0] = np.concatenate([self.grid[0], x[1:]])

        # if we hit a DomainExpansionRequest, need to expand the grid and continue integrating
        if np.any([ybnd, zbnd]):
            if self.verbose:
                print(f"Expanding grid at x={np.max(x):.2f} in y={ybnd} and z={zbnd}")

            self.adjust_grid_bounds(
                y=[y[0] - ybnd[0] * self.ybuff, y[-1] + ybnd[1] * self.ybuff],
                z=[z[0] - zbnd[0] * self.zbuff, z[-1] + zbnd[1] * self.zbuff],
                add_buffers=False,
            )
            self._march(xmax=xmax)  # recursive call to continue marching

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns the shape of the grid.
        """
        return (len(self.grid[0]), len(self.grid[1]), len(self.grid[2]))


@dataclass
class TurbineProperties:
    """
    Class to hold turbine properties.
    """

    xt: float
    yt: float
    zt: float
    D: float
    rotor_solution: RotorSolution


# ===========================================================================
# ======================== TURBULENCE MODELING ==============================
# ===========================================================================


class CurledTurbulenceModel(ABC):
    """
    Base class for the turbulence model in the curled wake model.

    Keeps track of all sub-classes with a self-registering factory.
    """

    _registry = {}
    name: str  # fill this in for each model

    def __init__(self, curledwake: CurledWakeWindfield):
        self.curledwake = curledwake  # link to the curled wake solver object
        self.need_reshape = False

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called when a subclass is created.
        It registers the subclass in the turbulence model registry.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls._registry[cls.name] = cls
        else:
            raise ValueError(f"Subclass {cls.__name__} must define a 'name' attribute.")

    @classmethod
    def get_model(cls, name: str, *args, **kwargs):
        """
        Factory method to get an instance of a turbulence model.
        """
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(
                f"Unknown turbulence model: '{name}'. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return model_class(*args, **kwargs)

    @abstractmethod
    def nu_T(self, x):
        """
        Returns nu_T, the eddy viscosity for the turbulence model
        """
        ...

    def update_ic(self, ic):
        """
        Update the initial condition for the turbulence model
        """
        return ic

    def unpack_inputs(self, state):
        """
        Unpack inputs for the forward marching u^n and possibly k^{n+1}

        Returns:
        - du: wake deficit
        - dk: k_wake field (default is 0)
        """
        if state.ndim == 1:
            self.need_reshape = True  # we will need this in packing the outputs
            state = state.reshape(self.curledwake.shape[1:])
        return state, np.zeros_like(state)

    def pack_outputs(self, dudx, dkdx):
        """
        Pack outputs for the forward marching dudx and possibly dkdx
        """
        if self.need_reshape:
            dudx = dudx.flatten()
        return dudx

    def unpack_outputs(self, ret):
        """
        Unpack the output result of the forward marching
        """
        return ret, np.zeros_like(ret)  # default: no k_wake field

    def compute_dkdx(self):
        """
        Computes the dk/dx term for the turbulence model.
        """
        return None  # by default, this is not needed

    def update_wake_fields(self, u, v, w, k, du, dv, dw, dk):
        """
        Update the wake fields for the turbulence model.
        """
        pass  # by default, these are not needed

    def __repr__(self):
        return f"CurledTurbulenceModel: {self.__class__.__name__}"


class CurledTurbulenceModel_const(CurledTurbulenceModel):
    """
    Constant eddy viscosity model for the curled wake model.
    """

    name = "const"

    def __init__(self, curledwake, nu_T=1e-3, **kwargs):
        """
        Initializes a constant eddy viscosity model with fixed model parameters.
        """
        super().__init__(curledwake=curledwake)
        self.nu_eff = nu_T

    def nu_T(self, x):
        """
        Returns the constant eddy viscosity.
        """
        return self.nu_eff


class CurledTurbulenceModel_2021(CurledTurbulenceModel):
    """
    Curled wake turbulence model from Martínez-Tossas et al. (2021) WES paper

    Uses an ABL mixing length model proposed by Blackadar (1962)
    """

    name = "2021"

    def __init__(
        self,
        curledwake,
        C: float = 4,
        lam: float = None,
        Ro: float = None,
        kappa: float = 0.4,
    ):
        """
        Initializes the turbulence model. Note that all variables must
        be non-dimensionalized or otherwise consistent with the
        curled wake solver.

        Parameters:
        - curledwake: The curled wake solver
        - C: Constant for the turbulence model (default: 4)
        - lam: Mixing length, non-dimensionalized (default: None)
        - Ro: Turbine diameter-based rossby number G/(f_c * D) (default: None)
        - kappa: von Karman constant (default: 0.4)
        """
        super().__init__(curledwake=curledwake)
        self.C = C
        self.Ro = Ro
        self.kappa = kappa
        if Ro is not None:
            self.lam = 0.00027 * Ro  #  lam/D
        elif lam is not None:
            self.lam = lam
        else:
            raise AttributeError(
                "Either `lam` or `Ro` must be provided to the turbulence model."
            )

    def nu_T(self, x):
        """
        Computes Eq. 13 in Martínez-Tossas et al. (2021)

        Note that as the baseflow may vary as a function of x, y, z,
        dU/dz is computed at the current x, y, z locations.
        """
        yg, zg = np.meshgrid(
            self.curledwake.grid[1], self.curledwake.grid[2], indexing="ij"
        )
        U = self.curledwake.base_windfield.wsp(x, yg, zg)
        dUdz = np.gradient(U, self.curledwake.dz, axis=-1)
        lmix = self.kappa * zg / (1 + self.kappa * zg / self.lam)
        lmix = np.clip(lmix, 1e-2, None)  # mixing length must be non-negative

        return self.C * lmix**2 * np.abs(dUdz)


class CurledTurbulenceModel_kl(CurledTurbulenceModel):
    """
    Class for the k-l turbulence model in the curled wake model.

    See Klemmer and Howland, JRSE (2025) for details and derivation.
    """

    name = "k-l"

    def __init__(self, curledwake, C_nu=0.04, C_k1=1, C_k2=1):
        """
        Initializes a k-l turbulence model with fixed model parameters.

        Parameters
        - curledwake: The curled wake solver
        - C_nu: Constant for the eddy viscosity (default: 0.04)
        - C_k1: transport term coefficient (default: 1)
        - C_k2: dissipation term coefficient (default: 1)
        """
        super().__init__(curledwake=curledwake)
        self.C_nu = C_nu
        self.C_k1 = C_k1
        self.C_k2 = C_k2
        self.nu_T_cached = 0
        self.dk = None  # we will save this field from parsing inputs
        self.u, self.v, self.w, self.k = None, None, None, None
        self.du, self.dv, self.dw, self.dk = None, None, None, None

    def update_ic(self, ic):
        """
        Stacks the k_wake field to the initial condition for the forward marching.
        """
        return np.stack([ic, self.curledwake.dk[-1, ...]], axis=-1)

    def update_wake_fields(self, u, v, w, k, du, dv, dw, dk):
        """
        Update the wake fields for the turbulence model.
        """
        self.u = u
        self.v = v
        self.w = w
        self.k = k
        self.du = du
        self.dv = dv
        self.dw = dw
        self.dk = dk

    def nu_T(self, x):
        """
        Computes Eq. 6 in Klemmer and Howland (2025)
        """
        lmix = interpolate_lmix(self.du, self.curledwake.grid[1])[:, None]
        if np.any(lmix <= 0):
            raise IntegrationException("lmix is non-positive")

        heaviside = get_heaviside(x, self.curledwake.grid[1], self.curledwake.turbines)[
            :, None
        ]
        self.nu_T_cached = self.C_nu * (
            (1 - heaviside) * np.sqrt(np.clip(self.k - self.dk, 0, None)) * 1
            + heaviside * np.sqrt(np.clip(self.k, 0, None)) * lmix
        )
        return self.nu_T_cached

    def compute_dkdx(self):
        """
        Computes the dk/dx term for the turbulence model.
        """
        y = self.curledwake.grid[1]
        z = self.curledwake.grid[2]
        nu_T = self.nu_T_cached
        lmix = interpolate_lmix(self.du, y)[:, None]
        if np.any(lmix <= 0):
            raise IntegrationException("lmix is non-positive")

        # transport equation for k_wake, written in parabolic form:
        dkdx = (
            -self.v * np.gradient(self.dk, y, axis=0)
            - self.w * np.gradient(self.dk, z, axis=1)
            + nu_T
            * (
                np.gradient(self.du, y, axis=0) * np.gradient(self.u, y, axis=0)
                + np.gradient(self.du, z, axis=1) * np.gradient(self.u, z, axis=1)
            )
            + self.C_k1  # pull out of gradient as C_k1 is constant
            * (
                np.gradient(nu_T * np.gradient(self.dk, y, axis=0), y, axis=0)
                + np.gradient(nu_T * np.gradient(self.dk, z, axis=1), z, axis=1)
            )
            # need np.clip for the sqrt here
            - self.C_k2 * (np.clip(self.dk, 0, None) ** (3 / 2) / lmix)
        ) / self.u
        return dkdx

    def unpack_inputs(self, state):
        """
        Returns:
        - du: wake deficit
        - dk: k_wake field
        """
        if state.ndim == 1:
            self.need_reshape = True  # we will need this in packing the outputs
            state = state.reshape(self.curledwake.shape[1:] + (2,))

        self.dk = state[..., 1]
        return state[..., 0], state[..., 1]

    def pack_outputs(self, dudx, dkdx):
        """
        Pack outputs for the forward marching dudx, dkdx
        """
        ret = np.stack([dudx, dkdx], axis=-1)
        if self.need_reshape:
            return ret.flatten()
        else:
            return ret

    def unpack_outputs(self, ret):
        """
        Unpack the output result of the forward marching
        """
        return ret[..., 0], ret[..., 1]


def check_state_bounds(state, thresh=1e-4):
    """
    Check values of 2D array `state` at the boundaries to see
    if a domain expansion is needed.
    """
    max_y = np.max(abs(state[[0, -1], :]), axis=1)
    max_z = np.max(abs(state[:, [0, -1]]), axis=0)

    expand_y = max_y > thresh
    expand_z = max_z > thresh

    return expand_y, expand_z


def ic_stencil(y, z, yt, zt, smooth_fact=1, ay=0.5, az=None) -> np.ndarray:
    """
    Stencil for turbine initial condition. This is a 2D Gaussian kernel that is
    convolved with an indicator function.

    Parameters:
    - y: y-coordinates.
    - z: z-coordinates.
    - smooth_fact: Smoothing factor for the initial condition stencil.
    - ay: Width of the stencil in the y-direction (default: 0.5).
    - az: Width of the stencil in the z-direction (default: ay).
    """
    az = ay if az is None else az

    yG, zG = np.meshgrid(y, z, indexing="ij")
    dy = y[1] - y[0]
    dz = z[1] - z[0]  # assume these are equally spaced axes
    kernel_y = np.arange(-10, 11)[:, None] * dy
    kernel_z = np.arange(-10, 11)[None, :] * dz

    # turb = ((yG - yt) ** 2 + (zG - zt) ** 2) < R**2
    turb = (((yG - yt) / ay) ** 2 + ((zG - zt) / az) ** 2) < 1.0
    gauss = np.exp(
        -(kernel_y**2 + kernel_z**2) / (np.sqrt(dy * dz) * smooth_fact) ** 2 / 2
    )
    gauss /= np.sum(gauss)  # make sure this is normalized to 1
    return convolve2d(turb, gauss, "same")


def get_wake_bounds_y(du, thresh=0.05, relative=True):
    """
    Parse wake bounds from the 2D du field, returns indices for
    all crossings of threshold `thresh` from the wake profile in y.

    Parameters:
    - du: 2D array of delta_u
    - thresh: threshold for wake bounds
    - relative: whether to use a threshold relative to max(abs(du))

    Returns:
    - ycross: array of y-crossings, arranged as [2 x N] array of (lower, upper) index pairs
    """

    du_y = np.max(abs(du), axis=1)

    _thresh = thresh * np.max(du_y) if relative else thresh

    ycross_lower = np.where((du_y[:-1] < _thresh) & (du_y[1:] >= _thresh))[0]
    ycross_upper = np.where((du_y[:-1] > _thresh) & (du_y[1:] <= _thresh))[0]
    ycross = np.vstack([ycross_lower, ycross_upper])

    return ycross


def get_wake_bounds_z(du, thresh=0.05, relative=True):
    """
    Parse wake bounds from the 2D du field, returns indices for
    all crossings of threshold `thresh` from the wake profile in z.

    Parameters:
    - du: 2D array of delta_u
    - thresh: threshold for wake bounds
    - relative: whether to use a threshold relative to max(abs(du))

    Returns:
    - zcross: array of z-crossings, arranged as [2 x N] array of (lower, upper) index pairs
    """

    du_z = np.max(abs(du), axis=0)

    _thresh = thresh * np.max(abs(du)) if relative else thresh

    zcross_lower = np.where((du_z[:-1] < _thresh) & (du_z[1:] >= _thresh))[0]
    zcross_upper = np.where((du_z[:-1] > _thresh) & (du_z[1:] <= _thresh))[0]
    zcross = np.vstack([zcross_lower, zcross_upper])

    return zcross


def interpolate_lmix(du, y, method="nearest", fill_value=1.0, max_value=None, pad=True):
    """
    Interpolates the mixing length scale from the du field.

    Parameters:
    - du: 2D array of delta_u
    - y: y-coordinates

    Returns:
    - lmix: 1D array of mixing length scale
    """
    if np.any(np.isnan(du)):
        raise ValueError("du contains NaN values")

    if pad:
        # zero-pad wake_bnds to ensure it goes to zero on both sides
        wake_bnds = get_wake_bounds_y(np.pad(du, (1, 1), mode="constant"))
        y_pad = np.pad(y, (1, 1), mode="edge")
        y_bounds = y_pad[wake_bnds]
    else:
        wake_bnds = get_wake_bounds_y(du)
        y_bounds = y[wake_bnds]

    if y_bounds.size == 0:
        return np.full_like(y, fill_value)

    y_mean = np.mean(y_bounds, axis=0)
    y_width = np.diff(y_bounds, axis=0).flatten()
    f = interp1d(
        y_mean,
        y_width,
        kind=method,
        fill_value=(y_width[0], y_width[-1]),
        bounds_error=False,
    )
    if max_value is None:
        return f(y)
    else:
        return np.clip(f(y), None, max_value)


def get_heaviside(x, yax, turbines, default_x0=1):
    """
    Computes the heaviside function, which is 1 in the far-wake and 0 in the near-wake.
    """
    ret = np.zeros_like(yax)
    for t in turbines:
        try:
            x0 = t.rotor_solution.extra.x0
            if x0 == np.inf:
                x0 = default_x0
        except AttributeError:
            x0 = default_x0

        if x >= t.xt and x < t.xt + x0:
            # yids = (yax >= (t.yt - t.D/2)) & (yax <= (t.yt + t.D/2))
            # ret[yids] = 1
            ret += np.exp(-((yax - t.yt) ** 2) / 2 / (t.D / 2) ** 2)

    ret = np.clip(ret, 0, 1)

    return 1 - ret


if __name__ == "__main__":
    pass
