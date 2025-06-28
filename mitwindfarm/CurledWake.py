"""
Curled wake model solver in MITWindfarm.
(Now in a separate file)

Kirby Heck
2025 June 6
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Union
from warnings import warn

from numpy.typing import ArrayLike
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import interpn, make_interp_spline

from mitwindfarm.Windfield import Windfield
from mitwindfarm.Rotor import RotorSolution
from mitwindfarm.utils.integrate import (
    Integrator,
    IntegrationException,
    DomainExpansionRequest,
)
from mitwindfarm.utils.differentiate import second_der


#  ██████ ██    ██ ██████  ██      ███████ ██████      ██     ██  █████  ██   ██ ███████
# ██      ██    ██ ██   ██ ██      ██      ██   ██     ██     ██ ██   ██ ██  ██  ██
# ██      ██    ██ ██████  ██      █████   ██   ██     ██  █  ██ ███████ █████   █████
# ██      ██    ██ ██   ██ ██      ██      ██   ██     ██ ███ ██ ██   ██ ██  ██  ██
#  ██████  ██████  ██   ██ ███████ ███████ ██████       ███ ███  ██   ██ ██   ██ ███████


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
        u_model: str = "default",
        v_model: Literal["analytical", "decay"] = "default",
        w_model: Literal["analytical", "decay"] = "default",
        k_model: Literal["const", "k-l"] = "const",
        u_kwargs: dict = None,
        v_kwargs: dict = None,
        w_kwargs: dict = None,
        k_kwargs: dict = None,
        ic_method: Literal["du", "fx"] = "du",
        bottom_wall_z: Union[float, bool] = None,
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
        - ic_method: Method for initial condition stamping (default: "du").
            NOTE: "fx" is experimental and only solves for EF marching.
        - u_model: Model for the u-velocity field (default: "default").
        - v_model: Model for the v-velocity field (default: "analytical").
        - w_model: Model for the w-velocity field (default: "analytical").
        - k_model: Turbulence model to use (default: "k-l").
        - u_kwargs: Additional arguments for the u-velocity model (default: None).
        - v_kwargs: Additional arguments for the v-velocity model (default: None).
        - w_kwargs: Additional arguments for the w-velocity model (default: None).
        - k_kwargs: Additional arguments for the turbulence model (default: None).
        - bottom_wall_z: If True, imposes a wall condition at the given value (default: False).
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

        if "scipy" not in self.ivp_name:
            self.ivp_kwargs.setdefault("dt", self.dx)

        self.ybuff = ybuff
        self.zbuff = zbuff

        self.extra_fx = None
        self.ic_method = ic_method  # "fx" DOES NOT WORK - ONLY USE "du"

        self.clip_u = clip_u
        self.use_r4 = use_r4
        self.auto_expand = auto_expand

        # ============ field evolution modules ============
        u_kwargs = dict() if u_kwargs is None else u_kwargs
        v_kwargs = dict() if v_kwargs is None else v_kwargs
        w_kwargs = dict() if w_kwargs is None else w_kwargs
        k_kwargs = dict() if k_kwargs is None else k_kwargs
        # Initialize the modules for u, v, w, and k
        self.modules = dict(
            du=CurledUModel.get_model(u_model, curledwake=self, **u_kwargs),
            dv=CurledVModel.get_model(v_model, curledwake=self, **v_kwargs),
            dw=CurledWModel.get_model(w_model, curledwake=self, **w_kwargs),
            dk=CurledTurbulenceModel.get_model(
                k_model,
                curledwake=self,
                **k_kwargs,
            ),
        )
        self.fields_to_integrate = [k for k, v in self.modules.items() if v.march_field]
        self.fields_other = [k for k, v in self.modules.items() if not v.march_field]

        # The grid will get initialized later in check_grid_init()
        self.grid = None  # list of [x, y, z] axes
        self.bottom_wall_z = -np.inf if bottom_wall_z is None else bottom_wall_z

        self.smooth_fact = smooth_fact  # smoothing factor for the IC stencil
        self.turbines = []

        self.verbose = verbose

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        self.march_to(x=x, y=y, z=z)  # check that the forward marching is sufficient

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        x, y, z = np.broadcast_arrays(x, y, z)

        wsp_base = self.base_windfield.wsp(x, y, z)
        wsp_wakes = interpn(
            (self.x, self.y, self.z),
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
            (self.x, self.y, self.z),
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
            self.y,
            self.z,
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

        # NOTE: rotor.Ct differs from Shapiro et al. (2018) definition - includes cos^2(yaw) already
        Gamma_0 = 0.5 * D * rotor.REWS * rotor.Ct * np.sin(rotor.yaw)

        v, w = compute_vortex_field(
            self.y,
            self.z,
            yt=yt,
            zt=zt,
            Gamma_0=Gamma_0,
            D=D,
            sigma_vortex=self.sigma_vortex,
            N_vortex=self.N_vortex,
        )
        if self.bottom_wall_z > -np.inf:
            # symmetry vortices (negative in sign, centered around zt_ghost)
            zt_ghost = (
                -zt - self.bottom_wall_z * 2
            )  # zt_ghost < 0; if bottom_wall_z = 0, then this is -zt
            vghost, wghost = compute_vortex_field(
                self.y,
                self.z,
                yt=yt,
                zt=zt_ghost,
                Gamma_0=Gamma_0,  # mirror the circulation also
                D=D,
                sigma_vortex=self.sigma_vortex,
                N_vortex=self.N_vortex,
            )
            v += vghost
            w += wghost

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
        ypad, zpad = (0, 0), (0, 0)
        if y is not None:
            y = np.atleast_1d(y)
            ymin = np.min(y) - self.ybuff * add_buffers
            ymax = np.max(y) + self.ybuff * add_buffers
            ypad_lower = np.arange(self.y[0] - self.dy, ymin - self.dy, -self.dy)[::-1]
            ypad_upper = np.arange(self.y[-1] + self.dy, ymax + self.dy, self.dy)
            # update y-grid
            self.grid[1] = np.concatenate([ypad_lower, self.y, ypad_upper])
            ypad = (len(ypad_lower), len(ypad_upper))

        if z is not None:
            z = np.atleast_1d(z)
            zmin = np.max(
                [np.min(z) - self.zbuff * add_buffers, self.bottom_wall_z - self.dz]
            )
            zmax = np.max(z) + self.zbuff * add_buffers
            zpad_lower = np.arange(self.z[0] - self.dz, zmin - self.dz, -self.dz)[::-1]
            zpad_upper = np.arange(self.z[-1] + self.dz, zmax + self.dz, self.dz)
            # update z-grid
            self.grid[2] = np.concatenate([zpad_lower, self.z, zpad_upper])
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
                np.arange(-self.ybuff + y, self.ybuff + self.dy + y, self.dy),
                np.arange(  # impose wall condition?
                    np.max([self.bottom_wall_z - self.dz, -self.zbuff + z]),
                    self.zbuff + z + self.dz,
                    self.dz,
                ),
            ]

            self.du = np.zeros(self.shape)
            self.dv = np.zeros(self.shape)
            self.dw = np.zeros(self.shape)
            self.dk = np.zeros(self.shape)
            self.extra_fx = np.zeros(self.shape[1:])

    def _march(self, xmax) -> None:
        """
        Forward marches the wake field solution up to `xmax`

        Returns
        - None (updates grid and self.du, self.dv, self.dw, self.dk in place)
        """
        if xmax <= np.max(self.x):
            return  # nothing to compute!

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

            # ========= assemble variables and fields =========
            if np.any(np.isnan(_state)):
                raise IntegrationException(f"nan value encountered at x={x:.3f}")
            vars = self._unpack_inputs(x, _state)  # computes all of the deficit fields
            if self.auto_expand:
                self._check_yz_bounds(x, vars)  # may raise DomainExpansionRequest

            # Full velocity fields for advection:
            wsp = self.base_windfield.wsp(x, self.y[:, None], self.z[None, :])
            wdir = self.base_windfield.wdir(x, self.y[:, None], self.z[None, :])
            # compute k_base: assume TI = sqrt(2/3 k)/U
            kb = (
                (self.base_windfield.TI(x, self.y[:, None], self.z[None, :]) * wsp) ** 2
                * 3
                / 2
            )
            vars["u"] = vars["du"] + wsp * np.cos(wdir)
            vars["v"] = vars["dv"] + wsp * np.sin(wdir)
            vars["w"] = vars["dw"] + 0
            vars["k"] = vars["dk"] + kb

            if (self.clip_u > 0) and np.any(vars["u"] < self.clip_u):
                vars["u"] = np.clip(vars["u"], self.clip_u, None)

            self.shared_flow_data = vars  # store this in a global variable
            self._impose_wall_conditions()  # impose wall conditions if necessary
            return self._return_derivatives(x)

        ic = self._pack_inputs()  # pack the initial conditions from the modules

        try:
            x, ret = self.integrator(_step, [self.x.max(), xmax], ic, **self.ivp_kwargs)
        except IntegrationException as e:
            x = e.partial_t
            ret = e.partial_u
            if self.verbose:
                print(f"Exiting integration at x={max(x)}:\n\t", e)
        except DomainExpansionRequest as e:
            x, ret = e.partial_t, e.partial_u
            ybnd, zbnd = e.expand_y, e.expand_z
            # every time we get here, expand the expansion...
            if np.any(ybnd):
                self.ybuff += 1
            if np.any(zbnd):
                self.zbuff += 1

        if len(x) > 1:
            # append and concatenate progress
            self._finalize_outputs(xnew=x[1:], ret=ret[1:])
            self.grid[0] = np.concatenate([self.x, x[1:]])

        # if we hit a DomainExpansionRequest, need to expand the grid and continue integrating
        if np.any([ybnd, zbnd]):
            if self.verbose:
                print(f"Expanding grid at x={np.max(x):.2f} in y={ybnd} and z={zbnd}")

            self.adjust_grid_bounds(
                y=[self.y[0] - ybnd[0] * self.ybuff, self.y[-1] + ybnd[1] * self.ybuff],
                z=[self.z[0] - zbnd[0] * self.zbuff, self.z[-1] + zbnd[1] * self.zbuff],
                add_buffers=False,
            )
            self._march(xmax=xmax)  # recursive call to continue marching

    def _pack_inputs(self):
        """Returns a flattened initial condition array for marched variables"""
        ic = []
        for name in self.fields_to_integrate:
            ic.append(self.modules[name].field[-1, ...])
        ic = np.stack(ic, axis=-1)
        return ic.flatten()

    def _unpack_inputs(self, x, state: ArrayLike) -> ArrayLike:
        """Returns a dictionary of variables from the flattened state reshaped to (ny, nz)"""
        reshape = state.reshape(self.shape[1:] + (len(self.fields_to_integrate),))
        ret = {name: reshape[..., i] for i, name in enumerate(self.fields_to_integrate)}
        # compute the fields that aren't in the inputs
        for name in self.fields_other:
            ret[name] = self.modules[name].get_field_x(x)
        return ret

    def _impose_wall_conditions(self):
        """
        Impose wall boundary conditions on fields:
        du/dz = 0; dv/dz = 0; dk/dz = 0; w=0 at the wall.
        """
        if self.bottom_wall_z > -np.inf:
            # update boundary conditions with symmetry and anti-symmetry conditions
            zid = np.argmin(np.abs(self.z - self.bottom_wall_z))  # zid at the wall
            ghost_id = zid - 1  # ghost point below the wall
            mirror_id = zid + 1  # mirror point above the wall

            # impose wall conditions on the fields
            flow = self.shared_flow_data
            for key in self.fields_to_integrate:
                flow[key][..., ghost_id] = flow[key][..., mirror_id]
                # flow[key][..., :ghost_id] = 0

            # flow["dw"][..., :zid] = 0
            # flow["w"][..., :zid] = 0

    def _return_derivatives(self, x) -> ArrayLike:
        """Returns a flattened array of the outputs from _step"""
        ret = []
        for name in self.fields_to_integrate:
            ret.append(self.modules[name].ddx(x))
        return np.stack(ret, axis=-1).flatten()

    def _finalize_outputs(self, xnew: ArrayLike, ret: ArrayLike):
        """
        Reshapes the flattened outputs from _step to the shape of the grid.
        Additionally, computes the analytical fields which are not marched in space.

        Returns
        -------
        None
        """
        shape = (len(xnew), *self.shape[1:], len(self.fields_to_integrate))
        reshape = ret.reshape(shape)
        # concatenate fields along x:
        for i, name in enumerate(self.fields_to_integrate):
            m = self.modules[name]
            m.field = np.concatenate([m.field, reshape[..., i]], axis=0)

        # now compute the analytical fields which are not marched in space
        for name in self.fields_other:
            m = self.modules[name]
            m.field = np.concatenate([m.field, m.get_field_x(xnew)], axis=0)

    def _check_yz_bounds(self, x, vars):
        """
        Check all integration variables for whether to expand the domain
        in y and/or z.
        """
        check_yz = []
        for name, m in self.modules.items():
            if m.march_field and m.check_yz:
                check_yz.append(check_state_bounds(vars[name], thresh=m.bound_thresh))

        ybnd, zbnd = np.max(check_yz, axis=0)
        if zbnd[0] and np.min(self.z) < self.bottom_wall_z:
            zbnd[0] = False  # don't expand if the wall condition is already imposed

        if np.any([ybnd, zbnd]):
            # if any of the checks fail, we need to expand the domain along those dimensions
            raise DomainExpansionRequest(
                f"Expanding domain at {x=:.2f}", expand_y=ybnd, expand_z=zbnd
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns the shape of the grid.
        """
        return (len(self.x), len(self.y), len(self.z))

    @property
    def x(self) -> ArrayLike:
        return self.grid[0]

    @property
    def y(self) -> ArrayLike:
        return self.grid[1]

    @property
    def z(self) -> ArrayLike:
        return self.grid[2]

    @property
    def du(self):
        return self.modules["du"].field  # solved du-field

    @property
    def dv(self):
        return self.modules["dv"].field  # solved dv-field

    @property
    def dw(self):
        return self.modules["dw"].field  # solved dw-field

    @property
    def dk(self):
        return self.modules["dk"].field  # solved k_wake field

    @du.setter
    def du(self, value):
        self.modules["du"].field = value

    @dv.setter
    def dv(self, value):
        self.modules["dv"].field = value

    @dw.setter
    def dw(self, value):
        self.modules["dw"].field = value

    @dk.setter
    def dk(self, value):
        self.modules["dk"].field = value


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


# ███████ ██ ███████ ██      ██████       ██████ ██       █████  ███████ ███████
# ██      ██ ██      ██      ██   ██     ██      ██      ██   ██ ██      ██
# █████   ██ █████   ██      ██   ██     ██      ██      ███████ ███████ ███████
# ██      ██ ██      ██      ██   ██     ██      ██      ██   ██      ██      ██
# ██      ██ ███████ ███████ ██████       ██████ ███████ ██   ██ ███████ ███████


class Field(ABC):
    """
    Base class for a field variable in the curled wake model. This
    field may evolve in space, or it may have an analytical solution, or
    it may be constant.
    """

    def __init__(self, curledwake: CurledWakeWindfield):
        """
        Initializes the field with a link to the curled wake solver object.
        """
        self.curledwake = curledwake
        self.field = None  # this will be initialized in a separate function
        self.march_field = False  # whether this field evolves in space
        self.check_yz = False
        self.bound_thresh = None

    def ddx(self):
        """Returns the derivative d(field)/dx if self.march_field is True"""
        if self.march_field:
            raise NotImplementedError()
        else:
            return None

    def get_field_x(self, x):
        """
        Computes the field at location x, or returns the field if this
        variable is marched in space. By default, this assumes a static
        field and returns the nearest value.
        """
        if np.isscalar(x):
            xid = np.argmin(np.abs(self.x - x))  # could also interpolate
            return self.field[xid, ...]
        else:
            ret = []
            for _x in x:
                ret.append(self.get_field_x(_x))
            return np.stack(ret, axis=0)

    @property
    def x(self):
        return self.curledwake.grid[0]


class CurledUModel(Field):
    """
    Abstract class for the u-velocity field in the curled wake model.
    """

    _registry = {}
    name: str  # fill this in for each model

    def __init__(self, curledwake: CurledWakeWindfield):
        self.curledwake = curledwake  # link to the curled wake solver object

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called when a subclass is created.
        It registers the subclass in the u-model registry.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls._registry[cls.name] = cls
        else:
            raise ValueError(f"Subclass {cls.__name__} must define a 'name' attribute.")

    @classmethod
    def get_model(cls, name: str, *args, **kwargs):
        """
        Factory method to get an instance of a u-model.
        """
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(
                f"Unknown u-model: '{name}'. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return model_class(*args, **kwargs)


class DefaultUModel(CurledUModel):
    """
    Default u-model for the curled wake model. This is a constant model
    that does not evolve in space.
    """

    name = "default"

    def __init__(self, curledwake: CurledWakeWindfield, thresh=1e-4):
        super().__init__(curledwake=curledwake)
        self.march_field = True
        self.check_yz = True
        self.bound_thresh = thresh  # abs threshold for checking bounds

    def ddx(self, x):
        """Computes d(du)/dx at location x"""
        _vars = self.curledwake.shared_flow_data
        du = _vars["du"]  # get the u-velocity field
        u = _vars["u"]
        v = _vars["v"]
        w = _vars["w"]
        nu_T = _vars.get("nu_T", self.curledwake.modules["dk"].nu_T(x))
        _vars["nu_T"] = nu_T  # update nu_T in shared flow data
        y, z = self.curledwake.grid[1:]
        # ============== du/dx computation ==============
        dudy = np.gradient(du, y, axis=0)
        dudz = np.gradient(du, z, axis=1)
        d2udy2 = np.gradient(nu_T * dudy, y, axis=0)
        d2udz2 = np.gradient(nu_T * dudz, z, axis=1)
        dudx = (-v * dudy - w * dudz + d2udy2 + d2udz2) / u
        # d2udy2 = second_der(du, self.curledwake.dy, axis=0)
        # d2udz2 = second_der(du, self.curledwake.dz, axis=1)
        # dudx = (-v * dudy - w * dudz + nu_T * (d2udy2 + d2udz2)) / u
        # if x > 9 and self.curledwake.bottom_wall_z > -np.inf:
        #     import matplotlib.pyplot as plt
        #     plt.pcolormesh(y, z, du.T); plt.gca().set_aspect(1)
        #     plt.show()

        return dudx


class CurledVModel(Field):
    """
    Abstract class for the v-velocity field in the curled wake model.
    This is a constant model that does not evolve in space.
    """

    _registry = {}
    name: str  # fill this in for each model

    def __init__(self, curledwake: CurledWakeWindfield):
        self.curledwake = curledwake  # link to the curled wake solver object

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls._registry[cls.name] = cls
        else:
            raise ValueError(f"Subclass {cls.__name__} must define a 'name' attribute.")

    @classmethod
    def get_model(cls, name: str, *args, **kwargs):
        """
        Factory method to get an instance of a v-model.
        """
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(
                f"Unknown v-model: '{name}'. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return model_class(*args, **kwargs)


class DefaultVModel(CurledVModel):
    """
    Default v-model for the curled wake model. This is a constant model
    that does not evolve in space.
    """

    name = "default"

    def __init__(self, curledwake: CurledWakeWindfield):
        super().__init__(curledwake=curledwake)
        self.march_field = False  # v does not evolve in space


class MarchedVMdodel(CurledVModel):
    """
    Forward-marched v-model which includes ABL effects, turbulence, and
    Coriolis forces.
    """

    name = "marched"

    def __init__(self, curledwake, Ro=1e10, check_zy=False):
        super().__init__(curledwake)
        self.march_field = True
        self.check_yz = check_zy
        self.Ro = Ro  # Rossby number with vertical rotation effects Ro = u_h/(fc * D)

    def ddx(self, x):
        """Computes d(du)/dx at location x"""
        _vars = self.curledwake.shared_flow_data
        du, dv, u, v, w = [_vars[key] for key in ["du", "dv", "u", "v", "w"]]
        nu_T = _vars.get("nu_T", self.curledwake.modules["dk"].nu_T(x))
        _vars["nu_T"] = nu_T  # update nu_T in shared flow data
        y, z = self.curledwake.grid[1:]
        # ============== du/dx computation ==============
        dvdy = 0  # np.gradient(dv, y, axis=0)
        dvdz = 0  # np.gradient(dv, z, axis=1)
        d2vy = 0  # np.gradient(nu_T * dvdy, y, axis=0)
        d2vz = 0  # np.gradient(nu_T * dvdz, z, axis=1)
        coriolis = -1 / self.Ro * (du)
        dvdx = (-v * dvdy - w * dvdz + coriolis + d2vy + d2vz) / u
        return dvdx


class CurledWModel(Field):
    """
    Abstract class for the w-velocity field in the curled wake model.
    This is a constant model that does not evolve in space.
    """

    _registry = {}
    name: str  # fill this in for each model

    def __init__(self, curledwake: CurledWakeWindfield):
        self.curledwake = curledwake  # link to the curled wake solver object

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls._registry[cls.name] = cls
        else:
            raise ValueError(f"Subclass {cls.__name__} must define a 'name' attribute.")

    @classmethod
    def get_model(cls, name: str, *args, **kwargs):
        """
        Factory method to get an instance of a w-model.
        """
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(
                f"Unknown w-model: '{name}'. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return model_class(*args, **kwargs)


class DefaultWModel(CurledWModel):
    """
    Default w-model for the curled wake model. This is a constant model
    that does not evolve in space.
    """

    name = "default"

    def __init__(self, curledwake: CurledWakeWindfield):
        super().__init__(curledwake=curledwake)
        self.march_field = False  # v does not evolve in space


# ████████ ██    ██ ██████  ██████  ██    ██ ██      ███████ ███    ██  ██████ ███████     ███    ███  ██████  ██████  ███████ ██      ███████
#    ██    ██    ██ ██   ██ ██   ██ ██    ██ ██      ██      ████   ██ ██      ██          ████  ████ ██    ██ ██   ██ ██      ██      ██
#    ██    ██    ██ ██████  ██████  ██    ██ ██      █████   ██ ██  ██ ██      █████       ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
#    ██    ██    ██ ██   ██ ██   ██ ██    ██ ██      ██      ██  ██ ██ ██      ██          ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
#    ██     ██████  ██   ██ ██████   ██████  ███████ ███████ ██   ████  ██████ ███████     ██      ██  ██████  ██████  ███████ ███████ ███████


class CurledTurbulenceModel(Field):
    """
    Base class for the turbulence model in the curled wake model.

    Keeps track of all sub-classes with a self-registering factory.
    """

    _registry = {}
    name: str  # fill this in for each model

    def __init__(self, curledwake: CurledWakeWindfield):
        self.curledwake = curledwake  # link to the curled wake solver object
        self.need_reshape = False
        self.march_field = False

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

    def __init__(self, curledwake, C_nu=0.04, C_k1=1, C_k2=1, thresh=1e-6):
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
        self.march_field = True
        self.check_yz = True
        self.bound_thresh = thresh  # abs threshold for checking bounds

    def nu_T(self, x):
        """
        Computes Eq. 6 in Klemmer and Howland (2025)
        """
        vars = self.curledwake.shared_flow_data
        lmix = compute_lmix(vars["du"], self.curledwake.grid[2])
        if np.any(lmix <= 0):
            raise IntegrationException("lmix is non-positive")

        heaviside = get_heaviside(x, self.curledwake.grid[1], self.curledwake.turbines)[
            :, None
        ]
        self.nu_T_cached = self.C_nu * (
            (1 - heaviside) * np.sqrt(np.clip(vars["k"] - vars["dk"], 0, None)) * 1
            + heaviside * np.sqrt(np.clip(vars["k"], 0, None)) * lmix
        )
        return self.nu_T_cached

    def ddx(self, x):
        """
        Computes the dk/dx term for the turbulence model.
        """
        y = self.curledwake.grid[1]
        z = self.curledwake.grid[2]
        nu_T = self.nu_T_cached
        vars = self.curledwake.shared_flow_data
        u, v, w, du, dk = [vars[name] for name in ["u", "v", "w", "du", "dk"]]
        lmix = compute_lmix(vars["du"], self.curledwake.grid[2])
        if np.any(lmix <= 0):
            raise IntegrationException("lmix is non-positive")

        # transport equation for k_wake, written in parabolic form:
        dkdx = (
            -v * np.gradient(dk, y, axis=0)
            - w * np.gradient(dk, z, axis=1)
            + nu_T
            * (
                np.gradient(du, y, axis=0) * np.gradient(u, y, axis=0)
                + np.gradient(du, z, axis=1) * np.gradient(u, z, axis=1)
            )
            + self.C_k1  # pull out of gradient as C_k1 is constant
            * (
                np.gradient(nu_T * np.gradient(dk, y, axis=0), y, axis=0)
                + np.gradient(nu_T * np.gradient(dk, z, axis=1), z, axis=1)
            )
            # need np.clip for the sqrt here
            - self.C_k2 * (np.clip(dk, 0, None) ** (3 / 2) / lmix)
        ) / u
        return dkdx


# ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
# █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
# ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████


def compute_vortex_field(y, z, yt, zt, Gamma_0, D=1, sigma_vortex=0.1, N_vortex=10):
    """
    Computes the CVP field from an elliptical distribution of Lamb-Oseen vortices
    spaced evenly along the vertical axis of the rotor disk.

    Parameters:
    - y: y-coordinates (1D array)
    - z: z-coordinates (1D array)
    - yt: y-coordinate of the turbine center
    - zt: z-coordinate of the turbine center
    - Gamma_0: Normalized circulation strength
    - D: Rotor diameter (default: 1)
    - sigma_vortex: Standard deviation of the vortex distribution (default: 0.1)
    - N_vortex: Number of vortices to distribute along the rotor disk (default: 10)

    Returns:
    - v, w: 2D arrays of the velocity field components in the y and z directions
    """
    dz = z[1] - z[0]
    # r-axis: clip edges to prevent singularities
    r_i = np.linspace(-(D - dz) / 2, (D - dz) / 2, N_vortex)
    Gamma_i = Gamma_0 * 4 * r_i / (N_vortex * D**2 * np.sqrt(1 - (2 * r_i / D) ** 2))
    sigma = sigma_vortex * D

    # now we build the main summation, which is 3D (y, z, i)
    yG, zG = np.meshgrid(y, z, indexing="ij")
    yG = yG[..., None]  # expand extra dimension
    zG = zG[..., None]  # expand extra dimension
    rsq = (yG - yt) ** 2 + (zG - zt - r_i[None, None, :]) ** 2  # 3D grid variable
    rsq = np.clip(rsq, 1e-8, None)  # avoid singularities

    # put pieces together:
    exponent = 1 - np.exp(-rsq / sigma**2)
    summation = exponent / (2 * np.pi * rsq) * Gamma_i[None, None, :]

    # sum all vortices along last dim
    v = np.sum(summation * (zG - zt - r_i[None, None, :]), axis=-1)
    w = np.sum(summation * -(yG - yt), axis=-1)

    return v, w


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


def interpolate_lmix(du, y, k=0, fill_value=1.0, max_value=None, pad=True):
    """
    Interpolates the mixing length scale from the du field.

    Parameters:
    - du: 2D array of delta_u
    - y: y-coordinates
    - k: interpolation order (default: 0, nearest neighbor)
    - fill_value: value to fill if no bounds are found (default: 1.0)
    - max_value: maximum value for the mixing length scale (default: None, no limit)
    - pad: whether to pad the du field with zeros (default: True)

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
    f = make_interp_spline(
        y_mean,
        y_width,
        k=k,  # nearest interpolation
    )
    if max_value is None:
        return f(y)
    else:
        return np.clip(f(y), None, max_value)


def compute_lmix(du, z, lmix_min=1, thresh=0.05, relative=True):
    """
    Computes `lmix` from the 2D du field by computing the local wake width (measuring
    the wake height, in z) at each y-location.

    Parameters:
    - du: 2D array of delta_u
    - z: z-coordinates

    Returns:
    - lmix: 2D yz-array of mixing length scale
    """
    if np.any(np.isnan(du)):
        raise ValueError("du contains NaN values")

    du = np.abs(du)
    nz = du.shape[1]
    _thresh = thresh * np.max(du) if relative else thresh
    above_thresh = du > _thresh

    # now we need to find the bounds of where du is above the threshold
    z_below = z[np.argmax(above_thresh, axis=1)]
    z_above = z[nz - 1 - np.argmax(np.flip(above_thresh, axis=1), axis=1)]
    lmix = (z_above - z_below) * np.any(above_thresh, axis=1)

    if lmix_min is not None:
        lmix = np.clip(lmix, lmix_min, None)
    return lmix[:, None]


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
