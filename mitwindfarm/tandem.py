"""
TANDEM (Turbulence ANd DEficit Momentum) model closures for the curled wake solver.

Turbulence closures, vortex-decay wake curling models, and near-wake helper
functions developed for the TANDEM wake model. Migrated in from the
`kl_model` analysis repository's `mitwf_extras.py`.

Kirby Heck
2025-2026
"""

import numpy as np

from scipy.interpolate import interpn, make_interp_spline

from .CurledWake import (
    DefaultUModel,
    CurledTurbulenceModel_kl,
    CurledWakeWindfield,
    TurbineProperties,
    CurledVModel,
    CurledWModel,
    Field,
    vortex_field_from_turbine,
)
from .utils.integrate import IntegrationException


def v_adv_upwind(y, u, v):
    """
    Computes upwind derivatives of u(y, z) and multiplies by v to get the advection term.
    Where v>0, uses backward differences; where v<0, uses forward differences.
    """
    dy = np.gradient(y)[:, None]

    # Forward difference (one-sided, for v < 0)
    du_dy_fwd = np.zeros_like(u)
    du_dy_fwd[:-2, :] = (-3/2 * u[:-2, :] + 2 * u[1:-1, :] - 1/2 * u[2:, :]) / dy[:-2]
    du_dy_fwd[-2:, :] = np.gradient(u[-3:, :], y[-3:], axis=0)[-2:, :]

    # Backward difference (one-sided, for v > 0)
    du_dy_bwd = np.zeros_like(u)
    du_dy_bwd[2:, :] = (3/2 * u[2:, :] - 2 * u[1:-1, :] + 1/2 * u[:-2, :]) / dy[2:]
    du_dy_bwd[:2, :] = np.gradient(u[:3, :], y[:3], axis=0)[:2, :]

    du_dy = np.where(v > 0, du_dy_bwd, du_dy_fwd)

    return v * du_dy


def w_adv_upwind(z, u, w):
    """
    Computes upwind derivatives of u(y, z) and multiplies by w to get the advection term.
    Where w>0, uses backward differences; where w<0, uses forward differences.

    CURRENTLY OVERRIDDEN BY NP.GRADIENT
    """
    return w * np.gradient(u, z, axis=1)


def dudy_periodic(y, u):
    """Computes central differences of u(y, z) in the y-direction with periodic BCs."""
    dy = np.gradient(y)[:, None]
    du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)
    return du_dy


def dudz_periodic(z, u):
    """Computes central differences of u(y, z) in the z-direction with periodic BCs."""
    dz = np.gradient(z)[None, :]
    du_dz = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dz)
    return du_dz


def dudy_interior(y, u):
    """Computes du/dy using central differences in the interior only and leaves du/dy = 0 on boundaries."""
    dudy = np.zeros_like(u)
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (y[2:, None] - y[:-2, None])
    return dudy


def dudz_interior(z, u):
    """Computes du/dz using central differences in the interior only and leaves du/dz = 0 on boundaries."""
    dudz = np.zeros_like(u)
    dudz[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (z[None, 2:] - z[None, :-2])
    return dudz


def softmin(arr, axis=0, Lambda=None):
    """Returns the "soft" minimization function of broadcastable arguments"""
    arr = np.stack(arr, axis=axis)
    if Lambda is None:
        # improve this based on Appendix A of Lopez-Gomez paper
        Lambda = 0.1 * np.ptp(arr, axis=axis, keepdims=True)

    exp_terms = np.exp(-arr / Lambda)
    sj_terms = exp_terms / np.sum(exp_terms, axis=axis, keepdims=True)
    return np.sum(arr * sj_terms, axis=axis)


def pmin(arr, axis=0, p=1):
    """Returns the pnorm-blending interpolation, which is the harmonic mean if p=1"""
    arr = np.stack(arr, axis=axis)
    return (np.sum(arr**(-p), axis=axis))**(-1/p)


def phi_m(xi):
    """Monin-Obukhov stability correction function for momentum"""
    ret = np.zeros_like(xi)
    ret[xi >= 0] = 1 + 5 * xi[xi >= 0]
    ret[xi < 0] = (1 - 16 * xi[xi < 0])**-0.25
    return ret


class Upwind_UModel(DefaultUModel):
    """Default u-model with upwind differences for advective terms"""

    name = "upwind"

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
        vdudy = v_adv_upwind(y, du, v)
        wdudz = w_adv_upwind(z, du, w)

        dudy = dudy_interior(y, du)
        dudz = dudz_interior(z, du)
        d2udy2 = dudy_interior(y, nu_T * dudy)
        d2udz2 = dudz_interior(z, nu_T * dudz)

        dudx = (-vdudy - wdudz + d2udy2 + d2udz2) / u
        return dudx


class Conservation_UModel(DefaultUModel):
    """
    u-model with output normalized such that momentum deficit integral
    is conserved (u*dudx integrated over y,z equals thrust deficit).
    """

    name = "cons"

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
        vdudy = v_adv_upwind(y, du, v)
        wdudz = w_adv_upwind(z, du, w)

        dudy = np.gradient(du, y, axis=0)
        dudz = np.gradient(du, z, axis=1)
        d2udy2 = np.gradient(nu_T * dudy, y, axis=0)
        d2udz2 = np.gradient(nu_T * dudz, z, axis=1)
        ddudx_raw = (-vdudy - wdudz + d2udy2 + d2udz2) / u

        # correction term
        zids = z >= self.curledwake.bottom_wall_z  # exclude ghost point in computation
        _z = z[zids]
        weight = (np.abs(du) > 0.01) * u  # different wake indicator function
        leakage = np.trapezoid(np.trapezoid(((u + du) * ddudx_raw)[:, zids], _z, axis=1), y, axis=0)  # double integral over y and z
        correction_norm = np.trapezoid(np.trapezoid((weight * (u + du))[:, zids], _z, axis=1), y, axis=0)
        lam = -leakage / correction_norm
        fx = lam * weight

        dudx_final = ddudx_raw + fx
        return dudx_final


class LES_VModel(CurledVModel):
    """Uses an LES field instead of the curled wake solution for delta v."""

    name = "les"

    def __init__(self, curledwake, les_dv):
        super().__init__(curledwake)
        self.les_dv = les_dv.to_numpy()
        self._x = les_dv.grid.x.to_numpy()
        self._y = les_dv.grid.y.to_numpy()
        self._z = les_dv.grid.z.to_numpy()

    def get_field_x(self, x):
        """Returns the v-velocity field at location x."""
        if np.isscalar(x):
            yy, zz = np.meshgrid(self.curledwake.grid[1], self.curledwake.grid[2], indexing="ij")
            xid = np.argmin(np.abs(self._x - x))  # nearest interpolation
            v_les_slice = self.les_dv[xid, ...]
            # interpolate in scipy for speed
            v_interp = interpn(
                (self._y, self._z),
                v_les_slice,
                np.array([yy.ravel(), zz.ravel()]).T,
                bounds_error=False,
                fill_value=0.0
            )
            return v_interp.reshape(yy.shape)
        else:
            ret = []
            for _x in x:
                ret.append(self.get_field_x(_x))
            return np.stack(ret, axis=0)


class Decay_VModel(CurledVModel):
    """
    Wake curling model that includes vortex decay into the v-velocity field.

    If the TKE model solves a marched equation for dk, then the decay depends
    on nu_T computed from the TKE model. This coupling can be shut off in __init__
    with the `couple` keyword. Without the coupling, equations from Shapiro
    et al. JFM Rapids (2020) are used to estimate the vorticity decay, using
    sqrt(tke) as the velocity scale rather than u_star (which is not known).
    """

    name = "decay"

    def __init__(self, curledwake, couple=None, static_vortices=True, recompute_threshold=0.2):
        """
        Initialize the decay module

        Parameters
        ----------
        curledwake : CurledWakeWindfield
            The curled wake wind field to which this v-model is applied.
        couple : bool, optional
            Couples nu_T from the TKE model with the vortex decay model.
            By default, this is True if the k_model marches dk, and False otherwise.
        static_vortices : bool, optional
            Ignores the mutual inductance of vortices if true. Default True.
        recompute_threshold : float, optional
            Threshold for relative change in sigma to trigger recomputation of the v, w fields.
            Default is 0.2 (20%). Increase to save computation; set to 0 to always recompute.
        """
        super().__init__(curledwake)
        self.march_field = False  # v field is not marched; compute analytically from eta
        self.v_field_2d = None  # 2D cache v field
        self.w_field_2d = None  # 2D cache w field
        self.w_field_3d = None  # 3D w field location
        self.w_cache_3d = None
        self._field = None
        self.sigma_last = np.array([])
        self.recompute_threshold = recompute_threshold

        self.static_vortices = static_vortices  # turns off mutual inductance
        if couple is None:  # default is to couple if the k_model marches dk
            couple = curledwake.modules["dk"].march_field

        if couple:
            if curledwake.modules["dk"].march_field:
                curledwake.modules["cvp"] = DecayVortexModel(curledwake)  # placeholder
            else:
                raise NotImplementedError(
                    "Decay_VModel: `couple` requires `k_model` to solve a dk equation"
                )
        self.couple = couple
        self.n_calls = 0

        # set w field also to "decay"
        self.curledwake.w_model = "decay"

    def get_field_x(self, x):
        """
        Computes the field at location x, or returns the field if this
        variable is marched in space. By default, this assumes a static
        field and returns the nearest value.
        """
        if np.isscalar(x):
            if x > self.x.max():
                # during marching: compute on-the-fly
                return self._get_field_x(x)
            else:
                xid = np.argmin(np.abs(self.x - x))  # nearest interpolation
                return self.field[xid, ...]
        else:
            ret = []
            cache_w = []
            for _x in x:
                ret.append(self.get_field_x(_x))
                cache_w.append(self.w_field_2d)  # also save w field

            # save x-range in cache
            self.w_cache_3d = np.stack(cache_w, axis=0)
            return np.stack(ret, axis=0)

    def _get_field_x(self, x):
        """Computes v field from all yawed turbines upstream of `x`"""
        sigma = np.zeros(len(self.curledwake.turbines))
        for k, t in enumerate(self.curledwake.turbines):
            if x < t.xt or (t.rotor_solution.yaw == 0 and t.rotor_solution.tilt == 0):
                continue
            sigma[k] = self.get_sigma(t, k, x)

        # check: if vorticies have evolved significantly, recompute v, w fields
        diff = np.abs(sigma - self.sigma_last) / self.sigma_last

        if np.any(diff > self.recompute_threshold) or self.n_calls == 0:
            # recompute fields if exceeding threshold
            field_v = np.zeros((len(self.curledwake.y), len(self.curledwake.z)))
            field_w = np.zeros_like(field_v)
            cw = self.curledwake
            for t, _sigma in zip(self.curledwake.turbines, sigma):
                self.n_calls += 1
                v, w = vortex_field_from_turbine(t, cw.y, cw.z, cw.N_vortex, cw.bottom_wall_z, _sigma)
                field_v += v
                field_w += w

            self.v_field_2d, self.w_field_2d = field_v, field_w  # cache these
            self.sigma_last = sigma
            return field_v
        else:
            return self.v_field_2d

    def get_sigma(self, turbine, turbine_id, x):
        """Placeholder function for computing vortex decay based on distance from turbine and nu_T"""
        if self.couple:
            if x > self.curledwake.modules["cvp"].x.max():
                # during forward marching: take from shared data
                sigma_sq = self.curledwake.shared_flow_data["cvp"] + self.curledwake.sigma_vortex**2
            else:
                sigma_sq = self.curledwake.modules["cvp"].get_field_x(x) + self.curledwake.sigma_vortex**2
            return np.sqrt(sigma_sq[turbine_id])
        else:
            # this is the analytical form given by Shapiro et al. (2020)
            _x = x - turbine.xt
            k = 0.64 * turbine.rotor_solution.TI  # let this be a * TI instead of Ustar/Uinf
            x0 = turbine.rotor_solution.extra.x0
            sigma = self.curledwake.sigma_vortex + k * np.clip((_x - x0), 1, None)
        return sigma

    def stamp_ic(self, turbine: TurbineProperties):
        """Stamp the initial condition of the rotor solution into the dv field."""
        curl = self.curledwake
        v, w = vortex_field_from_turbine(turbine, curl.y, curl.z, curl.N_vortex, curl.bottom_wall_z, curl.sigma_vortex)
        self.field[-1, ...] += v
        self.w_field_3d[-1, ...] += w  # also stamp w field into cache

        # also append to sigma_last cache
        self.sigma_last = np.append(self.sigma_last, self.curledwake.sigma_vortex)

    @property
    def field(self):
        """
        Store the actual 'field' information in a hidden attribute '_field'
        to allow for custom getter/setter behavior. In this case, we want to
        update the 2D cache whenever the field is set (for example, when a
        domain expansion is triggered)
        """
        return self._field

    @field.setter
    def field(self, value):
        self._field = value
        if value is not None:
            self.v_field_2d = value[-1, ...]


class DecayWModel(CurledWModel):
    """
    Decaying vortex w-model. Pairs with the Decay_VModel class.

    Because dv, dw are computed almost exactly the same way, computing
    dw separately would result in redundant computations. Instead, we
    point the field of this model to the v-module.
    """

    name = "decay"

    def __init__(self, curledwake):
        super().__init__(curledwake)
        self.march_field = False

    def get_field_x(self, x):
        """
        Computes the field at location x, or returns the field if this
        variable is marched in space. By default, this assumes a static
        field and returns the nearest value.
        """
        if np.isscalar(x):
            if x > self.x.max():
                return self.cache_2d
            else:
                xid = np.argmin(np.abs(self.x - x))  # nearest interpolation
                return self.field[xid, ...]
        else:
            return self.cache_3d

    @property
    def cache_2d(self):
        return self.curledwake.modules["dv"].w_field_2d  # from cache

    @property
    def cache_3d(self):
        return self.curledwake.modules["dv"].w_cache_3d  # from cache

    @property
    def field(self):
        # point the field property to the dv module's w_field_3d
        return self.curledwake.modules["dv"].w_field_3d

    @field.setter
    def field(self, value):
        # point the field property to the dv module's w_field_3d
        self.curledwake.modules["dv"].w_field_3d = value
        if value is not None:
            self.curledwake.modules["dv"].w_field_2d = value[-1, ...]  # also update 2D cache


class DummyField(Field):
    """A dummy field that does nothing"""

    def __init__(self, curledwake):
        super().__init__(curledwake)
        self.march_field = False  # field is not marched; compute analytically from eta
        self.field = np.array([0])

    def get_field_x(self, x):
        return np.array([0])


class CurledVortexModel(Field):
    """Abstract class for the vorticity field in the curled wake model."""

    _registry = {}
    name: str  # fill this in for each model

    def __init__(self, curledwake: CurledWakeWindfield):
        super().__init__(curledwake=curledwake)  # link to the curled wake solver object
        self.impose_wall = False
        self.check_yz = False

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called when a subclass is created.
        It registers the subclass in the vorticity model registry.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls._registry[cls.name] = cls
        else:
            raise ValueError(f"Subclass {cls.__name__} must define a 'name' attribute.")

    @classmethod
    def get_model(cls, name: str, *args, **kwargs):
        """Factory method to get an instance of a vorticity model."""
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(
                f"Unknown vorticity model: '{name}'. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return model_class(*args, **kwargs)


class DecayVortexModel(CurledVortexModel):
    """
    Forward marching vortex decay model. Here, we compute the integral in Shapiro
    et al. JFM Rapids (2020) eq. (4.3):

    sigma^2(x) = \\int_{x0}^x nu_T(x') / u(x') dx'

    This is computed numerically from the turbulence model closure nu_T.
    """

    name = "decay"

    def __init__(self, curledwake, static_vortices=True):
        super().__init__(curledwake)
        self.march_field = True
        self.static_vortices = static_vortices
        self.field = np.array([[]])  # Initialize as 1x0 array; will be updated to be [Nx x Nturb] array
        curledwake.shared_flow_data["cvp"] = self.field  # initialize this here to couple with v, w fields

    def stamp_ic(self, turbine):
        """Every time a turbine is added, expand self.field"""
        self.field = np.pad(self.field, ((0, 0), (0, 1)), mode="constant", constant_values=0)

    def ddx(self, x):
        ret = []
        cw = self.curledwake

        for i, t in enumerate(cw.turbines):
            if t.rotor_solution.yaw == 0 and t.rotor_solution.tilt == 0:
                ret.append(0)  # no decay for non-yawed turbines
                continue
            yid = np.argmin(np.abs(cw.y - t.yt))
            zid = np.argmin(np.abs(cw.z - (t.zt + t.D / 2)))
            nu = cw.modules["dk"].nu_T(x)[yid, zid] / cw.shared_flow_data["u"][yid, zid]
            ret.append(nu)

        return np.array(ret)


def get_turbine_yts(x, turbines, thresh=0.2):
    """Return the y-locations of turbines in the given y array."""
    yts = []
    for turb in turbines[::-1]:  # iterate thru backward
        if turb.xt > x:  # only consider turbines upstream of current x location
            continue
        if not np.any(np.abs(yts - turb.yt) < thresh):
            yts.append(turb.yt)
    yts.sort()
    return yts


def get_yid_midpoints(y, yts):
    """Return the y-indices of midpoints between turbine y-locations."""
    midpoints = []
    for i in range(len(yts) - 1):
        yid = np.argmin(np.abs(y - (yts[i] + yts[i+1]) / 2))
        midpoints.append(yid)

    return midpoints


def get_x_nw(x, yax, zax, turbines, use_constant_x0=None):
    """
    Computes the heaviside function, which is 1 in the far-wake and 0 in the near-wake.
    """
    ret = np.zeros((len(yax), len(zax)))
    l_baseflow = np.zeros((len(yax), len(zax)))

    for t in turbines:
        xlocal = x - t.xt
        if xlocal < 0:
            continue

        if use_constant_x0 is not None:
            x0 = use_constant_x0
        else:
            x0 = (
                t.rotor_solution.extra.x0
                if hasattr(t.rotor_solution.extra, "x0")
                else np.inf
            )
            if x0 == np.inf:
                raise ValueError(
                    "Rotor has no `x0` value defined, please provide a value for `use_constant_x0`."
                )

        normfact = (x0 - xlocal) / x0
        if "ic" in t.fields:
            # t.fields is always kept aligned with the current (yax, zax) grid
            # by CurledWakeWindfield.adjust_grid_bounds, so no shape-matching needed here.
            shape = t.fields["ic"]
        else:
            # probably we should change this width...
            shape = np.exp(-((yax[:, None] - t.yt) ** 2 + (zax[None, :] - t.zt) ** 2) / 2 / (t.D / 2) ** 2)

        ret = np.maximum(ret, normfact * shape)
        if "l_baseflow" in t.fields:
            l_baseflow = np.maximum(l_baseflow, (shape > 0.05) * t.fields["l_baseflow"])

    ret = np.clip(ret, 0, 1)

    return 1 - ret, l_baseflow


def _lmix_md(dk, k, shear, dk_scale=None, rel_tol=1e-4):
    """
    Compute MD mixing length from delta k, k, and the shear production field.

    If `dk_scale` is given (the peak dk over the full x-slice this chunk was
    carved from), a chunk whose own peak dk is below `rel_tol * dk_scale` is
    treated as genuinely freestream (no local wake) and returns lmix=0
    directly, instead of evaluating sqrt(num/den). Without this check, a
    chunk that is truly wake-free can still have tiny non-zero num/den from
    floating-point roundoff in the marched du/dk fields; because both are
    independently-noisy near-zero quantities, their ratio is not
    meaningful and can spuriously come out O(1) or larger (e.g. for
    freestream turbines on the edge of an array, whose own "local" chunk
    extends out to the domain edge with no real wake signal to dominate the
    roundoff). The `eps`-based regularization below only prevents literal
    0/0 NaNs; it does not guard against this noise-over-noise blowup.
    """
    dk_pos = np.maximum(dk, 0)
    if dk_scale is not None and np.max(dk_pos) <= rel_tol * dk_scale:
        return np.float64(0.0)

    eps = np.finfo(float).eps
    _lmix = np.sqrt(
        np.sum(dk_pos ** 1.5)
        / (
            np.sum(np.sqrt(np.maximum(k, 0) + eps) * np.maximum(shear, 0))
            + eps  # de-singularize Dual component
        )
    )
    return _lmix


class TurbulenceModel_tandem_md(CurledTurbulenceModel_kl):
    name = "tandem-md"

    def __init__(
        self,
        curledwake,
        C_nu=0.35,
        C_k1=1,
        l_nw=None,
        l_eps=0.78,
        thresh=1e-6,
        lmix_local=True,
        cache_lmix_ic=True,
        fix_nearwake=True,
    ):
        """
        Initializes the TANDEM curled turbulence model for nu_T with only the 
        minimum dissipation mixing length. Other mixing lengths can be added and
        interpolated in the `TurbulenceModel_tandem` class.

        Parameters
        ----------
        curledwake : CurledWakeWindfield
            The curled wake wind field to which this turbulence model is applied.
        C_nu : float, optional
            Coefficient for eddy viscosity calculation. Default is 0.35.
        C_k1 : float, optional
            Coefficient for turbulent transport. Default is 1.
        l_nw : float, optional
            Near-wake mixing length scale. Default is half the curledwake smoothing factor.
        l_eps : float, optional
            Dissipation length scale. Default is 0.78.
        lmix_local : bool, optional
            Computes local mixing length in chunks. Default True.
        """
        super().__init__(curledwake, C_nu=C_nu, C_k1=C_k1, thresh=thresh)
        self.l_nw = self.curledwake.sigma_ic if l_nw is None else l_nw
        self.l_eps = l_eps
        self.cache = dict()
        self.march_field = True
        self.lmix_local = lmix_local
        self.cache_lmix_ic = cache_lmix_ic
        self.fix_nearwake = fix_nearwake

    def stamp_ic(self, turbine):
        """
        For the KL-MD model, we are going to save the background lmix
        at the rotor disk and store it in the turbine properties
        """
        if not self.cache_lmix_ic:
            return
        du = self.curledwake.modules["du"].get_field_x(turbine.xt)
        dk = self.curledwake.modules["dk"].get_field_x(turbine.xt)
        # because ub, kb are hard to access, just use du and dk rather than full u, k
        lmix = self._lmix(turbine.xt, du, dk, du, dk, fix_nearwake=False)
        # kept in sync with the grid via turbine.fields (see CurledWakeWindfield.adjust_grid_bounds)
        turbine.fields["l_baseflow"] = lmix

    def _lmix(self, x, u, k, du, dk, fix_nearwake=True):
        """Returns the mixing length field at position x."""
        y, z = self.curledwake.y, self.curledwake.z

        # compute the deficit velocity gradient term in the shear production
        shear = (
            np.gradient(du, y, axis=0)
            * np.gradient(u, y, axis=0)
            + np.gradient(du, z, axis=1)
            * np.gradient(u, z, axis=1)
        )
        self.cache["shear"] = shear  # add to cache

        # peak dk over the full x-slice, used to flag individual chunks as
        # genuinely freestream (see `_lmix_md`) rather than noise-dominated
        dk_scale = np.max(np.maximum(dk, 0))

        if self.lmix_local:
            # get xids of turbines here:
            yts_and_edges = [y[0] - 1] + get_turbine_yts(x, self.curledwake.turbines) + [y[-1] + 1]
            yids_mid = get_yid_midpoints(y, yts_and_edges)

            # we have multiple turbines, split this integral into chunks
            with np.errstate(invalid="ignore", divide="ignore"):
                lmix_vals = []
                for kk in range(len(yids_mid) - 1):
                    # compute integral between all sections
                    y1 = yids_mid[kk]
                    y2 = yids_mid[kk+1]
                    _lmix = _lmix_md(dk[y1:y2, ...], k[y1:y2], shear[y1:y2, ...], dk_scale=dk_scale)
                    lmix_vals.append(_lmix)
                lmix_vals = [lmix_vals[0]] + lmix_vals + [lmix_vals[-1]]  # duplicate first and last values

                lmix_func = make_interp_spline(
                    yts_and_edges,
                    np.hstack(lmix_vals),
                    k=1,  # linear interpolation
                )
                lmix = lmix_func(y)[:, None]  # interpolate lmix to y-axis
        else:
            lmix = _lmix_md(dk, k, shear, dk_scale=dk_scale)[None, None]

        if fix_nearwake:
            # compute the near-wake mask
            nw_mask, l_base = get_x_nw(
                x,
                self.curledwake.y,
                self.curledwake.z,
                self.curledwake.turbines,
                use_constant_x0=self.curledwake.use_constant_x0,
            )  # note: this mask is 1 -> far wake, 0 -> near wake.
            self.cache["nw_mask"] = nw_mask  # add to cache

            # apply near-wake model
            lmix_nw = self.l_nw * nw_mask + l_base
            lmix = lmix * (nw_mask >= 0.95) + lmix_nw * (nw_mask < 0.95)

        if np.any(lmix < 0):
            raise ValueError("lmix is negative")
        if np.any(np.isnan(lmix)) or np.any(np.isinf(_lmix)):
            raise IntegrationException("Invalid lmix value: {_lmix} at x={x}, exiting.")

        self.cache["lmix"] = lmix  # add to cache
        return lmix

    def postprocess_lmix(self):
        """Compute lmix from the stored du, dk 3D windfields"""
        # right now, lmix is just one value per x-location
        lmix = np.zeros_like(self.curledwake.u)
        u, k, du, dk = [getattr(self.curledwake, name) for name in ["u", "k", "du", "dk"]]
        for i, x in enumerate(self.curledwake.x):
            _lmix = self._lmix(x, u[i, ...], k[i, ...], du[i, ...], dk[i, ...], fix_nearwake=self.fix_nearwake)
            lmix[i, ...] = _lmix

        return lmix

    def postprocess_nu_T(self):
        """Compute nu_T from the stored du, dk 3D windfields"""
        nu_T = np.zeros_like(self.curledwake.u)
        u, k, du, dk = [getattr(self.curledwake, name) for name in ["u", "k", "du", "dk"]]
        for i, x in enumerate(self.curledwake.x):
            self.curledwake.shared_flow_data = dict(u=u[i, ...], k=k[i, ...], du=du[i, ...], dk=dk[i, ...], kb=k[i, ...]-dk[i, ...])
            nu_T[i, ...] = self.nu_T(x)
        return nu_T

    def nu_T(self, x):
        """New formulation for the eddy viscosity"""
        vars = self.curledwake.shared_flow_data
        lmix = self._lmix(x, vars["u"], vars["k"], vars["du"], vars["dk"], fix_nearwake=self.fix_nearwake)

        sqrt_k = np.sqrt(np.clip(vars["k"], np.finfo(float).eps, None))
        self.nu_T_cached = self.C_nu * sqrt_k * lmix
        return self.nu_T_cached

    def ddx(self, x):
        """Computes the dk/dx term for the turbulence model."""
        y = self.curledwake.grid[1]
        z = self.curledwake.grid[2]
        nu_T = self.nu_T(x)
        vars = self.curledwake.shared_flow_data
        u, v, w, dk = [vars[name] for name in ["u", "v", "w", "dk"]]

        # Get shear from cache (computed in _lmix)
        shear = self.cache["shear"]

        # transport equation for k_wake, written in parabolic form:
        udkdx = (
            -v * np.gradient(dk, y, axis=0)
            - w * np.gradient(dk, z, axis=1)
            + nu_T * shear
            + self.C_k1  # pull out of gradient as C_k1 is constant
            * (
                np.gradient(nu_T * np.gradient(dk, y, axis=0), y, axis=0)
                + np.gradient(nu_T * np.gradient(dk, z, axis=1), z, axis=1)
            )
            # need np.clip for the sqrt here
            - (np.clip(dk, 0, None) ** (3 / 2)) / self.l_eps
        )

        # clear cache
        self.cache = dict()
        return udkdx / u


class TurbulenceModel_tandem(TurbulenceModel_tandem_md):
    name = "tandem"

    def __init__(
        self,
        curledwake,
        C_nu=0.32,
        C_k1=1,
        l_nw=None,
        l_eps=1.0,
        thresh=1e-6,
        lmix_local=True,
        L_obu=np.inf,
        C_w=3.0,
        C_b=1.0,
        kappa=0.4,
        inflow=None,
        cache_lmix_ic=True,
        fix_nearwake=True,
    ):
        """
        Initializes the TANDEM turbulence closure with stable boundary layer
        (stratification) corrections to the mixing length.

        Extra parameters for MOST: 
        - L_obu: float, optional
            Obukhov length for stratification, default is np.inf (neutral)
        - C_w: float, optional
            Coefficient for wall mixing length, default is 3.0
        - kappa: float, optional
            von Karman constant, default is 0.4

        Extra parameters for stratification: (unused)
        - inflow: xarray.Dataset or None
            Inflow data to derive mixing length from.
            Must contain keys `Tbar` and `tke` and attribute `Fr`.
        - C_b: float, optional
            Coefficient for buoyancy-based mixing length, default is 1.0
        """
        super().__init__(
            curledwake,
            C_nu=C_nu,
            C_k1=C_k1,
            l_nw=l_nw,
            l_eps=l_eps,
            thresh=thresh,
            lmix_local=lmix_local,
            cache_lmix_ic=cache_lmix_ic,
            fix_nearwake=fix_nearwake,
        )
        if inflow is not None:
            # Define buoyancy length scale as C_b * sqrt(tke) / N_bv, where N_bv is the Brunt-Vaisala frequency
            dTdz = np.clip(inflow["Tbar"].differentiate("z"), 1e-1, None)  # clip small and negative values
            N_bv = np.sqrt(dTdz / inflow["Tbar"]) / inflow.Fr
            self.lb = C_b * np.sqrt(inflow["tke"]) / N_bv
            self.inflow_z = np.array(inflow.z)
        else:
            self.lb = None
            self.inflow_z = None

        self.C_w = C_w
        self.C_b = C_b
        self.kappa = kappa
        self.L_obu = L_obu

    def _lmix(self, x, u, k, du, dk, fix_nearwake=True):
        l_md = super()._lmix(x, u, k, du, dk, fix_nearwake=fix_nearwake)  # this is lmix(y, z)
        to_interp = [l_md, ]

        # add wall mixing length, if applicable
        if self.curledwake.bottom_wall_z > -np.inf:
            zabs = self.curledwake.z - self.curledwake.bottom_wall_z
            #  if there is a point nearer the wall than dz/2, clip to when computing wall length scale
            dz = zabs[1] - zabs[0]
            zabs = np.clip(zabs, dz/2, None)
            lw = self.C_w * self.kappa * zabs / phi_m(zabs / self.L_obu)
            to_interp.append(lw[None, :])

        # can continue to add mixing lengths...

        # add buoyancy mixing length, if applicable (not used in 2026 TANDEM paper)
        if self.lb is not None:
            lb = np.interp(self.curledwake.z, self.inflow_z, self.lb)
            to_interp.append(lb)

        if len(to_interp) > 1:
            return np.min(np.broadcast_arrays(*to_interp), axis=0)  # expand dims
            # return softmin(np.broadcast_arrays(*to_interp), axis=0, Lambda=0.1)  # expand dims
        else:
            return l_md
