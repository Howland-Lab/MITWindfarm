import copy

import numpy as np
from attrs import define, field
from floris.core.rotor_velocity import \
    compute_tilt_angles_for_floating_turbines
from floris.core.turbine.turbine import select_multidim_condition
from floris.core.turbine import SimpleTurbine, CosineLossTurbine
from floris.core.wake_model import BaseWakeModel
from UnifiedMomentumModel.Utilities.Geometry import (
    calc_eff_yaw,
    eff_yaw_inv_rotation
)

from ._Layout import Layout
from .Rotor import Rotor, RotorSolution
from .windfarm import CurledWindfarm
from .Windfield import PowerLaw


@define
class FlorisCurledWindfarm(BaseWakeModel):
    """
    Interface for using the MITWindfarm CurledWindfarm solver as a wake model in FLORIS.
    This allows the use of the CurledWindfarm solver within the FLORIS framework (and using FLORIS
    turbine operation models).

    A power law background wind field is assumed, and the FLORIS flow field is used to set the wind
    speed and turbulence intensity at the reference height. Further, all turbines in the farm are
    assumed to be of the same type and heterogeneous inflows are not supported (errors are raised
    if these conditions are not met).
    """

    solver_kwargs: dict = field(default=None)
    use_floris_tilt: bool = field(default=True, init=True)

    def turbine_solve(self, farm, flow_field, grid):
        self._check_valid_turbine_types(farm)
        self._check_valid_flow_field(flow_field)
        powers, thrust_coefficients, axial_inductions = self._solve_and_evaluate(
            farm, flow_field, grid, grid
        )
        # Assign outputs to farm
        farm.set_turbine_outputs_by_original_ordering(
            powers=powers,
            thrust_coefficients=thrust_coefficients,
            axial_inductions=axial_inductions
        )

    def point_solve(self, farm, flow_field, grid):
        self._check_valid_turbine_types(farm)
        self._check_valid_flow_field(flow_field)
        turbine_grid = self.generate_turbine_grid_objects(farm, flow_field)[2]
        self._solve_and_evaluate(farm, flow_field, grid, turbine_grid)

    def _check_valid_turbine_types(self, farm):
        # Require all turbines to be the same type
        if not np.all([t == farm.turbines[0] for t in farm.turbines]):
            raise NotImplementedError(
                "Varying turbine models not supported in FlorisCurledWindfarm"
            )
        
    def _check_valid_flow_field(self, flow_field):
        if flow_field.het_map is not None or flow_field.heterogeneous_inflow_config is not None:
            raise NotImplementedError(
                "Heterogeneous inflows are not supported in FlorisCurledWindfarm."
            )

    def _solve_and_evaluate(self, farm, flow_field, grid, turbine_grid):

        D = farm.rotor_diameters[0]

        rotor_model = RotorWrapper(
            operation_model=farm.turbines[0].operation_model,
            power_thrust_table=farm.turbines[0].power_thrust_table,
            rotor_diameter=D,
            rotor_points=(
                turbine_grid.x_sorted[0,0],
                turbine_grid.y_sorted[0,0],
                turbine_grid.z_sorted[0,0]
            ),
            air_density=flow_field.air_density,
            tilt_interp=farm.turbines[0].tilt_interp,
            average_method=turbine_grid.average_method,
            cubature_weights=grid.cubature_weights,
            correct_cp_ct_for_tilt=farm.turbines[0].correct_cp_ct_for_tilt,
            use_floris_tilt=self.use_floris_tilt,
        )

        # Create variables to store turbine outputs, and assign to farm at 
        # end of solve routine.
        powers = np.zeros((flow_field.n_findex, farm.n_turbines))
        thrust_coefficients = np.zeros((flow_field.n_findex, farm.n_turbines))
        axial_inductions = np.zeros((flow_field.n_findex, farm.n_turbines))

        # Loop over findices, and called CurledWindfarm solver for each findex.
        # If CurledWindfarm is updated to handle multiple findices simultaneously, this loop can be
        # removed.
        for f in range(flow_field.n_findex):

            rotor_model.update_freestream_windspeed(flow_field.wind_speeds[f])

            # Handle possible multidimensional turbine conditions
            if flow_field.multidim_conditions is not None:
                rotor_model.set_multidim_condition(flow_field.multidim_conditions, f)

            turbines_x = turbine_grid.x_sorted.mean(axis=(2,3))[f]
            turbines_y = turbine_grid.y_sorted.mean(axis=(2,3))[f]
            turbines_z = turbine_grid.z_sorted.mean(axis=(2,3))[f]
            layout = Layout(turbines_x/D, turbines_y/D, turbines_z/D)

            wf_init_kwargs = {
                "rotor_model": rotor_model,
                "base_windfield": PowerLaw(
                    1.0, # Will be scaled up after solve
                    flow_field.reference_wind_height/D,
                    flow_field.wind_shear,
                    flow_field.turbulence_intensities[f]
                ),
                "TIamb": flow_field.turbulence_intensities[f],
                "solver_kwargs": self.solver_kwargs,
            }
            yaw = farm.yaw_angles[f, :]

            if not self.use_floris_tilt:
                tilt = 0.0 * np.ones_like(yaw)
            else:
                tilt = farm.turbines[0].ref_tilt * np.ones_like(yaw)
                # tilt correction to be applied in rotor solve, if applicable

            setpoints = list(zip(np.nan * np.ones_like(yaw), yaw, tilt))

            # Reinstantiate and solve for the current findex
            windfarm = CurledWindfarm(**wf_init_kwargs)
            windfarm_sol = windfarm(layout, setpoints)

            powers[f] = np.array([r.extra.power for r in windfarm_sol.rotors])
            thrust_coefficients[f] = np.array([r.extra.Ct for r in windfarm_sol.rotors])
            axial_inductions[f] = np.array([r.extra.an for r in windfarm_sol.rotors])

            # Extract the wind speeds at the turbine locations
            relative_velocities = windfarm_sol.windfield.wsp(
                grid.x_sorted[f]/D, grid.y_sorted[f]/D, grid.z_sorted[f]/D
            ) * flow_field.wind_speeds[f] # Scale by the reference wind speed for this findex

            # Assign to flow field
            flow_field.u_sorted[f] = relative_velocities

        return powers, thrust_coefficients, axial_inductions

class RotorWrapper(Rotor):
    """
    Wrapper for the FLORIS operation model to be used as a rotor model in MITWindfarm. 
    This allows the use of the FLORIS operation model within the CurledWindfarm solver, which is
    necessary for the FlorisCurledWindfarm wake model to work.
    """

    def __init__(self,
        operation_model,
        power_thrust_table,
        rotor_diameter,
        rotor_points,
        air_density = 1.225,
        tilt_interp = None,
        average_method = "cubic-mean",
        cubature_weights = None,
        correct_cp_ct_for_tilt = True,
        use_floris_tilt = True,
    ):
        self.operation_model = operation_model
        self.power_thrust_table = power_thrust_table
        self.rotor_diameter = rotor_diameter
        self.air_density = air_density
        self.tilt_interp = tilt_interp
        self.average_method = average_method
        self.cubature_weights = cubature_weights
        self.correct_cp_ct_for_tilt = correct_cp_ct_for_tilt
        self.use_floris_tilt = use_floris_tilt

        # Unpack and nondimensionalize the rotor points
        self.rotor_x = (rotor_points[0] - rotor_points[0].mean()) / rotor_diameter
        self.rotor_y = (rotor_points[1] - rotor_points[1].mean()) / rotor_diameter
        self.rotor_z = (rotor_points[2] - rotor_points[2].mean()) / rotor_diameter

        if "condition_keys" in power_thrust_table:
            self._power_thrust_table_md = copy.deepcopy(power_thrust_table)
            self.multidimensional_turbine = True
        else:
            self.multidimensional_turbine = False
        self.multidim_condition = None

    def set_multidim_condition(self, multidim_conditions, findex):
        """
        Select the power, thrust curves to evaluate.
        Only used if multidimensional turbines are used.
        """
        if self.multidimensional_turbine:
            pass
        else:
            raise ValueError(
                "Attempting to set a multidimensional condition for a turbine that does not have a "
                "multidimensional power/thrust table."
            )

        md_cond_f = copy.deepcopy(multidim_conditions)

        # Get findex position, if necessary
        for k, v in multidim_conditions.items():
            if isinstance(v, (list, np.ndarray)):
                md_cond_f[k] = v[findex]

        # Handle multidimensional turbine conditions.
        self.multidim_condition = tuple(select_multidim_condition(
            md_cond_f,
            [k for k in self._power_thrust_table_md.keys() if k != "condition_keys"],
             self._power_thrust_table_md["condition_keys"],
            1
        )[0][0])
        self.power_thrust_table = self._power_thrust_table_md[self.multidim_condition]

    def update_freestream_windspeed(self, Uref):
        self.Uref = Uref

    def __call__(
        self, x: float, y: float, z: float, windfield, Ctprime, yaw=0, tilt=0,
    ):
        """
        Note that the value of Ctprime passed will be ignored, as Ctprime is computed 
        during the call based on the evaluated thrust_coefficient
        """
        Us_grid = windfield.wsp(x + self.rotor_x, y + self.rotor_y, z + self.rotor_z)
        Us = Us_grid.mean()
        TIs = windfield.TI(x, y, z)

        if self.multidimensional_turbine and self.multidim_condition is None:
            raise ValueError(
                "A multidimensional turbine is being used, "
                "but multidimensional condition has not been set."
            )

        # Could query the wind speed at the rotor points, rather than just the hub height

        yaw_r = np.deg2rad(yaw)
        if self.correct_cp_ct_for_tilt:
            tilt = compute_tilt_angles_for_floating_turbines(
                tilt_angles=tilt,
                tilt_interp=self.tilt_interp,
                rotor_effective_velocities=Us * self.Uref,
            )
        tilt_r = np.deg2rad(tilt)


        # Now, should be able to evaluate the FLORIS operation model (thrust coefficient)?
        Ct = self.operation_model.thrust_coefficient(
            power_thrust_table=self.power_thrust_table,
            velocities=(Us_grid * self.Uref)[None, None, :, :],
            turbulence_intensities=np.array([[TIs]]),
            air_density=self.air_density,
            yaw_angles=np.array([[yaw]]),
            tilt_angles=np.array([[tilt]]),
            power_setpoints=None,
            awc_modes=None,
            awc_amplitudes=None,
            tilt_interp=self.tilt_interp,
            average_method=self.average_method,
            cubature_weights=self.cubature_weights,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
        )

        a = self.operation_model.axial_induction(
            power_thrust_table=self.power_thrust_table,
            velocities=(Us_grid * self.Uref)[None, None, :, :],
            turbulence_intensities=np.array([[TIs]]),
            air_density=self.air_density,
            yaw_angles=np.array([[yaw]]),
            tilt_angles=np.array([[tilt]]),
            power_setpoints=None,
            awc_modes=None,
            awc_amplitudes=None,
            tilt_interp=self.tilt_interp,
            average_method=self.average_method,
            cubature_weights=self.cubature_weights,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
        )

        P = self.operation_model.power(
            power_thrust_table=self.power_thrust_table,
            velocities=(Us_grid * self.Uref)[None, None, :, :],
            turbulence_intensities=np.array([[TIs]]),
            air_density=self.air_density,
            yaw_angles=np.array([[yaw]]),
            tilt_angles=np.array([[tilt]]),
            power_setpoints=None,
            awc_modes=None,
            awc_amplitudes=None,
            tilt_interp=self.tilt_interp,
            average_method=self.average_method,
            cubature_weights=self.cubature_weights,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
        )

        # Compute tilt for rotor solution.
        if self.correct_cp_ct_for_tilt and self.tilt_interp is not None:
            tilt = self.tilt_interp(Us)

        if hasattr(self.operation_model, "near_wake_velocities"):
            u4, v4, w4 = self.operation_model.near_wake_velocities(
                power_thrust_table=self.power_thrust_table,
                velocities=(Us_grid * self.Uref)[None, None, :, :],
                turbulence_intensities=np.array([[TIs]]),
                air_density=self.air_density,
                yaw_angles=np.array([[yaw]]),
                tilt_angles=np.array([[tilt]]),
                power_setpoints=None,
                awc_modes=None,
                awc_amplitudes=None,
                tilt_interp=self.tilt_interp,
                average_method=self.average_method,
                cubature_weights=self.cubature_weights,
                correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
            )
        elif isinstance(self.operation_model, CosineLossTurbine):
            # Note that in this case, the value for axial induction a returned by
            # CosineLossTurbine.axial_induction is _not_ consistent with the UMM,
            # which means that the value of Ctprime computed below is also not consistent
            # with the UMM.

            # Add second cosine term, as not done in CosineLoss model
            Ct = Ct * np.cos(calc_eff_yaw(yaw_r, tilt_r))

            u4, v4, w4 = near_wake_velocities_standin(Ct, Us, yaw_r, tilt_r)
        elif isinstance(self.operation_model, SimpleTurbine):
            if yaw_r != 0 or tilt_r != 0:
                raise NotImplementedError(
                    "The SimpleTurbine operation model does not support yaw or tilt. "
                    "Cannot compute near wake velocities. Consider using CosineLossTurbine instead."
                )

            u4, v4, w4 = near_wake_velocities_standin(Ct, Us, 0.0, 0.0)
        else:
            raise NotImplementedError(
                "The operation model does not have a near_wake_velocities method, and is not a" \
                "SimpleTurbine or CosineLossTurbine. Cannot compute near wake velocities."
            )

        # Compute Ctprime based on axial induction, thrust coefficient, and effective yaw angle
        Ctprime = Ct / ((1 - a)**2 * np.cos(calc_eff_yaw(yaw_r, tilt_r))**2)

        class extra:
            """
            Small class to return normalized values for axial induction and u4.
            Also used to store turbine power so that it can be passed back to FLORIS.
            """
            def __init__(self, a, u4, Ct, REWS, power):
                self.an = a
                self.Ct = Ct
                self.u4 = u4 / REWS # Non-dimensional version
                self.x0 = None # Computed within CurledWindfarm solve
                self.power = power

        # Remove null numpy dimensions and return floats as RotorSolution
        rotor_solution = RotorSolution(
            yaw=np.deg2rad(yaw),
            Cp=None, # Not computed by FLORIS rotor models
            Ct=Ct[0,0] * Us**2,
            Ctprime=Ctprime[0,0],
            an=a[0,0] * Us,
            u4=u4[0,0],
            v4=v4[0,0],
            REWS=Us,
            tilt=np.deg2rad(tilt) if self.use_floris_tilt else 0.0,
            w4=w4[0,0],
            TI=TIs,
            extra=extra(a[0,0], u4[0,0], Ct[0,0], Us, P[0,0])
        )
        return rotor_solution

def near_wake_velocities_standin(Ct, Us, yaw_r, tilt_r):

    yaw_r_eff = calc_eff_yaw(yaw_r, tilt_r)

    # In rotated frame of reference
    u4 = np.sqrt(1 - 1/16 * Ct**2 * np.sin(yaw_r_eff)**2 - Ct) * Us
    v4 = - (1/4) * Ct * np.sin(yaw_r_eff) * Us
    w4 = 0.0

    # Convert to global frame of reference
    u4, v4, w4 = eff_yaw_inv_rotation(u4, v4, w4, yaw_r_eff, yaw_r, tilt_r)

    return u4, v4, w4
