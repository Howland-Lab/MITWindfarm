from dataclasses import dataclass
from attrs import define, field

import copy
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import LinearNDInterpolator

from floris.core.wake_model import BaseWakeModel
from floris.core.turbine.turbine import select_multidim_condition

from .Windfield import PowerLaw
from ._Layout import Layout
from .windfarm import CurledWindfarm
from .Rotor import Rotor, RotorSolution

@define
class FlorisCurledWindfarm(BaseWakeModel):
    """
    Interface for using the MITWindfarm CurledWindfarm solver as a wake model in FLORIS.
    This allows the use of the CurledWindfarm solver within the FLORIS framework (and using FLORIS
    turbine operation models).

    A power law background wind field is assumed, and the FLORIS flow field is used to set the wind
    speed and turbulence intensity at the reference height. Further, 
    """
    # Parameters (to fill in)
    solver_kwargs = field(default=None)

    # TODO: Can I declare this directly with attrs?
    # Interestingly, this doesn't work, something to do with BaseLibrary
    # I think. Will need to work through that.
    # def __attrs_post_init__(self):
    #     supported_models = [CurledWindfarm]
    #     if type(self.windfarm) not in supported_models:
    #         raise NotImplementedError(
    #             "The FLORIS interface for MITWindfarm does not support type ",
    #             type(self.wndfarm),
    #             ". Supported types:",
    #             supported_models
    #         )
    # TODO: Check a ThrustBased or BEM model here; not a UMM model?
    # should take pitch and TSR as inputs, I think, or will need to figure
    # that out?
    # These other models will need to somehow pass pitch and tsr to the wake
    # model, right?

    # Check that rotor_model is the default value (AD); if not, raise a warning. Either way, 
    # ignore and use the wrapper for the FLORIS operation model.


    def turbine_solve(self, farm, flow_field, grid):
        self._solve_and_evaluate(farm, flow_field, grid, grid)

    def point_solve(self, farm, flow_field, grid):
        turbine_grid = self.generate_turbine_grid_objects(farm, flow_field)[2]
        self._solve_and_evaluate(farm, flow_field, grid, turbine_grid)

    def _solve_and_evaluate(self, farm, flow_field, grid, turbine_grid):
        # Assume all turbines have the same rotor diameter; not sure if
        # methods below handle varying rotor diameters?
        if not np.all(np.array(farm.turbine_type) == farm.turbine_type[0]):
            raise NotImplementedError("Varying turbine types not supported in FlorisCurledWindfarm")
        elif not np.all(farm.rotor_diameters == farm.rotor_diameters.mean()):
            raise NotImplementedError(
                "Varying rotor diameters not supported in FlorisCurledWindfarm"
            )
        else:
            D = farm.rotor_diameters.mean()
            turbine_type = farm.turbine_definitions[0]['turbine_type']

        if flow_field.het_map or flow_field.heterogeneous_inflow_config:
            raise NotImplementedError(
                "Heterogeneous inflows are not supported in FlorisCurledWindfarm."
            )

        rotor_model = RotorWrapper(
            thrust_coefficient_function=farm.turbine_thrust_coefficient_functions[turbine_type], #
            power_function=farm.turbine_power_functions[turbine_type],
            axial_induction_function=farm.turbine_axial_induction_functions[turbine_type],
            power_thrust_table=farm.turbine_power_thrust_tables[turbine_type],
            air_density=flow_field.air_density,
            tilt_interp=farm.turbine_tilt_interps[turbine_type],
            average_method=turbine_grid.average_method,
            cubature_weights=grid.cubature_weights,
            correct_cp_ct_for_tilt=True
        )

        farm.turbine_powers = np.zeros((flow_field.n_findex, farm.n_turbines))

        # Use sorted version
        for f in range(flow_field.n_findex):

            # Handle possible multidimensional turbine conditions
            if flow_field.multidim_conditions is not None:
                rotor_model.set_multidim_condition(flow_field.multidim_conditions, f)

            turbines_x = turbine_grid.x_sorted.mean(axis=(2,3))[f]
            turbines_y = turbine_grid.y_sorted.mean(axis=(2,3))[f]
            turbines_z = turbine_grid.z_sorted.mean(axis=(2,3))[f]
            layout = Layout(turbines_x/D, turbines_y/D, turbines_z/D)

            # Generate calling arguments based on windfarm type. Also depends on rotor model; not yet handled.
            # Sometimes, setpoints should include tsr and pitch; other times, ctprime? Depends on rotor model?
            wf_init_kwargs = {
                "rotor_model": rotor_model,
                "base_windfield": PowerLaw(
                    flow_field.wind_speeds[f],
                    flow_field.reference_wind_height/D,
                    flow_field.wind_shear,
                    flow_field.turbulence_intensities[f]
                ),
                "TIamb": flow_field.turbulence_intensities[f], # Needed? not sure
                "solver_kwargs": self.solver_kwargs,
            }
            yaw = farm.yaw_angles[f, :]
            tilt = farm.tilt_angles[f, :]
            CTprime = 2.0 * np.ones_like(yaw) # Temporary
            setpoints = list(zip(CTprime, yaw, tilt))

            # Reinstantiate and solve for the current findex
            windfarm = CurledWindfarm(**wf_init_kwargs)
            windfarm_sol = windfarm(layout, setpoints)

            farm.turbine_powers[f] = np.array([r.Cp for r in windfarm_sol.rotors])

            # Extract the wind speeds at the turbine locations
            relative_velocities = windfarm_sol.windfield.wsp(
                grid.x_sorted[f]/D, grid.y_sorted[f]/D, grid.z_sorted[f]/D
            )

            # Assign to flow field
            flow_field.u_sorted[f] = relative_velocities

class RotorWrapper(Rotor):
    """
    Wrapper for the FLORIS operation model to be used as a rotor model in MITWindfarm. 
    This allows the use of the FLORIS operation model within the CurledWindfarm solver, which is
    necessary for the FlorisCurledWindfarm wake model to work.
    """

    # TODO: Handle multidimensional turbine conditions; how much can I take from turbine.py?

    def __init__(self,
        thrust_coefficient_function,
        axial_induction_function,
        power_function,
        power_thrust_table,
        air_density = 1.225,
        tilt_interp = None,
        average_method = "cubic-mean",
        cubature_weights = None,
        correct_cp_ct_for_tilt = True,
    ):
        self.thrust_coefficient_function = thrust_coefficient_function
        self.power_function = power_function
        self.axial_induction_function = axial_induction_function
        self.power_thrust_table = power_thrust_table
        self.air_density = air_density
        self.tilt_interp = tilt_interp
        self.average_method = average_method
        self.cubature_weights = cubature_weights
        self.correct_cp_ct_for_tilt = correct_cp_ct_for_tilt

        if "condition_keys" in power_thrust_table:
            self._power_thrust_table_md = copy.deepcopy(power_thrust_table)
            self.multidimensional_turbine = True
        else:
            self.multidimensional_turbine = False
        self.multidim_condition = None

    def set_multidim_condition(self, multidim_conditions, findex):
        """
        Select the power, thrust curves to evaluate. Only used if multidimensional turbines are used.
        """
        if self.multidimensional_turbine:
            pass
        else:
            raise ValueError(
                "Attempting to set a multidimensional condition for a turbine that does not have a multidimensional power/thrust table."
            )
        
        # Get findex position, if necessary
        for k, v in multidim_conditions.items():
            if isinstance(v, (list, np.ndarray)):
                multidim_conditions[k] = v[findex]

        # Handle multidimensional turbine conditions.
        self.multidim_condition = tuple(select_multidim_condition(
            multidim_conditions,
            [k for k in self._power_thrust_table_md.keys() if k != "condition_keys"],
             self._power_thrust_table_md["condition_keys"],
            1
        )[0][0])
        self.power_thrust_table = self._power_thrust_table_md[self.multidim_condition]

    def __call__(
        self, x: float, y: float, z: float, windfield, Ctprime, yaw=0, tilt=0,
    ):
        # TODO: Do I need to account for yaw, tilt? Seems likely not.
        xs_glob = x
        ys_glob = y
        zs_glob = z
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        if self.multidimensional_turbine and self.multidim_condition is None:
            raise ValueError(
                "A multidimensional turbine is being used, "
                "but multidimensional condition has not been set."
            )

        # Now, should be able to evaluate the FLORIS operation model (thrust coefficient)?
        Ct = self.thrust_coefficient_function(
            power_thrust_table=self.power_thrust_table,
            velocities=Us,
            turbulence_intensities=TIs,
            air_density=self.air_density,
            yaw_angles=yaw,
            tilt_angles=tilt,
            power_setpoints=None, # Figure out how to raise warning if nondefault
            awc_modes=None,
            awc_amplitudes=None,
            tilt_interp=self.tilt_interp,
            average_method=self.average_method,
            cubature_weights=self.cubature_weights,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
        )

        a = self.axial_induction_function(
            power_thrust_table=self.power_thrust_table,
            velocities=Us,
            turbulence_intensities=TIs,
            air_density=self.air_density,
            yaw_angles=yaw,
            tilt_angles=tilt,
            power_setpoints=None, # Figure out how to raise warning if nondefault
            awc_modes=None,
            awc_amplitudes=None,
            tilt_interp=self.tilt_interp,
            average_method=self.average_method,
            cubature_weights=self.cubature_weights,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
        )

        P = self.power_function(
            power_thrust_table=self.power_thrust_table,
            velocities=Us,
            turbulence_intensities=TIs,
            air_density=self.air_density,
            yaw_angles=yaw,
            tilt_angles=tilt,
            power_setpoints=None, # Figure out how to raise warning if nondefault
            awc_modes=None,
            awc_amplitudes=None,
            tilt_interp=self.tilt_interp,
            average_method=self.average_method,
            cubature_weights=self.cubature_weights,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
        )

        ### MIT team to check: Are the following calculations correct? Do they need to be updated?
        REWS = np.mean(Us)
        RETI = np.mean(TIs)
        Ctprime = 4*a/(1-a)
        u4 = np.sqrt(np.maximum(1 - Ct, 0)) * Us # TODO: What is u4?
        v4 = - (1/4) * Ct * np.sin(np.deg2rad(yaw)) * Us
        w4 = np.zeros_like(Us) # TODO: What is w4?

        # Compute tilt for rotor solution.
        if self.correct_cp_ct_for_tilt and self.tilt_interp is not None:
            tilt = self.tilt_interp(Us)
        relative_tilt = tilt - self.power_thrust_table["ref_tilt"]

        class extra:
            """
            Small class to return normalized values for axial induction and u4.
            """
            def __init__(self, an, u4, REWS):
                self.an = an / REWS
                self.u4 = u4 / REWS

        ### MIT team to check: are these the correct values to pass to the RotorSolution object?
        rotor_solution = RotorSolution(
            yaw=np.deg2rad(yaw),
            Cp=P, # May not be needed
            Ct=Ct * REWS**2,
            Ctprime=Ctprime, # Check if computation valid
            an=a * REWS, # Axial induction (why multiply by REWS?)
            u4=u4,
            v4=v4,
            REWS=REWS,
            tilt=np.deg2rad(relative_tilt), # Correct? Or should this be absolute tilt?
            w4=w4,
            TI=RETI,
            extra=extra(a, u4, REWS), # Model needs normalized u4, axial induction
        )
        return rotor_solution

# TODO:
# - multidimensional turbine conditions
# - X Non power law base wind field (can I construct from FLORIS flow_field?) DOES NOT WORK; internal solver expects PowerLaw (z only).
# - Check accounting for yaw ang tilt in the xs_glob, ys_glob, zs_glob calculation.