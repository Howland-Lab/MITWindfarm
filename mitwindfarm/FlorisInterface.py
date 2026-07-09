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
    use_floris_tilt = field(default=True, init=True)

    def turbine_solve(self, farm, flow_field, grid):
        self._check_valid_turbine_types(farm)
        self._solve_and_evaluate(farm, flow_field, grid, grid)

    def point_solve(self, farm, flow_field, grid):
        self._check_valid_turbine_types(farm)
        turbine_grid = self.generate_turbine_grid_objects(farm, flow_field)[2]
        self._solve_and_evaluate(farm, flow_field, grid, turbine_grid)

    def _check_valid_turbine_types(self, farm):
                # Assume all turbines have the same rotor diameter; not sure if
        # methods below handle varying rotor diameters?
        if not np.all(np.array(farm.turbine_type) == farm.turbine_type[0]):
            raise NotImplementedError("Varying turbine types not supported in FlorisCurledWindfarm")
        elif not np.all(farm.rotor_diameters == farm.rotor_diameters.mean()):
            raise NotImplementedError(
                "Varying rotor diameters not supported in FlorisCurledWindfarm"
            )
        # TODO: Add check for a valid operation model (i.e., has the velocity components)

    def _solve_and_evaluate(self, farm, flow_field, grid, turbine_grid):

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
            correct_cp_ct_for_tilt=True, # TODO: Pass in?
            use_floris_tilt=self.use_floris_tilt
        )

        # Temporary; this shouldn't be needed, but it seems I have something not quite right in
        # initializing the farm object and its attributes.
        farm.turbine_powers = np.zeros((flow_field.n_findex, farm.n_turbines))
        farm.turbine_thrust_coefficients = np.zeros((flow_field.n_findex, farm.n_turbines))
        farm.turbine_axial_inductions = np.zeros((flow_field.n_findex, farm.n_turbines))

        # Use sorted version
        for f in range(flow_field.n_findex):

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
                    flow_field.wind_speeds[f],
                    flow_field.reference_wind_height/D,
                    flow_field.wind_shear,
                    flow_field.turbulence_intensities[f]
                ),
                "TIamb": flow_field.turbulence_intensities[f], # Needed? not sure
                "solver_kwargs": self.solver_kwargs,
            }
            yaw = farm.yaw_angles[f, :] # How is this used?
            tilt = farm.tilt_angles[f, :] # How is this used?
            setpoints = list(zip(np.nan * np.ones_like(yaw), yaw, tilt))

            # Reinstantiate and solve for the current findex
            windfarm = CurledWindfarm(**wf_init_kwargs)
            windfarm_sol = windfarm(layout, setpoints)

            farm.turbine_powers[f] = np.array([r.Cp for r in windfarm_sol.rotors])
            farm.turbine_thrust_coefficients[f] = np.array([r.extra.Ct for r in windfarm_sol.rotors])
            farm.turbine_axial_inductions[f] = np.array([r.extra.an for r in windfarm_sol.rotors])

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

    TODO: use attrs? Not strictly needed.
    """

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
        use_floris_tilt = True
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
        self.use_floris_tilt = use_floris_tilt

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
        """
        Note that the value of Ctprime passed will be ignored, as Ctprime is computed 
        during the call.
        """
        Us = windfield.wsp(x, y, z)
        TIs = windfield.TI(x, y, z)

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

        # Compute tilt for rotor solution.
        if self.correct_cp_ct_for_tilt and self.tilt_interp is not None:
            tilt = self.tilt_interp(Us)

        ### MIT team to check: Are the following calculations correct? Do they need to be updated?
        cos_eff_yaw = np.cos(yaw) * np.cos(tilt)
        REWS = np.mean(Us)
        RETI = np.mean(TIs)
        ### TODO: check a is equivalent to a_n (normal component?)
        ### Also check how c_t is defined; is there a single cosine yaw term there? Does that match?
        ### TODO: should there be another term for tilt? How does that come in?
        Ctprime = Ct / ((1 - a)**2 * cos_eff_yaw**2)
        # Ctprime = Ct / ((1 - a)**2 * np.cos(np.deg2rad(yaw))**2)
        ### TODO: perhaps not appropriate; need to come from flow model
        u4 = np.sqrt(np.maximum(1 - Ct, 0)) * Us
        v4 = - (1/4) * Ct * np.sin(np.deg2rad(yaw)) * Us
        # Or should these use 2.20a,b from Heck et al (2023)?
        # u4 = (4 - Ctprime*np.cos(np.deg2rad(yaw))**2) / (4 + Ctprime*np.cos(np.deg2rad(yaw))**2) * Us
        # v4 = - (4 * Ctprime * np.sin(np.deg2rad(yaw))*np.cos(np.deg2rad(yaw))**2) / (4 + Ctprime*np.cos(np.deg2rad(yaw))**2)**2 * Us
        w4 = np.zeros_like(Us)

        print(f"Tilt (deg): {tilt:.2f}")

        class extra:
            """
            Small class to return normalized values for axial induction and u4.
            """
            def __init__(self, a, u4, Ct, REWS):
                self.an = a
                self.Ct = Ct
                self.u4 = u4 / REWS

        ### MIT team to check: are these the correct values to pass to the RotorSolution object?
        rotor_solution = RotorSolution(
            yaw=np.deg2rad(yaw),
            Cp=P, # Needed to save off power in main FlorisCurledWindfarm solve
            Ct=Ct * REWS**2, # What is this?
            Ctprime=Ctprime,
            an=a * REWS, # Why multiply by REWS?
            u4=u4,
            v4=v4,
            REWS=REWS,
            tilt=np.deg2rad(tilt) if self.use_floris_tilt else 0.0,
            w4=w4,
            TI=RETI,
            extra=extra(a, u4, Ct, REWS)
        )
        return rotor_solution
