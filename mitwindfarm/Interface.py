from attrs import define, field

import numpy as np

from floris.core.wake_model import BaseWakeModel

from .Windfield import PowerLaw
from ._Layout import Layout
from .windfarm import Windfarm, CosineWindfarm, CurledWindfarm

@define
class FlorisWakeModel(BaseWakeModel):
    # Parameters (to fill in)
    windfarm = field(default=None)

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


    def turbine_solve(self, farm, flow_field, grid):
        self._solve_and_evaluate(farm, flow_field, grid, grid)

    def point_solve(self, farm, flow_field, grid):
        turbine_grid = self.generate_turbine_grid_objects(farm, flow_field)[2]
        self._solve_and_evaluate(farm, flow_field, grid, turbine_grid)

    def _solve_and_evaluate(self, farm, flow_field, grid, turbine_grid):
        # Assume all turbines have the same rotor diameter; not sure if
        # methods below handle varying rotor diameters?
        if not np.all(farm.rotor_diameters == farm.rotor_diameters.mean()):
            raise NotImplementedError("Varying rotor diameters not supported in FlorisWakeModel")
        else:
            D = farm.rotor_diameters.mean()

        # Use sorted version
        for f in range(flow_field.n_findex):
            
            turbines_x = turbine_grid.x_sorted.mean(axis=(2,3))[f]
            turbines_y = turbine_grid.y_sorted.mean(axis=(2,3))[f]
            turbines_z = turbine_grid.z_sorted.mean(axis=(2,3))[f]
            layout = Layout(turbines_x/D, turbines_y/D, turbines_z/D)

            # Generate calling arguments based on windfarm type. Also depends on rotor model; not yet handled.
            # Sometimes, setpoints should include tsr and pitch; other times, ctprime? Depends on rotor model?
            wf_init_kwargs = {
                "rotor_model": self.windfarm.rotor_model, # TODO: Pass FLORIS-like wrapper here?
                "base_windfield": PowerLaw( # TODO: Can we pass a more general wind field?
                    flow_field.wind_speeds[f],
                    flow_field.reference_wind_height/D,
                    flow_field.wind_shear,
                    flow_field.turbulence_intensities[f]
                ),
                "TIamb": flow_field.turbulence_intensities[f], # Needed? not sure
                "solver_kwargs": self.windfarm.solver_kwargs,
            }
            yaw = farm.yaw_angles[f, :]
            tilt = np.zeros_like(yaw) # Temporary
            CTprime = 2.0 * np.ones_like(yaw) # Temporary
            setpoints = list(zip(CTprime, yaw, tilt))

            # Reinstantiate and solve for the current findex
            self.windfarm = self.windfarm.__class__(**wf_init_kwargs)
            windfarm_sol = self.windfarm(layout, setpoints)

            # Extract the wind speeds at the turbine locations
            relative_velocities = windfarm_sol.windfield.wsp(
                grid.x_sorted[f]/D, grid.y_sorted[f]/D, grid.z_sorted[f]/D
            )

            # Assign to flow field
            flow_field.u_sorted[f] = relative_velocities

# TODO
# Create Rotor-style wrapper for Floris operation_model so that that can be called instead?
# How to pass tilt, etc in?
# How to pass yaw angle?
