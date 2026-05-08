from attrs import define, field

import numpy as np

from floris.core.wake_model import BaseWakeModel

from mitwindfarm import Layout

@define
class FlorisWakeModel(BaseWakeModel):
    # Parameters (to fill in)
    windfarm = field(default=None)

    def turbine_solve(self, farm, flow_field, grid):
        self._solve_and_evaluate(farm, flow_field, grid, grid)

    def point_solve(self, farm, flow_field, grid):
        turbine_grid = self.generate_turbine_grid_objects(farm, flow_field)[2]
        self._solve_and_evaluate(farm, flow_field, grid, turbine_grid)

    def _solve_and_evaluate(self, farm, flow_field, grid, turbine_grid):
        # Assume all turbines have the same rotor diameter; not sure if
        # methods below handle varying rotor diameters?
        if not np.all(farm.rotor_diameters == farm.rotor_diameters[0]):
            raise NotImplementedError("Varying rotor diameters not supported in FlorisWakeModel")
        else:
            D = farm.rotor_diameters[0]

        # Use sorted version
        for f in range(flow_field.n_findex):
            
            turbines_x = turbine_grid.x_sorted.mean(axis=(2,3))[f]
            turbines_y = turbine_grid.y_sorted.mean(axis=(2,3))[f]
            layout = Layout(turbines_x/D, turbines_y/D)
            # Extract quantities for this findex
            wd = flow_field.wind_directions[f]
            ws = flow_field.wind_speeds[f] # Single ws for now. Use u_initial_sorted.
            ti = flow_field.turbulence_intensities[f]

            yaw = farm.yaw_angles[f, :]
            #tilt = farm.tilt_angles[f, :]
            tilt = np.zeros_like(yaw) # Temporary
            pitch = 0.0 * np.ones_like(yaw)
            tsr = 7.0 * np.ones_like(yaw)

            # Create Windfarm calling arguments
            setpoints = list(zip(pitch, tsr, yaw, tilt))

            # Solve for the current findex
            windfarm_sol = self.windfarm(layout, setpoints)

            # Extract the wind speeds at the turbine locations
            relative_velocities = windfarm_sol.windfield.wsp(
                grid.x_sorted[f]/D, grid.y_sorted[f]/D, grid.z_sorted[f]/D
            )

            # Assign to flow field
            # (sorting may be an issue here. May need to reassign layout each time).
            flow_field.u_sorted[f] = relative_velocities * ws
