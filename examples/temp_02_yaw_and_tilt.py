import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane
from floris.layout_visualization import plot_turbine_rotors
from MITRotor import IEA15MW

from mitwindfarm import FlorisCurledWindfarm, Layout, Plotting, PowerLaw
from mitwindfarm.Rotor import UnifiedAD_TI
from mitwindfarm.windfarm import CurledWindfarm

# Set up MITWindfarm solver and run standalone as a baseline

if __name__ == "__main__":

    cmap_floris = "pink"

    rotation_angle = -5
    tilt = False
    yaw_angle = 25

    # (Ct-prime, yaw, tilt)
    setpoints = [
        (2, np.deg2rad(yaw_angle), np.deg2rad(6) if tilt else 0),
        (2, np.deg2rad(yaw_angle), np.deg2rad(6) if tilt else 0),
        (2, np.deg2rad(yaw_angle), np.deg2rad(6) if tilt else 0),
    ]
        

    D = 242.24
    H = 150.0
    wind_shear = 0.0
    TI = 0.06
    U = 8.0
    layout = Layout([0, 12 / 2, 24 / 2], [0, 0, 0], [0, 0, 0])

    solver_kwargs = dict(
        dy=1 / 10,
        dz=1 / 10,
        integrator="scipy_rk23",  # see mitwindfarm.utils.integrate
        k_model="k-l",  # alternatives: "const", "2021"
        verbose=False,
    )

    windfarm = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=PowerLaw(Uref=1.0, zref=H/D, exp=wind_shear, TIamb=TI),
        solver_kwargs=solver_kwargs,
        TIamb=TI,
    )

    windfarm_sol = windfarm(layout, setpoints)
    windfarm_sol_rotated = windfarm(layout.rotate(rotation_angle), setpoints)

    fig, axes_windfarm = plt.subplots(2)
    Plotting.plot_windfarm(windfarm_sol, axes_windfarm[0], vmin=0, vmax=2)
    Plotting.plot_windfarm(windfarm_sol_rotated, axes_windfarm[1], vmin=0, vmax=2)
    fig.suptitle("Direct call")

    # Establish Floris model
    fmodel = FlorisModel("defaults")
    fmodel.set(
        layout_x=layout.x * D,
        layout_y=layout.y * D,
        wind_speeds=[U, U],
        wind_directions=[270.0, 270.0 + rotation_angle], 
        turbulence_intensities=[TI, TI],
        turbine_type=["iea_15MW"]*len(layout.x),
        reference_wind_height=H, # IEA 15MW hub height
        wind_shear=wind_shear,
        yaw_angles=yaw_angle*np.ones((2, 3)),
    )
    # Assign MITWindfarm wake model
    fmodel.set_wake_model(FlorisCurledWindfarm(
        solver_kwargs=solver_kwargs,
        use_floris_tilt=tilt
    ))
    # Run FLORIS using MITWindfarm wake model/solver, get turbine powers
    fmodel.run()
    powers = fmodel.get_turbine_powers()

    # Extracting and plotting FLORIS results
    fig, axes_floris = plt.subplots(2)
    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=H,
        findex_for_viz=0,
    )
    plot_turbine_rotors(fmodel, ax=axes_floris[0])
    visualize_cut_plane(
        horizontal_plane,
        ax=axes_floris[0],
        cmap=cmap_floris,
        clevels=100,
        levels=[],
    )
    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=H,
        findex_for_viz=1,
    )
    plot_turbine_rotors(fmodel, ax=axes_floris[1], yaw_angles=[-rotation_angle]*len(layout.x))
    visualize_cut_plane(
        horizontal_plane,
        ax=axes_floris[1],
        cmap=cmap_floris,
        clevels=100,
        levels=[],
    )

    # Sample at specific points
    samples_x = np.array([1000, 1000, 2000, 2000])
    samples_y = np.array([0, 100, 0, 100])
    samples_z = np.array([H, H, H, H])

    axes_floris[0].scatter(samples_x, samples_y, color="k", marker=".")
    axes_floris[1].scatter(samples_x, samples_y, color="k", marker=".")
    axes_windfarm[0].scatter(samples_x/D, samples_y/D, color="k", marker=".")
    fig.suptitle("Called via FLORIS")

    floris_vels = fmodel.sample_flow_at_points(samples_x, samples_y, samples_z)

    print("\nFLORIS sampled velocities:")
    print(floris_vels)

    print("\nMITWindfarm sampled velocities:")
    windfarm_sol = windfarm(layout, setpoints)
    windfarm_v_rel = windfarm_sol.windfield.wsp(samples_x/D, samples_y/D, (samples_z-H)/D)
    windfarm_sol_rotated = windfarm(layout.rotate(rotation_angle), setpoints)

    rotated_x = (
        (samples_x/D - 6) * np.cos(np.radians(rotation_angle))
        - (samples_y/D - 0) * np.sin(np.radians(rotation_angle))
        + 6
    )
    rotated_y = (
        (samples_y/D - 0) * np.cos(np.radians(rotation_angle))
        + (samples_x/D - 6) * np.sin(np.radians(rotation_angle))
        + 0
    )
    axes_windfarm[1].scatter(rotated_x, rotated_y, color="k", marker=".")

    windfarm_v_rel_rot = windfarm_sol_rotated.windfield.wsp(rotated_x, rotated_y, (samples_z-H)/D)
    windfarm_v_rel = np.vstack([windfarm_v_rel, windfarm_v_rel_rot])
    print(windfarm_v_rel * U)

    print("\nRelative velocity differences (MITWindfarm - FLORIS) %:")
    print((windfarm_v_rel * U - floris_vels) / (windfarm_v_rel * U) * 100)

    print(powers / 1e6)

    # Generate plots
    plt.show()
