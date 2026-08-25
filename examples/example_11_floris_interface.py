"""
Demonstrates the use of the FlorisCurledWindfarm interface to call the CurledWindfarm
wake model from the FLORIS wake modeling framework (https://github.com/NatLabRockies/floris/).

The example compares the results of simulating a wind farm using:
- The CurledWindfarm wake model directly
- The CurledWindfarm wake model called via the FlorisCurledWindfarm interface
- The "gauss" wake model in FLORIS

Additionally, the example demonstrates how to set yaw and tilt angles for the turbines in the farm.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane
from floris.layout_visualization import plot_turbine_rotors

from mitwindfarm import FlorisCurledWindfarm, Layout, Plotting, PowerLaw
from mitwindfarm.Rotor import UnifiedAD_TI
from mitwindfarm.windfarm import CurledWindfarm

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

def run_model_comparison(yaw_angle=0.0, tilt=False):

    # Turbine parameters (for normalization)
    D = 242.24
    H = 150.0

    # Sample points for querying the flow field
    samples_x = np.array([1000, 1000, 2000, 2000])
    samples_y = np.array([0, 100, 0, 100])
    samples_z = np.array([H, H, H, H])

    # Flow parameters
    wind_shear = 0.0
    TI = 0.06
    U = 8.0
    rotation_angle = -5 # First wind direction is 270; second is 270 + rotation_angle

    # Create figure for placing plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    fig.suptitle(f"Wind farm comparison: yaw={yaw_angle} deg, tilt={tilt}")

    # CurledWindfarm parameters
    layout = Layout([0, 12 / 2, 24 / 2], [0, 0, 0], [0, 0, 0])

    solver_kwargs = {
        "dy": 1 / 10,
        "dz": 1 / 10,
        "integrator": "scipy_rk23",
        "k_model": "k-l",
        "verbose": False,
    }

    # (Ct-prime, yaw, tilt)
    setpoints = [
        (2, np.deg2rad(yaw_angle), np.deg2rad(6) if tilt else 0),
        (2, np.deg2rad(yaw_angle), np.deg2rad(6) if tilt else 0),
        (2, np.deg2rad(yaw_angle), np.deg2rad(6) if tilt else 0),
    ]

    windfarm = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=PowerLaw(Uref=1.0, zref=H/D, exp=wind_shear, TIamb=TI),
        solver_kwargs=solver_kwargs,
        TIamb=TI,
    )

    windfarm_sol = windfarm(layout, setpoints)
    windfarm_sol_rotated = windfarm(layout.rotate(rotation_angle), setpoints)

    Plotting.plot_windfarm(windfarm_sol, axes[0, 0], vmin=0, vmax=2) # Left plots
    Plotting.plot_windfarm(windfarm_sol_rotated, axes[1, 0], vmin=0, vmax=2)
    axes[0, 0].scatter(samples_x/D, samples_y/D, color="k", marker=".")

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
    axes[1, 0].scatter(rotated_x, rotated_y, color="k", marker=".")
    axes[0, 0].set_title("Direct CurledWindfarm call")

    windfarm_v_rel_rot = windfarm_sol_rotated.windfield.wsp(rotated_x, rotated_y, (samples_z-H)/D)
    windfarm_v_rel = np.vstack([windfarm_v_rel, windfarm_v_rel_rot])
    print("\nDirect CurledWindfarm sampled velocities [m/s]:")
    print(windfarm_v_rel * U)

    # Establish Floris model. Note that FLORIS alone will tilt the rotor according to
    # the shaft tilt angle.
    fmodel = FlorisModel("defaults") # Defaults to the "gauss" wake model in FLORIS
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
    fmodel.run()
    powers_gauss = fmodel.get_turbine_powers()
    floris_visualization(
        fmodel, H, rotation_angle, samples_x, samples_y, "FLORIS Gauss wake model",
        axes[0, 2], axes[1, 2] # Right plots
    )

    # Assign MITWindfarm wake model, rerun
    fmodel.set_wake_model(FlorisCurledWindfarm(
        solver_kwargs=solver_kwargs,
        use_floris_tilt=tilt,
    ))
    fmodel.run()
    powers_cwf = fmodel.get_turbine_powers()
    floris_visualization(
        fmodel, H, rotation_angle, samples_x, samples_y, "CurledWindfarm via FLORIS",
        axes[0, 1], axes[1, 1] # Middle plots
    )
    floris_vels = fmodel.sample_flow_at_points(samples_x, samples_y, samples_z)

    print("\nFLORIS sampled velocities [m/s]:")
    print(floris_vels)

    print("\nRelative velocity differences (MITWindfarm - FLORIS) [%]:")
    print((windfarm_v_rel * U - floris_vels) / (windfarm_v_rel * U) * 100)

    print("\nFLORIS Gauss turbine powers [MW]:")
    print(powers_gauss / 1e6)

    print("\nFLORIS CurledWindfarm turbine powers [MW]:")
    print(powers_cwf / 1e6)

    tilt_append = "_tilt" if tilt else ""
    fig.savefig(
        FIGDIR / f"{Path(__file__).stem}_yaw_{yaw_angle}{tilt_append}.png", bbox_inches="tight"
    )

def floris_visualization(fmodel, H, rotation_angle, samples_x, samples_y, title, ax0, ax1):
    cmap_floris = "pink"
    # Extracting and plotting FLORIS results
    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=H,
        findex_for_viz=0,
    )
    plot_turbine_rotors(fmodel, ax=ax0)
    visualize_cut_plane(
        horizontal_plane,
        ax=ax0,
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
    plot_turbine_rotors(
        fmodel, ax=ax1, yaw_angles=[-rotation_angle]*len(fmodel.layout_x)
    )
    visualize_cut_plane(
        horizontal_plane,
        ax=ax1,
        cmap=cmap_floris,
        clevels=100,
        levels=[],
    )

    ax0.scatter(samples_x, samples_y, color="k", marker=".")
    ax1.scatter(samples_x, samples_y, color="k", marker=".")
    ax0.set_title(title)


if __name__ == "__main__":
    # Formatting for printing arrays
    np.set_printoptions(precision=3, suppress=True)

    # Not yawed, not tilted
    print("="*30+"\nNo yaw, remove tilt\n"+"="*30)
    run_model_comparison(yaw_angle=0.0, tilt=False)

    # Include yaw and tilt
    print("\n\n"+"="*30+"\nYawed 25 degrees, include tilt\n"+"="*30)
    run_model_comparison(yaw_angle=25.0, tilt=True)
    plt.close()
