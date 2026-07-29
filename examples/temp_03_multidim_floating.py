import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel, ParFlorisModel
from floris.flow_visualization import visualize_cut_plane
from floris.layout_visualization import plot_turbine_rotors
from MITRotor import IEA15MW

from mitwindfarm import FlorisCurledWindfarm, Layout, Plotting, PowerLaw
from mitwindfarm.Rotor import UnifiedAD_TI
from mitwindfarm.windfarm import CurledWindfarm

# Set up MITWindfarm solver and run standalone as a baseline

if __name__ == "__main__":

    cmap_floris = "pink"
    use_parallel_model = False
    tilt_rotor_in_wake_model = True

    # (Ct-prime, yaw, tilt)
    setpoints = [
        (2, 0, 0),
        (2, 0, 0),
        (2, 0, 0),
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

    # Establish Floris model
    if use_parallel_model:
        fmodel = ParFlorisModel("defaults")
    else:
        fmodel = FlorisModel("defaults")

    multidim_conditions = {
       "Tp": np.array([2.5]*2),
        "Hs": np.array([4.1, 1.0]),
    }
    fmodel.set(
        layout_x=layout.x * D,
        layout_y=layout.y * D,
        wind_speeds=[U, U],
        wind_directions=[270.0, 270.0], 
        turbulence_intensities=[TI, TI],
        turbine_type=["iea_15MW_floating_multi_dim_cp_ct"]*len(layout.x),
        reference_wind_height=H, # IEA 15MW hub height
        wind_shear=wind_shear,
        multidim_conditions=multidim_conditions
    )
    # Assign MITWindfarm wake model
    fmodel.set_wake_model(FlorisCurledWindfarm(
        solver_kwargs=solver_kwargs,
        use_floris_tilt=tilt_rotor_in_wake_model
    ))
    # Run FLORIS using MITWindfarm wake model/solver, get turbine powers
    fmodel.run()
    powers = fmodel.get_turbine_powers()

    # Extracting and plotting FLORIS results
    fig, axes_floris = plt.subplots(1)
    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=H,
        findex_for_viz=0,
    )
    plot_turbine_rotors(fmodel, ax=axes_floris)
    visualize_cut_plane(
        horizontal_plane,
        ax=axes_floris,
        cmap=cmap_floris,
        clevels=100,
        levels=[],
    )
    
    # Sample at specific points
    samples_x = np.array([1000, 1000, 2000, 2000])
    samples_y = np.array([0, 100, 0, 100])
    samples_z = np.array([H, H, H, H])

    axes_floris.scatter(samples_x, samples_y, color="k", marker=".")
    fig.suptitle("Called via FLORIS")

    floris_vels = fmodel.sample_flow_at_points(samples_x, samples_y, samples_z)

    print("\nFLORIS sampled velocities:")
    print(floris_vels)

    plt.show()
