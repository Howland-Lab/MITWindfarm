from pathlib import Path

import numpy as np
from MITRotor.ReferenceTurbines import IEA15MW

from mitwindfarm.Rotor import AD, BEM, UnifiedAD
from mitwindfarm.Windfield import Uniform

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    windfield = Uniform()
    rotor_ad = AD()
    sol_ad = rotor_ad(0, 0, 0, windfield, 2, np.deg2rad(10))
    print(sol_ad)

    rotor_unified = UnifiedAD()
    sol_unified = rotor_unified(0, 0, 0, windfield, 2, np.deg2rad(10))
    print(sol_unified)

    rotor_bem = BEM(IEA15MW())
    sol_bem = rotor_bem(0, 0, 0, windfield, 0, 9, np.deg2rad(10))
    print(sol_bem)
