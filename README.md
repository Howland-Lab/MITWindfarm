# MITWindfarm
This repository is a steady-state wake solver which couples aerodynamic rotor models based on momentum theory with turbulent far-wake models. Rotor models can couple to blade-element momentum (BEM) theory to output realistic set points of pitch, tip-speed ratio, and yaw for wind turbine control. 


# Installation
To install this Python package follow one of the following methods.

### Direct installation from Github
To install directly from Github into the current Python environment, run:
```bash
pip install git+https://github.com/Howland-Lab/MITWindfarm.git
```


### Install from cloned repository
If you prefer to download the repository first (for example, to run the example and paper figure scripts), you can first clone the repository, either using http:
```bash
git clone https://github.com/Howland-Lab/MITWindfarm.git
```
or ssh:
```bash
git clone git@github.com:Howland-Lab/MITWindfarm.git
```
then, install locally using pip using `pip install .` for the base installation, or `pip install .[examples]` to install the extra dependencies required to run the examples scripts scripts. To include developer packages in the installation, use `pip install -e .[dev]`. 

```bash
cd MITWindfarm
pip install .
```