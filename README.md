# basilisk_env: A Basilisk-based Astrodynamics Reinforcement Learning Problems Package

basilisk_env is a package of reference planning and decision-making problems built on both the AVS Basilisk simulation 
engine (https://hanspeterschaub.info/basilisk/) and using the OpenAI gym API (https://gym.openai.com/)
##	Getting Started

These instructions will give you the information needed to install and use gym_orbit.

### Prerequsites
gym_orbit requires the following Python packages:

- Numerics and plotting: numpy, scipy, matplotlib, pandas
- Spacecraft Simulation: Basilisk
- Machine Learning + Agent Libraries: tensorflow, keras, gym, stable-baselines

### Installation

To install the package, run:

```
pip install -e .
```

while in the base directory. Test the installation by opening a Python terminal and calling

```
import gym
import basilisk_env
```

Optical Navigation scenarios are run with the Basilisk Vizard visualization suite. This can be downloaded from (http://hanspeterschaub.info/basilisk/Vizard/VizardDownload.html).

To use Vizard with basilisk-env, you must set up environment variables corresponding to the vizard application directory and address/port ('viz_app','viz_address', and 'viz_port', respectively). For example:

```
$ export viz_app = '/mnt/c/Users/aharr/OneDrive/dev/Vizard/vizard20201211L/Vizard.exe'
$ export viz_port = '5556'
$ export viz_address = '172.20.128.1'
```

## Repository Structure

`basilisk_env` is broken into two principal components: `/envs`, which contains environment wrappers for Basilisk, 
and `simulators`, which contains Basilisk simulation definitions for various scenarios. 

Within `simulators`, core simulation routines representing interchangable dynamics and flight software stacks
are stored in `simulators/dynamics` and `simulators/fsw` respectively. 


###	Contributions



## Authors
Maintainer: Mr. Andrew T. Harris (andrew.t.harris@colorado.edu)

Mr. Thibaud Teil (Thibaud.Teil@colorado.edu)

##	License

This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

[The Autonomous Vehicle Systems Laboratory](http://hanspeterschaub.info/main.html)
