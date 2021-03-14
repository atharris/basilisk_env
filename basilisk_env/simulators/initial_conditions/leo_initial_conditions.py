from basilisk_env.simulators.initial_conditions import leo_orbit, sc_attitudes

import numpy as np 
from numpy.random import uniform
from Basilisk.utilities import orbitalMotion

def sampled_400km_leo_smallsat_tumble():
    # Sample orbital parameters
    oe, rN,vN = leo_orbit.sampled_400km()

    # Sample attitude and rates
    sigma_init, omega_init = sc_attitudes.random_tumble(maxSpinRate=0.00001)

    # Dict of initial conditions
    initial_conditions = {
        # Mass
        "mass": 330, # kg

        # Orbital parameters
        "oe": oe,
        "rN": rN,
        "vN": vN,

        # Spacecraft dimensions
        "width": 1.38,
        "depth": 1.04,
        "height": 1.58,

        # Attitude and rate initialization
        "sigma_init": sigma_init,
        "omega_init": omega_init,

        # Atmospheric density
        "planetRadius": orbitalMotion.REQ_EARTH * 1000.,
        "baseDensity": 1.22, #kg/m^3
        "scaleHeight": 8e3,  #m

        # Disturbance Torque
        "disturbance_magnitude": 2e-4,
        "disturbance_vector": np.random.standard_normal(3),

        # Reaction Wheel speeds
        "wheelSpeeds": uniform(-800,800,3), # RPM

        # Solar Panel Parameters
        "nHat_B": np.array([0,-1,0]),
        "panelArea": 0.2*0.3,
        "panelEfficiency": 0.20,

        # Power Sink Parameters
        "powerDraw": -5.0, # W

        # Battery Parameters
        "storageCapacity": 20.0 * 3600.,
        "storedCharge_Init": np.random.uniform(8.*3600., 20.*3600., 1)[0],

        # Sun pointing FSW config
        "sigma_R0N": [1,0,0],

        # RW motor torque and thruster force mapping FSW config
        "controlAxes_B": [1, 0, 0,
                            0, 1, 0,
                            0, 0, 1],

        # Attitude controller FSW config
        "K": 7,
        "Ki": -1.0,  # Note: make value negative to turn off integral feedback
        "P": 35,

        # Momentum dumping config
        "hs_min": 4.,  # Nms

        # Thruster force mapping FSW module
        "thrForceSign": 1,

        # Thruster momentum dumping FSW config
        "maxCounterValue": 4,
        "thrMinFireTime": 0.002 #   Seconds
    }

    return initial_conditions
def reasonable_400km_leo_smallsat_tumble():
    # Sample orbital parameters
    oe, rN,vN = leo_orbit.inclined_400km()

    # Sample attitude and rates
    sigma_init, omega_init = sc_attitudes.static_inertial()

    # Dict of initial conditions
    initial_conditions = {
        # Mass
        "mass": 330, # kg

        # Orbital parameters
        "oe": oe,
        "rN": rN,
        "vN": vN,

        # Spacecraft dimensions
        "width": 1.38,
        "depth": 1.04,
        "height": 1.58,

        # Attitude and rate initialization
        "sigma_init": sigma_init,
        "omega_init": omega_init,

        # Atmospheric density
        "planetRadius": orbitalMotion.REQ_EARTH * 1000.,
        "baseDensity": 1.22, #kg/m^3
        "scaleHeight": 8e3,  #m

        # Disturbance Torque
        "disturbance_magnitude": 2e-4,
        "disturbance_vector": [1,1,1] / np.sqrt([3]), #  unit vector in 1,1,1 direction

        # Reaction Wheel speeds
        "wheelSpeeds": [400,400,400], # RPM

        # Solar Panel Parameters
        "nHat_B": np.array([0,-1,0]),
        "panelArea": 0.2*0.3,
        "panelEfficiency": 0.20,

        # Power Sink Parameters
        "powerDraw": -5.0, # W

        # Battery Parameters
        "storageCapacity": 20.0 * 3600.,
        "storedCharge_Init": 15. * 3600.,

        # Sun pointing FSW config
        "sigma_R0N": [1,0,0],

        # RW motor torque and thruster force mapping FSW config
        "controlAxes_B": [1, 0, 0,
                            0, 1, 0,
                            0, 0, 1],

        # Attitude controller FSW config
        "K": 7,
        "Ki": -1.0,  # Note: make value negative to turn off integral feedback
        "P": 35,

        # Momentum dumping config
        "hs_min": 4.,  # Nms

        # Thruster force mapping FSW module
        "thrForceSign": 1,

        # Thruster momentum dumping FSW config
        "maxCounterValue": 4,
        "thrMinFireTime": 0.002 #   Seconds
    }

    return initial_conditions