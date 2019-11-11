from Basilisk.simulation import thrusterDynamicEffector, reactionWheelStateEffector
from Basilisk.utilities import simIncludeThruster, simIncludeRW
from Basilisk.utilities import fswSetupThrusters, fswSetupRW
from numpy.random import uniform

def balancedHR16Triad(useRandom = False, randomBounds = (-600, 600)):
    """
        Creates a set of eight ADCS thrusters using MOOG Monarc-1 attributes.
        Returns a set of thrusters and thrusterFac instance to add thrusters to a spacecraft.
        @return thrusterSet: thruster dynamic effector instance
        @return thrusterFac: factory containing defined thrusters
    """
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = rwFactory.BalancedWheels
    if useRandom:
        W1 = rwFactory.create('Honeywell_HR16'
                              , [1, 0, 0]
                              , maxMomentum=50.
                              , Omega=uniform(randomBounds[0],randomBounds[1])  # RPM
                              , RWModel=varRWModel
                              )
        RW2 = rwFactory.create('Honeywell_HR16'
                               , [0, 1, 0]
                               , maxMomentum=50.
                               , Omega=uniform(randomBounds[0],randomBounds[1])  # RPM
                               , RWModel=varRWModel
                               )
        RW3 = rwFactory.create('Honeywell_HR16'
                               , [0, 0, 1]
                               , maxMomentum=50.
                               , Omega=uniform(randomBounds[0],randomBounds[1]) # RPM
                               , rWB_B=[0.5, 0.5, 0.5]  # meters
                               , RWModel=varRWModel
                               )
    else:
        RW1 = rwFactory.create('Honeywell_HR16'
                               , [1, 0, 0]
                               , maxMomentum=50.
                               , Omega=100.  # RPM
                               , RWModel=varRWModel
                               )
        RW2 = rwFactory.create('Honeywell_HR16'
                               , [0, 1, 0]
                               , maxMomentum=50.
                               , Omega=200.  # RPM
                               , RWModel=varRWModel
                               )
        RW3 = rwFactory.create('Honeywell_HR16'
                               , [0, 0, 1]
                               , maxMomentum=50.
                               , Omega=300.  # RPM
                               , rWB_B=[0.5, 0.5, 0.5]  # meters
                               , RWModel=varRWModel
                               )
    numRW = rwFactory.getNumOfDevices()
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()

    return rwStateEffector, rwFactory


def idealMonarc1Octet():
    """
        Creates a set of eight ADCS thrusters using MOOG Monarc-1 attributes.
        Returns a set of thrusters and thrusterFac instance to add thrusters to a spacecraft.
        @return thrusterSet: thruster dynamic effector instance
        @return thrusterFac: factory containing defined thrusters
    """
    location = [
        [
            3.874945160902288e-2,
            -1.206182747348013,
            0.85245
        ],
        [
            3.874945160902288e-2,
            -1.206182747348013,
            -0.85245
        ],
        [
            -3.8749451609022656e-2,
            -1.206182747348013,
            0.85245
        ],
        [
            -3.8749451609022656e-2,
            -1.206182747348013,
            -0.85245
        ],
        [
            -3.874945160902288e-2,
            1.206182747348013,
            0.85245
        ],
        [
            -3.874945160902288e-2,
            1.206182747348013,
            -0.85245
        ],
        [
            3.8749451609022656e-2,
            1.206182747348013,
            0.85245
        ],
        [
            3.8749451609022656e-2,
            1.206182747348013,
            -0.85245
        ]
    ]

    direction = [
        [
            -0.7071067811865476,
            0.7071067811865475,
            0.0
        ],
        [
            -0.7071067811865476,
            0.7071067811865475,
            0.0
        ],
        [
            0.7071067811865475,
            0.7071067811865476,
            0.0
        ],
        [
            0.7071067811865475,
            0.7071067811865476,
            0.0
        ],
        [
            0.7071067811865476,
            -0.7071067811865475,
            0.0
        ],
        [
            0.7071067811865476,
            -0.7071067811865475,
            0.0
        ],
        [
            -0.7071067811865475,
            -0.7071067811865476,
            0.0
        ],
        [
            -0.7071067811865475,
            -0.7071067811865476,
            0.0
        ]
    ]
    thrusterSet = thrusterDynamicEffector.ThrusterDynamicEffector()
    thFactory = simIncludeThruster.thrusterFactory()
    for pos_B, dir_B in zip(location, direction):
        thFactory.create('MOOG_Monarc_1', pos_B, dir_B)

    thrModelTag = "ACSThrusterDynamics"
    return thrusterSet, thFactory