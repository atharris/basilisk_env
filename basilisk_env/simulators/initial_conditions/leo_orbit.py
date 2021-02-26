from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import macros as mc

from numpy.random import uniform

def inclined_circular_300km():
    """
    Returns an inclined, circular LEO orbit.
    :return: 
    """

    mu =  0.3986004415E+15
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 300. * 1000
    oe.e = 0.0
    oe.i = 45.0 * mc.D2R

    oe.Omega = 0.0 * mc.D2R
    oe.omega = 0.0 * mc.D2R
    oe.f = 0.0 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN

def sampled_400km():
    """
    Returns an elliptical, prograde LEO orbit with an SMA of 400km.
    :return:
    """
    mu =  0.3986004415E+15
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 500. * 1000
    oe.e = uniform(0,0.05, 1)
    oe.i = uniform(-90*mc.D2R, 90*mc.D2R,1)
    oe.Omega = uniform(0*mc.D2R, 360*mc.D2R,1)
    oe.omega = uniform(0*mc.D2R, 360*mc.D2R,1)
    oe.f = uniform(0*mc.D2R, 360*mc.D2R,1)
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN

def inclined_400km():
    """
    Returns an elliptical, prograde LEO orbit with an SMA of 400km.
    :return:
    """
    mu =  0.3986004415E+15
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 500. * 1000
    oe.e = 0.0001
    oe.i = mc.D2R*45.
    oe.Omega = mc.D2R * 45.
    oe.omega = 0.0
    oe.f = 0.0
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN