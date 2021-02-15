from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import macros as mc

from numpy.random import uniform

mu =  0.3986004415E+15
def inclined_circular_300km():
    """
    Returns an inclined, circular LEO orbit.
    :return: 
    """

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
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371 * 1000.0 + 500. * 1000
    oe.e = uniform(0,0.05, 1)
    oe.i = uniform(-90*mc.D2R, 90*mc.D2R,1)
    oe.Omega = uniform(0*mc.D2R, 360*mc.D2R,1)
    oe.omega = uniform(0*mc.D2R, 360*mc.D2R,1)
    oe.f = uniform(0*mc.D2R, 360*mc.D2R,1)
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    return oe, rN, vN

def coordinated_pass_1():
    r_sc1 = (6378. + 500.) * 1000      # meters
    oe_sc1 = orbitalMotion.ClassicElements()
    oe_sc1.a = r_sc1
    oe_sc1.e = 0.00001
    oe_sc1.i = 63.0 * mc.D2R
    oe_sc1.Omega = 165.0 * mc.D2R
    oe_sc1.omega = 184.8 * mc.D2R
    oe_sc1.f = 85.3 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe_sc1)

    return oe_sc1, rN, vN

def coordinated_pass_2():
    r_sc2 = (6378. + 2000.) * 1000     # meters
    oe_sc2 = orbitalMotion.ClassicElements()
    oe_sc2.a = r_sc2
    oe_sc2.e = 0.00001
    oe_sc2.i = 63.0 * mc.D2R
    oe_sc2.Omega = 150.0 * mc.D2R
    oe_sc2.omega = 345.0 * mc.D2R
    oe_sc2.f = 85.3 * mc.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe_sc2)

    return oe_sc2, rN, vN