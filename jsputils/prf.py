import numpy as np
import matplotlib.pyplot as plt

def normalizerange(m, targetmin, targetmax, sourcemin=None, sourcemax=None, chop=True, mode=0):
    if m.size == 0:
        return m

    if sourcemin is None:
        sourcemin = np.nanmin(m) if mode == 0 else -3
    if sourcemax is None:
        sourcemax = np.nanmax(m) if mode == 0 else 3

    if mode == 1:
        mn = np.nanmean(m)
        sd = np.nanstd(m)
        sourcemin = mn + sourcemin * sd
        sourcemax = mn + sourcemax * sd

    if sourcemin == sourcemax:
        raise ValueError("sourcemin and sourcemax are the same")

    if chop:
        m = np.clip(m, sourcemin, sourcemax)

    val = (targetmax - targetmin) / (sourcemax - sourcemin)
    f = m * val - (sourcemin * val - targetmin)
    return f

def calcunitcoordinates(res):
    return np.meshgrid(np.linspace(-0.5, 0.5, res), np.linspace(0.5, -0.5, res))

def makegaussian2d(res, r, c, sr, sc, xx=None, yy=None, ang=0, omitexp=False):
    if xx is None or yy is None:
        xx, yy = calcunitcoordinates(res)
    
    r = normalizerange(r, -.5, .5, .5, res + .5)
    c = normalizerange(c, -.5, .5, .5, res + .5)
    sr = sr / res
    sc = sc / res

    theta = np.radians(ang)
    coord = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ np.vstack([xx.flatten() - c, yy.flatten() - r])

    if sc == sr:
        f = (coord[0, :]**2 + coord[1, :]**2) / -(2 * sc**2)
    else:
        f = coord[0, :]**2 / -(2 * sc**2) + coord[1, :]**2 / -(2 * sr**2)
    
    if not omitexp:
        f = np.exp(f)

    return f.reshape(xx.shape)