import numpy as np
import math

PI = 3.1415926

def hashkey(dh, dw, W, args):

    gy = dh.ravel()
    gx = dw.ravel()
    
    G = np.vstack((gy, gx)).T
    GTWG = G.T.dot(W).dot(G) / (args.neigborSize **2)
    g00 = GTWG[0,0]
    g01 = GTWG[0,1]
    g11 = GTWG[1,1]

    # do svd for GTWG
    tmp1 = g00 + g11
    tmp2 = math.sqrt((g00-g11)**2+4*g01**2)
    S1 = (tmp1+tmp2) / 2.0
    S2 = (tmp1-tmp2) / 2.0

    # Quantize theta
    theta = 0
    if g01 != 0:
        theta = math.atan2(-g01, (g00-S1)) / PI * 180
    else:
        if g00 > g11:
            theta = 90
        else:
            theta = 0
    if theta < 0:
        theta = theta + 180
    
    Q_theta = math.floor(theta / 180 * args.Qangle)
    if Q_theta == args.Qangle:
        Q_theta = Q_theta - 1

    # Quantize strength
    # threshold: 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2
    s = math.sqrt(S1)
    if math.floor(s / 0.025) < args.Qstrength:
        Q_s = math.floor(s / 0.025)
    else:
        Q_s = args.Qstrength - 1 
    
    # Quantize mu
    # threshold: 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1
    mu = (math.sqrt(S1) - math.sqrt(S2)) / (math.sqrt(S1) + math.sqrt(S2) + 0.0001)
    if math.floor(mu / 0.125) < args.Qcoherence:
        Q_mu = math.floor(mu / 0.125)
    else:
        Q_mu = args.Qcoherence - 1
    
    
    return Q_theta, Q_s, Q_mu