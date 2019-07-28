import numpy as np
import math

PI = 3.1415926

def hashkey(dh, dw, W, args):

    gy = dh.ravel()
    gx = dw.ravel()
    
    G = np.vstack((gy, gx)).T
    GTWG = G.T.dot(W).dot(G) / (args.neigborSize **2)

    ## do SVD
    w, v = np.linalg.eig(GTWG)
    idx = w.argsort()[::-1]
    S1, S2 = w[idx]
    v = v[:,idx]
    theta = math.atan2(v[1,0], v[0,0]) / PI * 180
    if theta < 0:
        theta = theta + 180
    Q_theta = math.floor(theta / 180 * args.Qangle)
 
    # Quantize strength
    # threshold: 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2
    s = math.sqrt(S1)
    if math.floor(s / 0.025) < args.Qstrength:
        Q_s = math.floor(s / 0.025)
    else:  ## restrict bound
        Q_s = args.Qstrength - 1 
    
    # Quantize mu
    # threshold: 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1
    mu = (math.sqrt(S1) - math.sqrt(S2)) / (math.sqrt(S1) + math.sqrt(S2) + 0.0001)
    if math.floor(mu / 0.125) < args.Qcoherence:
        Q_mu = math.floor(mu / 0.125)
    else:  ## restrict bound
        Q_mu = args.Qcoherence - 1
    
    
    return Q_theta, Q_s, Q_mu