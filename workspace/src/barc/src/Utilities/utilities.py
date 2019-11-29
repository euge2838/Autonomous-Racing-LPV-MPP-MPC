import numpy as np
import pdb
def Regression(x, u, lamb):
    """Estimates linear system dynamics
    x, u: date used in the regression
    lamb: regularization coefficient
    """

    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    Y = x[2:x.shape[0], :]
    X = np.hstack((x[1:(x.shape[0] - 1), :], u[1:(x.shape[0] - 1), :]))

    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)

    A = W.T[:, 0:6]
    B = W.T[:, 6:8]

    ErrorMatrix = np.dot(X, W) - Y
    ErrorMax = np.max(ErrorMatrix, axis=0)
    ErrorMin = np.min(ErrorMatrix, axis=0)
    Error = np.vstack((ErrorMax, ErrorMin))

    return A, B, Error



# EA: Modified for taking also the desired velocity
def Curvature(s, PointAndTangent):
    """curvature and desired velocity computation
    s: curvilinear abscissa at which the curvature has to be evaluated
    PointAndTangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
    """
    TrackLength = PointAndTangent[-1,3]+PointAndTangent[-1,4]

    # In case on a lap after the first one
    while (s > TrackLength):
        s = s - TrackLength

    # Given s \in [0, TrackLength] compute the curvature
    # Compute the segment in which system is evolving
    index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)

    i = int(np.where(np.squeeze(index))[0]) #EA: this works 
    #i = np.where(np.squeeze(index))[0]     #EA: this does not work
    curvature = PointAndTangent[i, 5]

    return curvature



def GBELLMF(x, a, b, c):

    tmp = ((x - c)/a)**2

    if (tmp == 0) and (b == 0):
        y = 0.5
    elif (tmp == 0) and (b < 0):
        y = 0
    else:
        #tmp = tmp**b
        tmp = np.power(tmp,b)
        y = 1.0/(1 + tmp)
    
    return y



