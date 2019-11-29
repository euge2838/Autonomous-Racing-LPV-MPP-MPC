#!/usr/bin/env python
"""
    File name: stateEstimator.py
    Author: Shuqi Xu and Ugo Rosolia
    Email: shuqixu@berkeley.edu (xushuqi8787@gmail.com)
    Modified: Eugenio Alcala
    Email: eugenio.alcala@upc.edu
    Date: 09/30/2018
    Python Version: 2.7.12
"""
from scipy import linalg, sparse
import numpy as np
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from utilities import Curvature, GBELLMF
import datetime
import numpy as np
from numpy import linalg as la
import pdb
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from numpy import tan, arctan, cos, sin, pi
import rospy

solvers.options['show_progress'] = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class PathFollowingLPV_MPC:
    """Create the Path Following LMPC controller with LTV model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, Q, R, dR, N, vt, dt, map, Solver, steeringDelay, velocityDelay):

        # Vehicle parameters:
        self.lf = rospy.get_param("lf")
        self.lr = rospy.get_param("lr")
        self.m  = rospy.get_param("m")
        self.I  = rospy.get_param("Iz")
        self.Cf = rospy.get_param("Cf")
        self.Cr = rospy.get_param("Cr")
        self.mu = rospy.get_param("mu")

        self.g  = 9.81

        self.max_vel = rospy.get_param("/TrajectoryPlanner/max_vel")     
        # self.max_vel = self.max_vel - 0.2*self.max_vel     

        self.A    = []
        self.B    = []
        self.C    = []
        self.N    = N
        self.n    = Q.shape[0]
        self.d    = R.shape[0]
        self.vt   = vt
        self.Q    = Q
        self.R    = R
        self.dR   = dR              # Slew rate
        self.LinPoints = np.zeros((self.N+2,self.n))
        self.dt = dt                # Sample time 33 ms
        self.map = map              # Used for getting the road curvature
        self.halfWidth = map.halfWidth

        self.first_it = 1

        self.steeringDelay = steeringDelay
        self.velocityDelay = velocityDelay

        self.OldSteering = [0.0]*int(1 + steeringDelay)

        self.OldAccelera = [0.0]*int(1)        

        self.OldPredicted = [0.0]*int(1 + steeringDelay + N)

        self.Solver = Solver

        self.F, self.b = _buildMatIneqConst(self)

        self.G = []
        self.E = []
        self.L = []
        self.Eu =[]      
        



    def solve(self, x0, Last_xPredicted, uPred, NN_LPV_MPC, vel_ref, A_L, B_L ,C_L, first_it):
        """Computes control action
        Arguments:
            x0: current state position
            EA: Last_xPredicted: it is just used for the warm up
            EA: uPred: set of last predicted control inputs used for updating matrix A LPV
            EA: A_L, B_L ,C_L: Set of LPV matrices
        """
        startTimer              = datetime.datetime.now()

        if (NN_LPV_MPC == False) and (first_it < 10):
            self.A, self.B, self.C  = _EstimateABC(self, Last_xPredicted, uPred)   
        else:
            self.A = A_L
            self.B = B_L
            self.C = C_L



        self.G, self.E, self.L, self.Eu  = _buildMatEqConst(self) # It's been introduced the C matrix (L in the output)

        self.M, self.q          = _buildMatCost(self, uPred[0,:], vel_ref)


        endTimer                = datetime.datetime.now()
        deltaTimer              = endTimer - startTimer
        self.linearizationTime  = deltaTimer

        M = self.M
        q = self.q
        G = self.G
        E = self.E
        L = self.L
        Eu= self.Eu
        F = self.F
        b = self.b
        n = self.n
        N = self.N
        d = self.d

        uOld  = [self.OldSteering[0], self.OldAccelera[0]]

        
        if self.Solver == "CVX":
            startTimer = datetime.datetime.now()
            sol = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0))
            endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
            self.solverTime = deltaTimer
            if sol['status'] == 'optimal':
                self.feasible = 1
            else:
                self.feasible = 0

            self.xPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[np.arange(n * (N + 1))]), (N + 1, n))))
            self.uPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[n * (N + 1) + np.arange(d * N)]), (N, d))))
        else:
            startTimer = datetime.datetime.now()

            res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F),
             b, sparse.csr_matrix(G), np.add( np.dot(E,x0),L[:,0],np.dot(Eu,uOld) ) )
            
            if feasible == 0:
                print 'QUIT...'

            Solution = res_cons.x

            endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
            self.solverTime = deltaTimer
            self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n * (N + 1))]), (N + 1, n))))
            self.uPred = np.squeeze(np.transpose(np.reshape((Solution[n * (N + 1) + np.arange(d * N)]), (N, d))))

        self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )
        self.xPred = self.xPred.T
        self.uPred = self.uPred.T



    def LPVPrediction(self, x, u, vel_ref, curv_ref, Cf_new, LapNumber):

        lf  = self.lf
        lr  = self.lr
        m   = self.m
        I   = self.I
        Cf  = Cf_new
        Cr  = Cf_new       
        mu  = self.mu
        g   = self.g

        STATES_vec = np.zeros((self.N, 6))

        Atv = []
        Btv = []
        Ctv = []

        for i in range(0, self.N):

            if i==0:
                states  = np.reshape(x, (6,1))

            vy      = float(states[1])
            epsi    = float(states[3])
            s       = float(states[4])
            ey      = float(states[5])

            if LapNumber == 0:       # First lap at 1 m/s       
                PointAndTangent = self.map.PointAndTangent         
                cur     = Curvature(s, PointAndTangent)

            else:
                cur     = float(curv_ref[i])             

            vx      = float(vel_ref[i])  
            delta   = float(u[i,0])            # EA: steering angle at K-1

            A11 =  -mu
            A12 = (np.sin(delta) * Cf) / (m*vx)
            A13 = (np.sin(delta) * Cf * lf) / (m*vx) + vy
            A22 = -(Cr + Cf * np.cos(delta)) / (m*vx)
            A23 = -(lf * Cf * np.cos(delta) - lr * Cr) / (m*vx) - vx
            A32 = -(lf * Cf * np.cos(delta) - lr * Cr) / (I*vx)
            A33 = -(lf * lf * Cf * np.cos(delta) + lr * lr * Cr) / (I*vx)
            A51 = (1/(1-ey*cur)) * ( -np.cos(epsi) * cur )
            A52 = (1/(1-ey*cur)) * ( +np.sin(epsi)* cur )
            A61 = np.cos(epsi) / (1-ey*cur)
            A62 = np.sin(epsi) / (1-ey*cur)
            A7      = np.sin(epsi)
            A8      = np.cos(epsi)
            B11     = -(np.sin(delta)*Cf)/m
            B21     = (np.cos(delta)*Cf)/ m
            B31     = (lf*Cf*np.cos(delta)) / I        

            Ai = np.array([ [A11    ,  A12 ,  A13 ,  0., 0., 0.],  # [vx]
                            [0.     ,  A22 ,  A23 ,  0., 0., 0.],  # [vy]
                            [0.     ,  A32 ,  A33 ,  0., 0., 0.],  # [wz]
                            [A51    ,  A52 ,   1. ,  0., 0., 0.],  # [epsi]
                            [A61    ,  A62 ,   0. ,  0., 0., 0.],  # [s]
                            [A7     ,   A8 ,   0. ,  0., 0., 0.]])  # [ey]


            Bi  = np.array([[ B11, 1 ], #[delta, a]
                            [ B21, 0 ],
                            [ B31, 0 ],
                            [ 0,   0 ],
                            [ 0,   0 ],
                            [ 0,   0 ]])


            Ci  = np.array([[ 0 ], 
                            [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ]])               


            Ai = np.eye(len(Ai)) + self.dt * Ai
            Bi = self.dt * Bi
            Ci = self.dt * Ci

            states_new = np.dot(Ai, states) + np.dot(Bi, np.transpose(np.reshape(u[i,:],(1,2))))

            STATES_vec[i] = np.reshape(states_new, (6,))

            states = states_new

            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)

        return STATES_vec, Atv, Btv, Ctv
        




# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for MPC reformulation to QP =======================================
# ======================================================================================================================
# ======================================================================================================================




def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    # EA: P represents the quadratic weight composed by N times Q and R matrices.
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
    q : numpy.array Quadratic cost vector.
    G : scipy.sparse.csc_matrix Linear inequality constraint matrix.
    h : numpy.array Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
    b : numpy.array, optional Linear equality constraint vector.
    initvals : numpy.array, optional Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """
    osqp = OSQP()
    if G is not None:
        l = -inf * ones(len(h))
        if A is not None:
            qp_A = vstack([G, A]).tocsc()
            qp_l = hstack([l, b])
            qp_u = hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l
            qp_u = h
        osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=True, polish=True)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()

    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    feasible = 0
    if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = 1
    return res, feasible



def _buildMatIneqConst(Controller):
    N = Controller.N
    n = Controller.n
    max_vel = Controller.max_vel

    Fx = np.array([[-1., 0., 0., 0., 0., 0.],
                   [+1., 0., 0., 0., 0., 0.]])
    bx = np.array([[-0.01],
                   [max_vel]]) # vx min

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[0.249],  # Max right Steering
                   [0.249],  # Max left Steering
                   [4.0],  # Max Acceleration
                   [1.0]])  # Max DesAcceleration


    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx] * (N)
    Mat = linalg.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0], n))  # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    bxtot = np.tile(np.squeeze(bx), N)

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
    Dummy2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))
    F = np.vstack((Dummy1, Dummy2))
    b = np.hstack((bxtot, butot))

    if Controller.Solver == "CVX":
        F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0].astype(int), np.nonzero(F)[1].astype(int), F.shape)
        F_return = F_sparse
    else:
        F_return = F
    
    return F_return, b



def _buildMatCost(Controller, uOld, vel_ref):
    # EA: This represents to be: [(r-x)^T * Q * (r-x)] up to N+1
    # and [u^T * R * u] up to N

    Q  = Controller.Q
    n  = Q.shape[0]
    R  = Controller.R
    dR = Controller.dR
    # P  = Controller.Q
    N  = Controller.N
    # vt = Controller.vt
    # vt = vel_ref

    uOld  = [Controller.OldSteering[0], Controller.OldAccelera[0]]

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    #c = [R] * (N)
    c = [R + 2 * np.diag(dR)] * (N) # Need to add dR for the derivative input cost

    # print [R + 2 * np.diag(dR)]
    # print "  "

    Mu = linalg.block_diag(*c)
    # print Mu
    # print Mu[:, 2:].shape
    # print "\n\n\n\n"
    # print "  "

    # Need to condider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]


    # Derivative Input Cost
    # Mu has to be symmetric
    OffDiaf = -np.tile(dR, N-1)
    # print Mu
    # print "  "
    np.fill_diagonal(Mu[2:], OffDiaf)  # upper matrix side
    # print Mu
    # print "  "
    np.fill_diagonal(Mu[:, 2:], OffDiaf)   # lower matrix side
    # print Mu[:6,:6]
    # print Mu
    # print "\n\n\n\n"
    # print "  "

    # This is without slack lane:
    M0 = linalg.block_diag(Mx, Q, Mu)

    xtrack = np.array([vel_ref[0], 0, 0, 0, 0, 0])
    for i in range(1, N):
        xtrack = np.append(xtrack, [vel_ref[i], 0, 0, 0, 0, 0])

    xtrack = np.append(xtrack, [vel_ref[-1], 0, 0, 0, 0, 0])

    # print xtrack

    # print "  "

    #xtrack = np.append(xtrack, [vel_ref[N], 0, 0, 0, 0, 0])    

    
    q = - 2 * np.dot(np.append(xtrack, np.zeros(R.shape[0] * N)), M0)     

    # print q

    # print "  "

    # print np.dot(np.append(xtrack, np.zeros(R.shape[0] * N)), M0) 

    # print "\n\n\n\n"

    # vt = 1.0
    # xtrack = np.array([vt, 0, 0, 0, 0, 0])
    # q = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M0)

    # Derivative Input
    q[n*(N+1):n*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost


    if Controller.Solver == "CVX":
        M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0].astype(int), np.nonzero(M)[1].astype(int), M.shape)
        M_return = M_sparse
    else:
        M_return = M

    return M_return, q



def _buildMatEqConst(Controller):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicted input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    # G * z = L + E * x(t) + Eu * OldInputs

    A = Controller.A
    B = Controller.B
    C = Controller.C
    N = Controller.N
    n = Controller.n
    d = Controller.d

    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    #E = np.zeros((n * (N + 1), n))
    E = np.zeros((n * (N + 1) + Controller.steeringDelay, n)) #new
    E[np.arange(n)] = np.eye(n)

    Eu = np.zeros((n * (N + 1) + Controller.steeringDelay, d)) #new

    # L = np.zeros((n * (N + 1) + n + 1, 1)) # n+1 for the terminal constraint
    # L = np.zeros((n * (N + 1), 1))
    # L[-1] = 1 # Summmation of lamba must add up to 1

    L = np.zeros((n * (N + 1) + Controller.steeringDelay, 1))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]
        L[ind1, :]              =  C[i]

    G = np.hstack((Gx, Gu))

    # Delay implementation:
    if Controller.steeringDelay > 0:
        xZerosMat = np.zeros((Controller.steeringDelay, n *(N+1)))
        uZerosMat = np.zeros((Controller.steeringDelay, d * N))
        for i in range(0, Controller.steeringDelay):
            ind2Steer = i * d
            L[n * (N + 1) + i, :] = Controller.OldSteering[i+1]
            uZerosMat[i, ind2Steer] = 1.0        

        Gdelay = np.hstack((xZerosMat, uZerosMat))
        G = np.vstack((G, Gdelay))
    
    return G, E, L, Eu










def ABC_computation_5SV_new(vx, vy, omega, steer, accel, A, B, C, BellParams ):
    # Membership functions:
    num_sched_vars = 5
    num_MF  = 2

    weights = np.zeros(num_MF**num_sched_vars)
    norm_weights = np.zeros(num_MF**num_sched_vars)
    Anew    = np.zeros(3)
    Bnew    = np.zeros(2)
    Cnew    = 0.0
    
    inputs  = np.transpose(np.array([vx,vx,vy,vy,omega,omega,steer,steer,accel,accel]))
    Weights = 1.0/(1.0 + np.power(np.absolute(np.divide(inputs - BellParams[:,2], BellParams[:,0])), 2.0*BellParams[:,1]))    

    # startTimer = datetime.datetime.now() 
    weights[0] = Weights[0]*Weights[2]*Weights[4]*Weights[6]*Weights[8]
    weights[1] = Weights[0]*Weights[2]*Weights[4]*Weights[6]*Weights[9]
    weights[2] = Weights[0]*Weights[2]*Weights[4]*Weights[7]*Weights[8]
    weights[3] = Weights[0]*Weights[2]*Weights[4]*Weights[7]*Weights[9]
    weights[4] = Weights[0]*Weights[2]*Weights[5]*Weights[6]*Weights[8]
    weights[5] = Weights[0]*Weights[2]*Weights[5]*Weights[6]*Weights[9]
    weights[6] = Weights[0]*Weights[2]*Weights[5]*Weights[7]*Weights[8]
    weights[7] = Weights[0]*Weights[2]*Weights[5]*Weights[7]*Weights[9]

    weights[8]  = Weights[0]*Weights[3]*Weights[4]*Weights[6]*Weights[8]
    weights[9]  = Weights[0]*Weights[3]*Weights[4]*Weights[6]*Weights[9]
    weights[10] = Weights[0]*Weights[3]*Weights[4]*Weights[7]*Weights[8]
    weights[11] = Weights[0]*Weights[3]*Weights[4]*Weights[7]*Weights[9]
    weights[12] = Weights[0]*Weights[3]*Weights[5]*Weights[6]*Weights[8]
    weights[13] = Weights[0]*Weights[3]*Weights[5]*Weights[6]*Weights[9]
    weights[14] = Weights[0]*Weights[3]*Weights[5]*Weights[7]*Weights[8]
    weights[15] = Weights[0]*Weights[3]*Weights[5]*Weights[7]*Weights[9]

    weights[16] = Weights[1]*Weights[2]*Weights[4]*Weights[6]*Weights[8]
    weights[17] = Weights[1]*Weights[2]*Weights[4]*Weights[6]*Weights[9]
    weights[18] = Weights[1]*Weights[2]*Weights[4]*Weights[7]*Weights[8]
    weights[19] = Weights[1]*Weights[2]*Weights[4]*Weights[7]*Weights[9]
    weights[20] = Weights[1]*Weights[2]*Weights[5]*Weights[6]*Weights[8]
    weights[21] = Weights[1]*Weights[2]*Weights[5]*Weights[6]*Weights[9]
    weights[22] = Weights[1]*Weights[2]*Weights[5]*Weights[7]*Weights[8]
    weights[23] = Weights[1]*Weights[2]*Weights[5]*Weights[7]*Weights[9]

    weights[24] = Weights[1]*Weights[3]*Weights[4]*Weights[6]*Weights[8]
    weights[25] = Weights[1]*Weights[3]*Weights[4]*Weights[6]*Weights[9]
    weights[26] = Weights[1]*Weights[3]*Weights[4]*Weights[7]*Weights[8]
    weights[27] = Weights[1]*Weights[3]*Weights[4]*Weights[7]*Weights[9]
    weights[28] = Weights[1]*Weights[3]*Weights[5]*Weights[6]*Weights[8]
    weights[29] = Weights[1]*Weights[3]*Weights[5]*Weights[6]*Weights[9]
    weights[30] = Weights[1]*Weights[3]*Weights[5]*Weights[7]*Weights[8]
    weights[31] = Weights[1]*Weights[3]*Weights[5]*Weights[7]*Weights[9]                          
    # deltaTimer = datetime.datetime.now() - startTimer
    # print datetime.datetime.now() - startTimer

    #startTimer = datetime.datetime.now() 
    SUM             = np.sum(weights)
    norm_weights    = np.divide(weights, SUM) 
    Anew            = np.dot(np.transpose(norm_weights),A)
    Bnew            = np.dot(np.transpose(norm_weights),B)
    Cnew            = np.dot(np.transpose(norm_weights),C)
    #deltaTimer = datetime.datetime.now() - startTimer
    #print 'for final_2: ',datetime.datetime.now() - startTimer

    return Anew, Bnew, Cnew





def ABC_computation_5SV(vx, vy, omega, steer, accel, A, B, C, BellParams ):
    # Membership functions:
    num_sched_vars = 5
    num_MF  = 2

    M_vx    = np.zeros(2)
    M_vy    = np.zeros(2)
    M_w     = np.zeros(2)
    M_steer = np.zeros(2)
    M_accel = np.zeros(2)
    weights = np.zeros(num_MF**num_sched_vars)
    norm_weights = np.zeros(num_MF**num_sched_vars)
    Anew    = np.zeros(3)
    Bnew    = np.zeros(2)
    Cnew    = 0.0

    # startTimer = datetime.datetime.now()
    M_vx[0]     = 1.0/(1 + np.power(np.absolute(((vx - BellParams[0,2])/BellParams[0,0]))**2,BellParams[0,1]))
    M_vx[1]     = 1.0/(1 + np.power(np.absolute(((vx - BellParams[1,2])/BellParams[1,0]))**2,BellParams[1,1]))
    M_vy[0]     = 1.0/(1 + np.power(np.absolute(((vy - BellParams[2,2])/BellParams[2,0]))**2,BellParams[2,1]))
    M_vy[1]     = 1.0/(1 + np.power(np.absolute(((vy - BellParams[3,2])/BellParams[3,0]))**2,BellParams[3,1]))
    M_w[0]      = 1.0/(1 + np.power(np.absolute(((omega - BellParams[4,2])/BellParams[4,0]))**2,BellParams[4,1]))
    M_w[1]      = 1.0/(1 + np.power(np.absolute(((omega - BellParams[5,2])/BellParams[5,0]))**2,BellParams[5,1]))
    M_steer[0]  = 1.0/(1 + np.power(np.absolute(((steer - BellParams[6,2])/BellParams[6,0]))**2,BellParams[6,1]))
    M_steer[1]  = 1.0/(1 + np.power(np.absolute(((steer - BellParams[7,2])/BellParams[7,0]))**2,BellParams[7,1]))
    M_accel[0]  = 1.0/(1 + np.power(np.absolute(((accel - BellParams[8,2])/BellParams[8,0]))**2,BellParams[8,1]))
    M_accel[1]  = 1.0/(1 + np.power(np.absolute(((accel - BellParams[9,2])/BellParams[9,0]))**2,BellParams[9,1]))
    # deltaTimer = datetime.datetime.now() - startTimer
    # print 'Viejo method: ', datetime.datetime.now() - startTimer

    #print M_vx, M_vy, M_w, M_steer, M_accel

    # startTimer = datetime.datetime.now()    
    # counter = 0
    # for i in range(0,2):
    #     for j in range(0,2):
    #         for k in range(0,2):
    #             for l in range(0,2):
    #                 for m in range(0,2):
    #                     weights[counter] = M_vx[i]*M_vy[j]*M_w[k]*M_steer[l]*M_accel[m]
    #                     counter += 1  
    # deltaTimer = datetime.datetime.now() - startTimer
    # print datetime.datetime.now() - startTimer

    # startTimer = datetime.datetime.now() 
    weights[0] = M_vx[0]*M_vy[0]*M_w[0]*M_steer[0]*M_accel[0]
    weights[1] = M_vx[0]*M_vy[0]*M_w[0]*M_steer[0]*M_accel[1]
    weights[2] = M_vx[0]*M_vy[0]*M_w[0]*M_steer[1]*M_accel[0]
    weights[3] = M_vx[0]*M_vy[0]*M_w[0]*M_steer[1]*M_accel[1]
    weights[4] = M_vx[0]*M_vy[0]*M_w[1]*M_steer[0]*M_accel[0]
    weights[5] = M_vx[0]*M_vy[0]*M_w[1]*M_steer[0]*M_accel[1]
    weights[6] = M_vx[0]*M_vy[0]*M_w[1]*M_steer[1]*M_accel[0]
    weights[7] = M_vx[0]*M_vy[0]*M_w[1]*M_steer[1]*M_accel[1]

    weights[8]  = M_vx[0]*M_vy[1]*M_w[0]*M_steer[0]*M_accel[0]
    weights[9]  = M_vx[0]*M_vy[1]*M_w[0]*M_steer[0]*M_accel[1]
    weights[10] = M_vx[0]*M_vy[1]*M_w[0]*M_steer[1]*M_accel[0]
    weights[11] = M_vx[0]*M_vy[1]*M_w[0]*M_steer[1]*M_accel[1]
    weights[12] = M_vx[0]*M_vy[1]*M_w[1]*M_steer[0]*M_accel[0]
    weights[13] = M_vx[0]*M_vy[1]*M_w[1]*M_steer[0]*M_accel[1]
    weights[14] = M_vx[0]*M_vy[1]*M_w[1]*M_steer[1]*M_accel[0]
    weights[15] = M_vx[0]*M_vy[1]*M_w[1]*M_steer[1]*M_accel[1]

    weights[16] = M_vx[1]*M_vy[0]*M_w[0]*M_steer[0]*M_accel[0]
    weights[17] = M_vx[1]*M_vy[0]*M_w[0]*M_steer[0]*M_accel[1]
    weights[18] = M_vx[1]*M_vy[0]*M_w[0]*M_steer[1]*M_accel[0]
    weights[19] = M_vx[1]*M_vy[0]*M_w[0]*M_steer[1]*M_accel[1]
    weights[20] = M_vx[1]*M_vy[0]*M_w[1]*M_steer[0]*M_accel[0]
    weights[21] = M_vx[1]*M_vy[0]*M_w[1]*M_steer[0]*M_accel[1]
    weights[22] = M_vx[1]*M_vy[0]*M_w[1]*M_steer[1]*M_accel[0]
    weights[23] = M_vx[1]*M_vy[0]*M_w[1]*M_steer[1]*M_accel[1]

    weights[24] = M_vx[1]*M_vy[1]*M_w[0]*M_steer[0]*M_accel[0]
    weights[25] = M_vx[1]*M_vy[1]*M_w[0]*M_steer[0]*M_accel[1]
    weights[26] = M_vx[1]*M_vy[1]*M_w[0]*M_steer[1]*M_accel[0]
    weights[27] = M_vx[1]*M_vy[1]*M_w[0]*M_steer[1]*M_accel[1]
    weights[28] = M_vx[1]*M_vy[1]*M_w[1]*M_steer[0]*M_accel[0]
    weights[29] = M_vx[1]*M_vy[1]*M_w[1]*M_steer[0]*M_accel[1]
    weights[30] = M_vx[1]*M_vy[1]*M_w[1]*M_steer[1]*M_accel[0]
    weights[31] = M_vx[1]*M_vy[1]*M_w[1]*M_steer[1]*M_accel[1]                          
    # deltaTimer = datetime.datetime.now() - startTimer
    # print datetime.datetime.now() - startTimer


    # startTimer = datetime.datetime.now() 
    SUM = np.sum(weights)

    for i in range(0,num_MF**num_sched_vars):
        norm_weights[i] = weights[i] / SUM 
        Anew = Anew + norm_weights[i]*A[i,:]
        Bnew = Bnew + norm_weights[i]*B[i,:]
        Cnew = Cnew + norm_weights[i]*C[i]

    # deltaTimer = datetime.datetime.now() - startTimer
    # print 'for final: ',datetime.datetime.now() - startTimer


    return Anew, Bnew, Cnew    





#############################################
## States:
##   long velocity    [vx]
##   lateral velocity [vy]
##   angular velocity [wz]
##   theta error      [epsi]
##   distance traveled[s]
##   lateral error    [ey]
##
## Control actions:
##   Steering angle   [delta]
##   Acceleration     [a]
##
## Scheduling variables:
##   vx
##   vy
##   epsi
##   ey
##   cur
#############################################

def _EstimateABC(Controller,Last_xPredicted, uPredicted):

    N   = Controller.N
    dt  = Controller.dt
    lf  = Controller.lf
    lr  = Controller.lr
    m   = Controller.m
    I   = Controller.I
    Cf  = Controller.Cf
    Cr  = Controller.Cr
    mu  = Controller.mu
    g   = Controller.g

    Atv = []
    Btv = []
    Ctv = []

    for i in range(0, N):

        PointAndTangent = Controller.map.PointAndTangent
        
        vy      = Last_xPredicted[i,1]
        epsi    = Last_xPredicted[i,3]
        s       = Last_xPredicted[i,4]
        ey      = Last_xPredicted[i, 5]
        cur     = Curvature(s, PointAndTangent)
        vx      = Last_xPredicted[i,0]
        delta   = uPredicted[i,0]             #EA: set of predicted steering angles

        A11 =  -mu
        A12 = (np.sin(delta) * Cf) / (m*vx)
        A13 = (np.sin(delta) * Cf * lf) / (m*vx) + vy
        A22 = -(Cr + Cf * np.cos(delta)) / (m*vx)
        A23 = -(lf * Cf * np.cos(delta) - lr * Cr) / (m*vx) - vx
        A32 = -(lf * Cf * np.cos(delta) - lr * Cr) / (I*vx)
        A33 = -(lf * lf * Cf * np.cos(delta) + lr * lr * Cr) / (I*vx)
        A51 = (1/(1-ey*cur)) * ( -np.cos(epsi) * cur )
        A52 = (1/(1-ey*cur)) * ( +np.sin(epsi)* cur )
        A61 = np.cos(epsi) / (1-ey*cur)
        A62 = np.sin(epsi) / (1-ey*cur)
        A7      = np.sin(epsi)
        A8      = np.cos(epsi)
        B11     = -(np.sin(delta)*Cf)/m
        B21     = (np.cos(delta)*Cf)/ m
        B31     = (lf*Cf*np.cos(delta)) / I        

        Ai = np.array([ [A11    ,  A12 ,  A13 ,  0., 0., 0.],  # [vx]
                        [0.     ,  A22 ,  A23 ,  0., 0., 0.],  # [vy]
                        [0.     ,  A32 ,  A33 ,  0., 0., 0.],  # [wz]
                        [A51    ,  A52 ,   1. ,  0., 0., 0.],  # [epsi]
                        [A61    ,  A62 ,   0. ,  0., 0., 0.],  # [s]
                        [A7     ,   A8 ,   0. ,  0., 0., 0.]])  # [ey]


        Bi  = np.array([[ B11, 1 ], #[delta, a]
                        [ B21, 0 ],
                        [ B31, 0 ],
                        [ 0,   0 ],
                        [ 0,   0 ],
                        [ 0,   0 ]])

        Ci  = np.array([[ 0 ], 
                        [ 0 ],
                        [ 0 ],
                        [ 0 ],
                        [ 0 ],
                        [ 0 ]])         

        Ai = np.eye(len(Ai)) + dt * Ai
        Bi = dt * Bi
        Ci = dt * Ci

        #############################################
        Atv.append(Ai)
        Btv.append(Bi)
        Ctv.append(Ci)

    return Atv, Btv, Ctv


