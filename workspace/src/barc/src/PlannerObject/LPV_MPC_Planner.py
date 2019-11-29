#!/usr/bin/env python
"""
    File name: stateEstimator.py
    Author: Eugenio Alcala
    Email: eugenio.alcala@upc.edu
    Email: euge2838@gmail.com
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

# solvers.options['show_progress'] = False



class LPV_MPC_Planner:
    """Create the Path Following LMPC Planner with LTV model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, Q, R, dR, L_cf, N, dt, map, Solver):

        self.A    = []
        self.B    = []
        self.C    = []
        self.N    = N
        self.nx   = Q.shape[0]
        self.nu   = R.shape[0]
        self.Q    = Q
        self.QN   = Q
        self.R    = R
        self.dR   = dR
        self.L_cf = L_cf

        self.LinPoints = np.zeros((self.N+2,self.nx))
        self.dt = dt                # Sample time 100 ms
        self.map = map              # Used for getting the road curvature
        self.halfWidth = map.halfWidth

        self.first_it = 1

        self.Solver = Solver

        self.Aeq = []
        self.E = []
        self.L = []
        self.Eu =[]

        self.steeringDelay = 0

        self.OldSteering = [0.0]*int(1)
        self.OldAccelera = [0.0]*int(1)           


        ## EN ALGUN MOMENTO EXTERNALIZAR ESTOS PARAMETROS...
        # Vehicle parameters:
        self.lf = rospy.get_param("lf")
        self.lr = rospy.get_param("lr")
        self.m  = rospy.get_param("m")
        self.I  = rospy.get_param("Iz")
        self.Cf = rospy.get_param("Cf")
        self.Cr = rospy.get_param("Cr")
        self.mu = rospy.get_param("mu")

        self.g      = 9.81
        self.epss   = 0.00000001

        self.max_vel = rospy.get_param("/TrajectoryPlanner/max_vel")
        self.min_vel = rospy.get_param("/TrajectoryPlanner/min_vel")



    def solve(self, x0, Last_xPredicted, uPred, A_LPV, B_LPV ,C_LPV, first_it, max_ey):
        """Computes control action
        Arguments:
            x0: current state position
            EA: Last_xPredicted: it is just used for the warm up
            EA: uPred: set of last predicted control inputs used for updating matrix A LPV
            EA: A_LPV, B_LPV ,C_LPV: Set of LPV matrices
        """

        # pdb.set_trace()

        startTimer              = datetime.datetime.now()

        if (first_it<2):
            self.A, self.B, self.C  = _EstimateABC(self, Last_xPredicted, uPred)   
        else:
            self.A = A_LPV
            self.B = B_LPV
            self.C = C_LPV

        ########################################################
        # Equality constraint section: (Aeq and beq)
        self.Aeq, self.E, self.L, self.Eu   = _buildMatEqConst(self) 

        # NOT considering slew rate:
        # beq   = np.add( np.dot(self.E,x0), self.L[:,0] )  # upper and lower equality constraint

        # Considering slew rate:
        uOld  = [self.OldSteering[0], self.OldAccelera[0]]
        beq   = np.add( np.dot(self.E,x0), self.L[:,0], np.dot(self.Eu,uOld) )  # upper and lower equality constraint
        ########################################################


        Q   = self.Q
        QN  = self.QN
        R   = self.R
        L_cf=self.L_cf
        Aeq = self.Aeq      # Aeq
        N   = self.N        # Hp
        nx  = self.nx       # num. states
        nu  = self.nu       # num. control actions
        dR  = self.dR

        
        ################################################################
        # NEW OPTIMIZATION PROBLEM CODE:
        ################################################################
        # # - quadratic objective
        # P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
        #                        sparse.kron(sparse.eye(N), R)]).tocsc()
        # # - linear objective
        # q = np.hstack([np.kron(np.ones(N), L_cf), L_cf,
        #                np.zeros(N*nu)])


        ################################################################
        # NEW OPTIMIZATION PROBLEM CODE:
        # - Slew rate addition (dR)
        ################################################################
        b = [Q] * (N)
        Mx = linalg.block_diag(*b)

        c = [R + 2 * np.diag(dR)] * (N)
        Mu = linalg.block_diag(*c)

        # Need to condider that the last input appears just once in the difference
        Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
        Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

        # Derivative Input Cost
        OffDiaf = -np.tile(dR, N-1)
        np.fill_diagonal(Mu[2:], OffDiaf)
        np.fill_diagonal(Mu[:, 2:], OffDiaf)

        # This is without slack lane:
        M0 = linalg.block_diag(Mx, Q, Mu)

        q = np.hstack([np.kron(np.ones(N), L_cf), L_cf,
                       np.zeros(N*nu)])

        # Derivative Input
        q[nx*(N+1):nx*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )

        P = sparse.csr_matrix(2 * M0)
        

        # Inequality constraint section:
        umin    = np.array([-0.249, -0.7]) # delta, a 
        umax    = np.array([+0.249, +2.0]) # delta, a 
      
        xmin    = np.array([ self.min_vel, -1, -2, -max_ey, -0.8]) # [ vx vy w ey epsi ]
        xmax    = np.array([ self.max_vel,  1,  2,  max_ey,  0.8])    

        Aineq   = sparse.eye((N+1)*nx + N*nu)
        lineq   = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq   = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])


        """ No se como introducir constraints en "du" dado que esta formulacion QP al parecer
        no me lo permite ya que Aeq ha de ser del mismo tamanio que Aineq. Pero Aeq solo tiene en cuenta 
        las matrices A y B mientras que Aineq tendria en cuenta lineq = [xmin, umin, dumin]

        Una opcion factible para introducir esto es remodelar el modelo ampliando con dos estados, las 
        acciones de control como ya hicimos en un trabajo anterior.
        """

        # dumin    = np.array([-0.1, -0.2]) # ddelta, da 
        # dumax    = np.array([+0.1, +0.3]) # ddelta, da 

        # Aineq   = sparse.eye((N+1)*nx + N*nu + N*nu) # x, u and du
        # lineq   = np.hstack([ np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin), np.kron(np.ones(N), dumin) ])
        # uineq   = np.hstack([ np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax), np.kron(np.ones(N), dumax)])


        A = sparse.vstack([Aeq, Aineq]).tocsc()
        l = np.hstack([beq, lineq])
        u = np.hstack([beq, uineq])

        osqp = OSQP()
        osqp.setup(P, q, A, l, u, warm_start=True, verbose=False, polish=True)
 
        # if initvals is not None:
        #     osqp.warm_start(x=initvals)

        res = osqp.solve()

        # if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        #     # print("OSQP exited with status '%s'" % res.info.status)
        feasible = 0
        if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
            feasible = 1
        # pdb.set_trace()

        ################################################################
        ################################################################

        if feasible == 0:
            print 'QUIT...'

        Solution = res.x

        endTimer = datetime.datetime.now(); 
        deltaTimer = endTimer - startTimer

        self.solverTime = deltaTimer
        self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(nx * (N + 1))]), (N + 1, nx))))
        self.uPred = np.squeeze(np.transpose(np.reshape((Solution[nx * (N + 1) + np.arange(nu * N)]), (N, nu))))

        self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )
        self.xPred = self.xPred.T
        self.uPred = self.uPred.T





    def LPVPrediction(self, x, SS, u):

        lf  = self.lf
        lr  = self.lr
        m   = self.m
        I   = self.I    
        Cf  = self.Cf
        Cr  = self.Cr 
        mu  = self.mu
        g   = self.g

        epss= self.epss

        STATES_vec = np.zeros((self.N, self.nx))

        Atv = []
        Btv = []
        Ctv = []

        for i in range(0, self.N):
            if i==0:
                states  = np.reshape(x, (self.nx,1))

            vx      = float(states[0])
            vy      = float(states[1])
            ey      = float(states[3])
            epsi    = float(states[4])

            PointAndTangent = self.map.PointAndTangent         
            cur     = Curvature(SS[i], PointAndTangent)

            delta   = float(u[i,0])  
          
            A5      = (np.sin(delta)*Cf) / (m*vx)
            A6      = (np.sin(delta)*Cf*lf) / (m*vx) + vy
            A7      = -(Cr + Cf*np.cos(delta)) / (m*vx)
            A8      = -(lf*Cf*np.cos(delta) - lr*Cr) / (m*vx) - vx
            A9      = -(lf*Cf*np.cos(delta) - lr*Cr) / (I*vx)
            A10     = -(lf*lf*Cf*np.cos(delta) + lr*lr*Cr)  / (I*vx)
            A1      = (1/(1-ey*cur)) 
            A2      = np.sin(epsi)
            A4      = vx
            B11     = -(np.sin(delta)*Cf)/m
            B21     = (np.cos(delta)*Cf)/ m
            B31     = (lf*Cf*np.cos(delta)) / I

            Ai = np.array([[-mu     , A5        , A6    , 0.    , 0.   ],  # [vx]
                            [0.     , A7        , A8    , 0.    , 0.   ],  # [vy]
                            [0.     , A9        , A10   , 0.    , 0.   ],  # [wz]
                            [0.     , 1.        , 0.    , 0.    , A4   ],  # [ey]
                            [-A1*cur, A1*A2*cur , 1.    , 0.    , 0.   ]]) # [epsi]

            Bi  = np.array([[ B11, 1 ], #[delta, a]
                            [ B21, 0 ],
                            [ B31, 0 ],
                            [ 0,   0 ],
                            [ 0,   0 ]])

            Ci  = np.array([[ 0 ], 
                            [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ]])         

            Ai = np.eye(len(Ai)) + self.dt * Ai
            Bi = self.dt * Bi
            Ci = self.dt * Ci

            states_new = np.dot(Ai, states) + np.dot(Bi, np.transpose(np.reshape(u[i,:],(1,2))))

            STATES_vec[i] = np.reshape(states_new, (self.nx,))

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


# def osqp_solve_qp(P, q, Aeq=None, ueq=None, Aineq=None, lineq=None, uineq=None, initvals=None):

#     """
#     Solve a Quadratic Program defined as:
#         minimize
#             (1/2) * x.T * P * x + q.T * x
#         subject to
#             lineq <= Aineq  * x <= uineq
#                         Aeq * x == ueq


#     using OSQP <https://github.com/oxfordcontrol/osqp>.
#     Parameters
#     ----------
#     P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
#     q : numpy.array Quadratic cost vector.
#     Aeq : scipy.sparse.csc_matrix Linear inequality constraint matrix.
#     ueq : numpy.array Linear equality constraint vector.
#     leq : numpy.array Linear equality constraint vector == ueq.
#     Aineq : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
#     lineq : numpy.array, optional lowe linear inequality constraint vector.
#     uineq : el vector lineq ya representa los limites negativos y positivos de los constraints, por eso luego se define como un vector de valores infinitos
#     initvals : numpy.array, optional Warm-start guess vector.
#     Returns
#     -------
#     x : array, shape=(n,)
#         Solution to the QP, if found, otherwise ``None``.
#     Note
#     ----
#     OSQP requires `P` to be symmetric, and won't check for errors otherwise.
#     Check out for this point if you e.g. `get nan values
#     <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
#     """

# # Nota: # Esto difiere un poco de como lo teniamos antes (chequear en PathFollowingLPVMPC.py)
# # Esto es debido a que hemos cambiado el orden de los const. de igualdad y los de inigualdad.
# # Pero es exactamente lo mismo.

#     leq = ueq

#     osqp = OSQP()
#     if Aeq is not None:
#         # lineq = -inf * ones(len(uineq)) 
#         if Aeq is not None:
#             print Aeq.shape
#             print Aineq.shape
#             print leq.shape
#             print lineq.shape
#             print ueq.shape
#             print uineq.shape
#             pdb.set_trace()
#             qp_A = sparse.vstack([Aeq, Aineq]).tocsc()
#             qp_l = np.hstack([leq, lineq])
#             qp_u = np.hstack([ueq, uineq]) 
#         else:  # no inequality constraint
#             qp_A = Aineq
#             qp_l = lineq
#             qp_u = uineq
#         osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
#     else:
#         # osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=True)
#         osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)

#     if initvals is not None:
#         osqp.warm_start(x=initvals)

#     res = osqp.solve()

#     # pdb.set_trace()

#     # if res.info.status_val != osqp.constant('OSQP_SOLVED'):
#     #     print("OSQP exited with status '%s'" % res.info.status)

#     feasible = 0

#     if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
#         feasible = 1

#     return res, feasible






def _buildMatEqConst(Planner):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicted input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    # G * z = L + E * x(t) + Eu * OldInputs

    A = Planner.A
    B = Planner.B
    C = Planner.C
    N = Planner.N
    n = Planner.nx
    d = Planner.nu

    Gx = np.eye(n * (N+1))
    Gu = np.zeros((n * (N+1), d * (N)))

    # E = np.zeros((n * (N + 1), n))
    E = np.zeros((n * (N+1) + Planner.steeringDelay, n)) #new
    E[np.arange(n)] = np.eye(n)

    Eu = np.zeros((n * (N + 1) + Planner.steeringDelay, d)) #new

    # L = np.zeros((n * (N + 1) + n + 1, 1))  # n+1 for the terminal constraint
    # L = np.zeros((n * (N + 1), 1))
    # L[-1] = 1                               # Summmation of lamba must add up to 1

    L = np.zeros((n * (N+1) + Planner.steeringDelay, 1))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]
        L[ind1, :]              =  C[i]

    G = np.hstack((Gx, Gu))

    # # Delay implementation:
    # if Planner.steeringDelay > 0:
    #     xZerosMat = np.zeros((Planner.steeringDelay, n *(N+1)))
    #     uZerosMat = np.zeros((Planner.steeringDelay, d * N))
    #     for i in range(0, Planner.steeringDelay):
    #         ind2Steer = i * d
    #         L[n * (N + 1) + i, :] = Planner.OldSteering[i+1]
    #         uZerosMat[i, ind2Steer] = 1.0        

    #     Gdelay = np.hstack((xZerosMat, uZerosMat))
    #     G = np.vstack((G, Gdelay))
    
    return G, E, L, Eu











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

def _EstimateABC(Planner, Last_xPredicted, uPredicted):

    N   = Planner.N
    dt  = Planner.dt
    lf  = Planner.lf
    lr  = Planner.lr
    m   = Planner.m
    I   = Planner.I
    Cf  = Planner.Cf
    Cr  = Planner.Cr
    mu  = Planner.mu
    g   = Planner.g

    epss= Planner.epss

    Atv = []
    Btv = []
    Ctv = []

    for i in range(0, N):

        PointAndTangent = Planner.map.PointAndTangent
        
        vx      = Last_xPredicted[i,0]
        vy      = Last_xPredicted[i,1]
        ey      = Last_xPredicted[i,3]
        epsi    = Last_xPredicted[i,4]
        s       = Last_xPredicted[i,5]
        cur     = Curvature(s, PointAndTangent)
        
        delta   = uPredicted[i]  

        A5      = (np.sin(delta)*Cf) / (m*vx)
        A6      = (np.sin(delta)*Cf*lf) / (m*vx) + vy
        A7      = -(Cr + Cf*np.cos(delta)) / (m*vx)
        A8      = -(lf*Cf*np.cos(delta) - lr*Cr) / (m*vx) - vx
        A9      = -(lf*Cf*np.cos(delta) - lr*Cr) / (I*vx)
        A10     = -(lf*lf*Cf*np.cos(delta) + lr*lr*Cr)  / (I*vx)
        A1      = (1/(1-ey*cur)) 
        A2      = np.sin(epsi)
        A4      = vx
        B11     = -(np.sin(delta)*Cf)/m
        B21     = (np.cos(delta)*Cf)/ m
        B31     = (lf*Cf*np.cos(delta)) / I

        Ai = np.array([[-mu     , A5        , A6    , 0.    , 0.   ],  # [vx]
                        [0.     , A7        , A8    , 0.    , 0.   ],  # [vy]
                        [0.     , A9        , A10   , 0.    , 0.   ],  # [wz]
                        [0.     , 1.        , 0.    , 0.    , A4   ],  # [ey]
                        [-A1*cur, A1*A2*cur , 1.    , 0.    , 0.   ]]) # [epsi]

        Bi  = np.array([[ B11, 1 ], #[delta, a]
                        [ B21, 0 ],
                        [ B31, 0 ],
                        [ 0,   0 ],
                        [ 0,   0 ]])

        Ci  = np.array([[ 0 ], 
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


