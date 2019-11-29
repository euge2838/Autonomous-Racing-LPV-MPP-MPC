#!/usr/bin/env python
"""
    File name: stateEstimator.py
    Author: Eugenio Alcala
    Email: eugenio.alcala@upc.edu
    Python Version: 2.7.12
"""
# ---------------------------------------------------------------------------
# Attibution Information: This project ROS code-based was developed at UPC-IRI
# in the CS2AC group by Eugenio Alcala Baselga (eugenio.alcala@upc.edu).
# 
# This is basically a estimation algorithm based on a polytopic gain scheduling
# approach, similar to Extended Kalman Filter although better.
# ---------------------------------------------------------------------------

import rospy
import os
import sys
from datetime import datetime
sys.path.append(sys.path[0]+'/Utilities')
from trackInitialization import wrap
homedir = os.path.expanduser("~")
sys.path.append(os.path.join(homedir,"barc/workspace/src/barc/src/library"))
from barc.msg import ECU, pos_info, Vel_est, simulatorStates, My_IMU
from marvelmind_nav.msg import hedge_imu_fusion, hedge_pos
from std_msgs.msg import Header
from numpy import eye, zeros, diag, tan, cos, sin, vstack, linalg, pi
from numpy import ones, polyval, size, dot, add
from scipy.linalg import inv, cholesky
from tf import transformations
import math
import numpy as np
import scipy.io as sio
import pdb


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def main():
    # node initialization
    rospy.init_node("state_estimation")
    a_delay     = 0.0
    df_delay    = 0.0
    loop_rate   = 200.0

    counter = 0

    t0 = rospy.get_rostime().to_sec()
    imu = ImuClass(t0)
    gps = GpsClass(t0)
    enc = EncClass(t0)
    ecu = EcuClass(t0)
    sim = SimulatorClass(t0)
    est = Estimator(t0,loop_rate,a_delay,df_delay)

    estMsg = pos_info()
    
    saved_x_est      = []
    saved_y_est      = []
    saved_vx_est     = []
    saved_vy_est     = []
    saved_psi_est    = []
    saved_psiDot_est = []
    saved_ax_est     = []
    saved_ay_est     = []
    saved_switch     = []

    rospy.sleep(0.1)   # Soluciona los problemas de inicializacion


    while not rospy.is_shutdown():
        
        est.estimateState(imu, gps, enc, ecu, sim, est.GS_LPV_Est, est.Continuous_AB_Comp, est.L_Gain_Comp)

        # Save estimator Input.
        saved_x_est.append(estMsg.x)
        saved_y_est.append(estMsg.y)
        saved_vx_est.append(estMsg.v_x)
        saved_vy_est.append(estMsg.v_y)
        saved_psi_est.append(estMsg.psi)
        saved_psiDot_est.append(estMsg.psiDot)
        saved_ax_est.append(estMsg.a_x)
        saved_ay_est.append(estMsg.a_y)

        # Save estimator output
        est.saveHistory()

        estMsg.header.stamp = rospy.get_rostime()
        estMsg.v        = np.sqrt(est.vx_est**2 + est.vy_est**2)
        estMsg.x        = est.x_est 
        estMsg.y        = est.y_est
        estMsg.v_x      = est.vx_est 
        estMsg.v_y      = est.vy_est
        estMsg.psi      = est.yaw_est
        estMsg.psiDot   = est.psiDot_est
        estMsg.a_x      = est.ax_est
        estMsg.a_y      = est.ay_est
        estMsg.u_a      = ecu.a
        estMsg.u_df     = ecu.df
        est.state_pub_pos.publish(estMsg)

        est.rate.sleep()


    print "gps x      package lost:", float(est.x_count)/float(est.pkg_count)
    print "gps y      package lost:", float(est.y_count)/float(est.pkg_count)
    print "enc v      package lost:", float(est.v_meas_count)/float(est.pkg_count)
    print "imu ax     package lost:", float(est.ax_count)/float(est.pkg_count)
    print "imu ay     package lost:", float(est.ay_count)/float(est.pkg_count)
    print "imu psiDot package lost:", float(est.psiDot_count)/float(est.pkg_count)
    
    homedir = os.path.expanduser("~")
    pathSave = os.path.join(homedir,"barc_data/estimator_output.npz")
    np.savez(pathSave,yaw_est_his       = saved_psi_est,
                      psiDot_est_his    = saved_psiDot_est,
                      x_est_his         = saved_x_est,
                      y_est_his         = saved_y_est,
                      vx_est_his        = saved_vx_est,
                      vy_est_his        = saved_vy_est,
                      ax_est_his        = saved_ax_est,
                      ay_est_his        = saved_ay_est,
                      gps_time          = est.gps_time,
                      imu_time          = est.imu_time,
                      enc_time          = est.enc_time,
                      inp_x_his         = est.x_his,
                      inp_y_his         = est.y_his,
                      inp_v_meas_his    = est.v_meas_his,
                      inp_ax_his        = est.ax_his,
                      inp_ay_his        = est.ay_his,
                      inp_psiDot_his    = est.psiDot_his,
                      inp_a_his         = est.inp_a_his,
                      inp_df_his        = est.inp_df_his,
                      flagVy            = saved_switch,
                      roll_his          = est.roll_his,
                      pitch_his         = est.pitch_his,
                      wx_his            = est.wx_his,
                      wy_his            = est.wy_his,
                      wz_his            = est.wz_his,
                      v_rl_his          = est.v_rl_his,
                      v_rr_his          = est.v_rr_his,
                      v_fl_his          = est.v_fl_his,
                      v_fr_his          = est.v_fr_his,
                      psi_raw_his       = est.psi_raw_his)

    pathSave = os.path.join(homedir,"barc_data/estimator_imu.npz")
    np.savez(pathSave,psiDot_his    = imu.psiDot_his,
                      roll_his      = imu.roll_his,
                      pitch_his     = imu.pitch_his,
                      yaw_his       = imu.yaw_his,
                      ax_his        = imu.ax_his,
                      ay_his        = imu.ay_his,
                      imu_time      = imu.time_his)

    pathSave = os.path.join(homedir,"barc_data/estimator_gps.npz")
    np.savez(pathSave,x_his         = gps.x_his,
                      y_his         = gps.y_his,
                      gps_time      = gps.time_his)

    pathSave = os.path.join(homedir,"barc_data/estimator_enc.npz")
    np.savez(pathSave,v_fl_his          = enc.v_fl_his,
                      v_fr_his          = enc.v_fr_his,
                      v_rl_his          = enc.v_rl_his,
                      v_rr_his          = enc.v_rr_his,
                      enc_time          = enc.time_his)

    pathSave = os.path.join(homedir,"barc_data/estimator_ecu.npz")
    np.savez(pathSave,a_his         = ecu.a_his,
                      df_his        = ecu.df_his,
                      ecu_time      = ecu.time_his)

    print "Finishing saveing state estimation data"







class Estimator(object):
    """ Object collecting estimated state data
    Attributes:
        Estimated states:
            1.vx_est     2.vy_est    3.psiDot_est    
            4.x_est      5.y_est     6.yaw_est  

        Time stamp
            1.t0 2.time_his 3.curr_time

    Methods:
        stateEstimate( imu, gps, enc, ecu, sim, observer, AB_comp, L_comp ):
            Estimate current state from sensor data

        GS_LPV_Est( states_est, y_meas, u, Continuous_AB_Comp, L_Gain_Comp ):
            Gain Scheduling LPV Estimator

        Continuous_AB_Comp( vx, vy, theta, steer):
            LPV State Space computation

        L_Gain_Comp(self, vx, vy, theta, steer):
            Scheduling-based polytopic estimator gain

        ekf(y,u):
            Extended Kalman filter
        ukf(y,u):
            Unscented Kalman filter
        numerical_jac(func,x,u):
            Calculate jacobian numerically
        f(x,u):
            System prediction model
        h(x,u):
            System measurement model
    """
    def __init__(self,t0,loop_rate,a_delay,df_delay):

        dt             = 1.0 / loop_rate
        self.rate      = rospy.Rate(loop_rate)
        self.dt             = dt
        self.a_delay        = a_delay
        self.df_delay       = df_delay
        self.motor_his      = [0.0]*int(a_delay/dt)
        self.servo_his      = [0.0]*int(df_delay/dt)



        # Variables nuevas (12-07-19):
        self.n_states  = 6
        self.n_meas    = 5
        self.L_gain    = np.zeros((self.n_states, self.n_meas))

        estimator_path              = '/home/euge/GitHub/barc/MATLAB/OBSERVER_6states_WORKS' 
        self.Est_Gains_LS           = sio.loadmat(estimator_path+'/Estimator_Gains_LS.mat')['Llmi']     # vy max = 0.3 / w max = 3
        self.SchedVars_Limits_LS    = sio.loadmat(estimator_path+'/Estimator_Gains_LS.mat')['SchedVars_Limits'] 

        self.Est_Gains_HS           = sio.loadmat(estimator_path+'/Estimator_Gains_HS.mat')['Llmi']     # vy max = 0.3 / w max = 3
        self.SchedVars_Limits_HS    = sio.loadmat(estimator_path+'/Estimator_Gains_HS.mat')['SchedVars_Limits']         

        #  No funciona tan bien como el anterior:
        # self.Est_Gains_HS           = sio.loadmat(estimator_path+'/Estimator_Gains_HS_test.mat')['Llmi']     # vy max = 0.3 / w max = 3
        # self.SchedVars_Limits_HS    = sio.loadmat(estimator_path+'/Estimator_Gains_HS_test.mat')['SchedVars_Limits']   

        self.index = 0;
        self.C_obs = np.array([[  1., 0., 0., 0., 0., 0. ],  # vx
                               [  0., 0., 1., 0., 0., 0. ],  # omega
                               [  0., 0., 0., 1., 0., 0. ],  # x
                               [  0., 0., 0., 0., 1., 0. ],  # y
                               [  0., 0., 0., 0., 0., 1. ]]) # yaw

        self.state_pub_pos  = rospy.Publisher('pos_info', pos_info, queue_size=1)
        self.t0             = t0
        
        self.x_est          = 0.0
        self.y_est          = 0.0
        self.vx_est         = rospy.get_param("simulator/init_vx")
        self.vy_est         = 0.0
        self.v_est          = 0.0
        self.ax_est         = 0.0
        self.ay_est         = 0.0
        self.yaw_est        = 0.0
        self.psiDot_est     = 0.0
        self.states_est     = np.array([ self.vx_est, self.vy_est, self.psiDot_est, self.x_est, self.y_est, self.yaw_est ])

        self.curr_time      = rospy.get_rostime().to_sec() - self.t0
        self.prev_time      = self.curr_time

        self.x_est_his          = []
        self.y_est_his          = []
        self.vx_est_his         = []
        self.vy_est_his         = []
        self.v_est_his          = []
        self.ax_est_his         = []
        self.ay_est_his         = []
        self.yaw_est_his        = []
        self.psiDot_est_his     = []
        self.time_his           = []

        # SAVE THE measurement/input SEQUENCE USED BY observer
        self.x_his       = []
        self.y_his       = []
        self.v_meas_his  = []
        self.vy_meas_his = []
        self.ax_his      = []
        self.ay_his      = []
        self.psiDot_his  = []
        self.inp_a_his   = []
        self.inp_df_his  = []
        self.psi_raw_his = []

        # Angular Velocities
        self.wx_his     = []
        self.wy_his     = []
        self.wz_his     = []
        
        # Roll an pitch
        self.roll_his   = []
        self.pitch_his  = []

        # Encored Readinds
        self.v_rl_his   =[]
        self.v_rr_his   =[]
        self.v_fl_his   =[]
        self.v_fr_his   =[]

        # COUNTERS FOR PACKAGE LOST
        self.pkg_count    = 0
        self.x_count      = 0
        self.y_count      = 0
        self.v_meas_count = 0
        self.ax_count     = 0
        self.ay_count     = 0
        self.psiDot_count = 0

        self.gps_time = []
        self.enc_time = []
        self.imu_time = []


    # ecu command update
    def estimateState(self, imu, gps, enc, ecu, sim, observer, AB_comp, L_comp ):
        """
        Estima el vector [ vx vy omega x y yaw ] a partir del vector medido [ vx, omega, x, y, yaw ]
        a traves de los sensores (encoder, imu y gps).
        """


        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.motor_his.append(ecu.a)
        self.servo_his.append(ecu.df)


        if self.curr_time > 0.02:
            y = np.array([ enc.v_meas, imu.psiDot, gps.x, gps.y, imu.yaw])
        else:
            y = np.array([ self.vx_est, imu.psiDot, gps.x, gps.y, sim.yaw])

        # Measurements from simulator:
        # y = np.array([ sim.vx, sim.psiDot, sim.x, sim.y, sim.yaw])

        # Measurements from sensors:
        # y = np.array([ enc.v_meas, imu.psiDot, gps.x, gps.y, imu.yaw])

        u = [self.servo_his.pop(0), self.motor_his.pop(0)]

        observer( self.states_est, y, u, AB_comp, L_comp )




    def GS_LPV_Est(self, states_est, y_meas, u, Continuous_AB_Comp, L_Gain_Comp ):

        """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Vehicle states observation:
        %   x_est: [ vx vy omega x y yaw ]
        %   y_meas: Real Vehicle Output [ vx, omega, x, y, yaw ]
        %   u : control actions         [ steer, accel ]
        %   C_obs: Output matrix
        %   index: control loop index
        %   Ts: Sample time for integrating the observer model 
        %   L: Observer gain [8 vertexes]
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        Ts = self.curr_time - self.prev_time
        # print('Ts = ', self.dt)
        self.prev_time = self.curr_time

        states_est = self.states_est

        if self.curr_time > 0.02:
            steer   = u[0]
            Vx      = states_est[0]
            Vy      = states_est[1]
            Theta   = states_est[5]  
        else:
            steer   = u[0]
            Vx      = y_meas[0]
            Vy      = 0.0
            Theta   = y_meas[4]            

        Continuous_AB_Comp(Vx, Vy, Theta, steer)

        L_Gain_Comp(Vx, Vy, Theta, steer)    

        self.states_est  = states_est + ( self.dt * dot( ( self.A_obs + dot(self.L_gain, self.C_obs) ), states_est ) 
                         +    self.dt * dot(self.B_obs, u) 
                         -    self.dt * dot(self.L_gain, y_meas) )

        self.vx_est     = self.states_est[0]
        self.vy_est     = self.states_est[1]
        self.psiDot_est = self.states_est[2]
        self.x_est      = self.states_est[3]
        self.y_est      = self.states_est[4]
        self.yaw_est    = self.states_est[5]

        self.ax_est = 0 # no las estimo pero las dejo a cero por si acaso
        self.ay_est = 0
        
        self.index += 1



    def Continuous_AB_Comp(self, vx, vy, theta, steer):

        lf          = 0.125;
        lr          = 0.125;
        m           = 1.98;
        I           = 0.03;
        Cf          = 60;
        Cr          = 60;  
        mu          = 0.05;
            
        B11 = -(np.sin(steer) * Cf) / m
        B21 = (np.cos(steer) * Cf) / m
        B31 = (lf * Cf * np.cos(steer)) / I  

        self.B_obs = np.array([[B11,    1.],  # [steer, acceleration]
                               [B21,    0.],
                               [B31,    0.],
                               [0.,     0.],
                               [0.,     0.],
                               [0.,     0.]])

        A11 =  -mu
        A12 = (np.sin(steer) * Cf) / (m*vx)
        A13 = (np.sin(steer) * Cf * lf) / (m*vx) + vy
        A22 = -(Cr + Cf * np.cos(steer)) / (m*vx)
        A23 = -(lf * Cf * np.cos(steer) - lr * Cr) / (m*vx) - vx
        A32 = -(lf * Cf * np.cos(steer) - lr * Cr) / (I*vx)
        A33 = -(lf * lf * Cf * np.cos(steer) + lr * lr * Cr) / (I*vx)

        self.A_obs = np.array([[ A11,                   A12,       A13,    0.,  0.,  0.],   # [vx]
                               [ 0.,                    A22,       A23,    0.,  0.,  0.],   # [vy]
                               [ 0.,                    A32,       A33,    0.,  0.,  0.],   # [wz]  
                               [ np.cos(theta),  -np.sin(theta),    0.,    0.,  0.,  0.],   # [x]  
                               [ np.sin(theta),   np.cos(theta),    0.,    0.,  0.,  0.],   # [y]   
                               [ 0.,                    0.,         1.,    0.,  0.,  0.]]); # [theta]  


    def L_Gain_Comp(self, vx, vy, theta, steer):
        """ 
        Hybrid gain scheduling:
        Two polytopes --> from vx=0.1 to vx=1.1
                      --> from vx=1.0 to vx=4.0
        """
        mu    = np.zeros((16, 1))

        if vx > self.SchedVars_Limits_LS[0,1]:
            SchedVars_Limits = self.SchedVars_Limits_HS
            Est_Gains   = self.Est_Gains_HS
        else:
            SchedVars_Limits = self.SchedVars_Limits_LS
            Est_Gains   = self.Est_Gains_LS


        result = np.zeros((self.n_states, self.n_meas))

        M_vx_despl_min      = (SchedVars_Limits[0,1] - vx)    / (SchedVars_Limits[0,1] - SchedVars_Limits[0,0] )
        M_vy_despl_min      = (SchedVars_Limits[1,1] - vy)    / (SchedVars_Limits[1,1] - SchedVars_Limits[1,0] )
        M_steer_min         = (SchedVars_Limits[3,1] - steer) / (SchedVars_Limits[3,1] - SchedVars_Limits[3,0])
        M_theta_min         = (SchedVars_Limits[5,1] - theta) / (SchedVars_Limits[5,1] - SchedVars_Limits[5,0])

        # if vx > SchedVars_Limits[0,1] or vx < SchedVars_Limits[0,0]:
        #     print( '[ESTIMATOR/L_Gain_Comp]: Vx is out of the polytope ...' )
        # elif vy > SchedVars_Limits[1,1] or vy < SchedVars_Limits[1,0]:
        #     print( '[ESTIMATOR/L_Gain_Comp]: Vy is out of the polytope ...' )
        # elif steer > SchedVars_Limits[3,1] or steer < SchedVars_Limits[3,0]:
        #     print( '[ESTIMATOR/L_Gain_Comp]: Steering is out of the polytope ... = ',steer)
        # elif theta > SchedVars_Limits[5,1] or theta < SchedVars_Limits[5,0]:
        #     print( '[ESTIMATOR/L_Gain_Comp]: Theta is out of the polytope ...', theta )

        mu[0]               = M_vx_despl_min         * M_vy_despl_min      * M_steer_min      *  M_theta_min
        mu[1]               = M_vx_despl_min         * M_vy_despl_min      * M_steer_min      *  (1-M_theta_min) 
        mu[2]               = M_vx_despl_min         * M_vy_despl_min      * (1-M_steer_min)  *  M_theta_min
        mu[3]               = M_vx_despl_min         * M_vy_despl_min      * (1-M_steer_min)  *  (1-M_theta_min)
        mu[4]               = M_vx_despl_min         * (1-M_vy_despl_min)  * M_steer_min      *  M_theta_min
        mu[5]               = M_vx_despl_min         * (1-M_vy_despl_min)  * M_steer_min      *  (1-M_theta_min)
        mu[6]               = M_vx_despl_min         * (1-M_vy_despl_min)  * (1-M_steer_min)  *  M_theta_min
        mu[7]               = M_vx_despl_min         * (1-M_vy_despl_min)  * (1-M_steer_min)  *  (1-M_theta_min)
        
        mu[8]               = (1-M_vx_despl_min)     * M_vy_despl_min      * M_steer_min      *  M_theta_min
        mu[9]               = (1-M_vx_despl_min)     * M_vy_despl_min      * M_steer_min      *  (1-M_theta_min) 
        mu[10]              = (1-M_vx_despl_min)     * M_vy_despl_min      * (1-M_steer_min)  *  M_theta_min
        mu[11]              = (1-M_vx_despl_min)     * M_vy_despl_min      * (1-M_steer_min)  *  (1-M_theta_min)
        mu[12]              = (1-M_vx_despl_min)     * (1-M_vy_despl_min)  * M_steer_min      *  M_theta_min
        mu[13]              = (1-M_vx_despl_min)     * (1-M_vy_despl_min)  * M_steer_min      *  (1-M_theta_min)
        mu[14]              = (1-M_vx_despl_min)     * (1-M_vy_despl_min)  * (1-M_steer_min)  *  M_theta_min
        mu[15]              = (1-M_vx_despl_min)     * (1-M_vy_despl_min)  * (1-M_steer_min)  *  (1-M_theta_min)  

        for i in range(0, 16):
            result += mu[i] * Est_Gains[:,:,i]

        self.L_gain = result
     
        

    def saveHistory(self):
        self.time_his.append(self.curr_time)

        self.x_est_his.append(self.x_est)
        self.y_est_his.append(self.y_est)
        self.vx_est_his.append(self.vx_est)
        self.vy_est_his.append(self.vy_est)
        self.v_est_his.append(self.v_est)
        self.ax_est_his.append(self.ax_est)
        self.ay_est_his.append(self.ay_est)
        self.yaw_est_his.append(self.yaw_est)
        self.psiDot_est_his.append(self.psiDot_est)








# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================


class SimulatorClass(object):
    """ Object collecting simulator measurement data:
        This class listen the published simulator state topic which consists on the vehicle state vector.
    """

    def __init__(self,t0):

        rospy.Subscriber('simulatorStates', simulatorStates, self.simulator_callback, queue_size=1)

        # Simulator measurement
        self.x      = 0.0
        self.y      = 0.0
        self.yaw    = 0.0
        self.vx     = rospy.get_param("simulator/init_vx")
        self.vy     = 0.0
        self.psiDot = 0.0

        # simulator_flag = False


    def simulator_callback(self,data):
        """Unpack message from sensor, IMU"""      

        # simulator_flag = True    

        self.x       = data.x
        self.y       = data.y
        self.yaw     = data.psi
        self.vx      = data.vx
        self.vy      = data.vy
        self.psiDot  = data.psiDot



class ImuClass(object):
    """ Object collecting GPS measurement data
    Attributes:
        Measurement:
            1.roll 2.pitch  3.yaw  4.ax  5.ay  6.psiDot
        Measurement history:
            1.roll_his 2.pitch_his  3.yaw_his  4.ax_his  5.ay_his  6.psiDot_his
        Time stamp
            1.t0 2.time_his
    """
    def __init__(self,t0):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """

        rospy.Subscriber('imu/data', My_IMU, self.imu_callback, queue_size=1)

        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.ax      = 0.0
        self.ay      = 0.0        
        self.psiDot  = 0.0

        # Imu measurement history
        self.roll_his    = [0.0]
        self.pitch_his   = [0.0]
        self.yaw_his     = [0.0]
        self.ax_his      = [0.0]
        self.ay_his      = [0.0]
        self.psiDot_his  = [0.0]        
        # self.yaw_raw_his = [0.0]
        
        # time stamp
        self.t0          = t0
        self.time_his    = [0.0]

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

    def imu_callback(self, data):
        """Unpack message from sensor, IMU"""
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.roll   = data.roll
        self.pitch  = data.pitch
        self.yaw    = data.yaw         
        self.psiDot = data.psiDot
        self.ax     = data.ax
        self.ay     = data.ay


        self.saveHistory()
        
    def saveHistory(self):
        """ Save measurement data into history array"""
        self.time_his.append(self.curr_time)
        self.yaw_his.append(self.yaw)
        self.psiDot_his.append(self.psiDot)
        self.ax_his.append(self.ax)
        self.ay_his.append(self.ay)
        self.roll_his.append(self.roll)
        self.pitch_his.append(self.pitch)



class GpsClass(object):
    """ Object collecting GPS measurement data
    Attributes:
        Measurement:
            1.x 2.y
        Measurement history:
            1.x_his 2.y_his
        Time stamp
            1.t0 2.time_his 3.curr_time
    """
    def __init__(self,t0):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """

        rospy.Subscriber('hedge_pos', hedge_pos, self.gps_callback, queue_size=1)

        # GPS measurement
        self.angle  = 0.0
        self.x      = 0.0
        self.y      = 0.0
        self.x_ply  = 0.0
        self.y_ply  = 0.0
        
        # GPS measurement history
        self.angle_his  = np.array([0.0])
        self.x_his      = np.array([0.0])
        self.y_his      = np.array([0.0])
        self.x_ply_his  = np.array([0.0])
        self.y_ply_his  = np.array([0.0])
        
        # time stamp
        self.t0         = t0
        self.time_his   = np.array([0.0])
        self.time_ply_his = np.array([0.0])
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0

    def gps_callback(self, data):
        """Unpack message from sensor, GPS"""
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        dist = np.sqrt((data.x_m-self.x_his[-1])**2+(data.y_m-self.y_his[-1])**2)
        # if dist < 0.5:
        # if self.x_his[-1] != data.x_m:
        self.x = data.x_m
        self.y = data.y_m            

        # 1) x(t) ~ c0x + c1x * t + c2x * t^2
        # 2) y(t) ~ c0y + c1y * t + c2y * t^2
        # c_X = [c0x c1x c2x] and c_Y = [c0y c1y c2y] 
        n_intplt = 50 # 50*0.01=0.5s data
        if size(self.x_ply_his,0) > n_intplt:
            x_intplt = self.x_ply_his[-n_intplt:]
            y_intplt = self.y_ply_his[-n_intplt:]
            t_intplt = self.time_ply_his[-n_intplt:]-self.time_ply_his[-n_intplt]
            t_matrix = vstack([t_intplt**2, t_intplt, ones(n_intplt)]).T
            c_X = linalg.lstsq(t_matrix, x_intplt)[0]
            c_Y = linalg.lstsq(t_matrix, y_intplt)[0]
            self.x_ply = polyval(c_X, self.curr_time-self.time_ply_his[-n_intplt])
            self.y_ply = polyval(c_Y, self.curr_time-self.time_ply_his[-n_intplt])

        self.saveHistory()

    def saveHistory(self):
        self.time_his = np.append(self.time_his,self.curr_time)
        self.angle_his= np.append(self.angle_his,self.angle)
        self.x_his    = np.append(self.x_his,self.x)
        self.y_his    = np.append(self.y_his,self.y)
        # if self.x_ply_his[-1] != self.x_ply:
        self.x_ply_his  = np.append(self.x_ply_his,self.x_ply)
        self.y_ply_his  = np.append(self.y_ply_his,self.y_ply)
        self.time_ply_his = np.append(self.time_ply_his,self.curr_time)



class EncClass(object):
    """ Object collecting ENC measurement data
    Attributes:
        Measurement:
            1.v_fl 2.v_fr 3. v_rl 4. v_rr
        Measurement history:
            1.v_fl_his 2.v_fr_his 3. v_rl_his 4. v_rr_his
        Time stamp
            1.t0 2.time_his 3.curr_time
    """
    def __init__(self,t0):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """
        rospy.Subscriber('vel_est', Vel_est, self.enc_callback, queue_size=1)

        # ENC measurement
        self.v_fl      = 0.0
        self.v_fr      = 0.0
        self.v_rl      = 0.0
        self.v_rr      = 0.0
        self.v_meas    = 0.0
        
        # ENC measurement history
        self.v_fl_his    = []
        self.v_fr_his    = []
        self.v_rl_his    = []
        self.v_rr_his    = []
        self.v_meas_his  = []
        
        # time stamp
        self.v_count    = 0
        self.v_prev     = 0.0
        self.t0         = t0
        self.time_his   = []
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0

    def enc_callback(self,data):
        """Unpack message from sensor, ENC"""
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.v_fl = data.vel_fl
        self.v_fr = data.vel_fr
        self.v_rl = data.vel_bl
        self.v_rr = data.vel_br
        v_est = (self.v_rl + self.v_rr)/2
        if v_est != self.v_prev:
            self.v_meas = v_est
            self.v_prev = v_est
            self.v_count = 0
        else:
            self.v_count += 1
            if self.v_count > 40:
                self.v_meas = 0       

        self.saveHistory()

    def saveHistory(self):
        self.time_his.append(self.curr_time)
        
        self.v_fl_his.append(self.v_fl)
        self.v_fr_his.append(self.v_fr)
        self.v_rl_his.append(self.v_rl)
        self.v_rr_his.append(self.v_rr)

        self.v_meas_his.append(self.v_meas)




class EcuClass(object):
    """ Object collecting CMD command data
    Attributes:
        Input command:
            1.a 2.df
        Input command history:
            1.a_his 2.df_his
        Time stamp
            1.t0 2.time_his 3.curr_time
    """
    def __init__(self,t0):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """
        rospy.Subscriber('ecu', ECU, self.ecu_callback, queue_size=1)

        # ECU measurement
        self.a  = 0.0
        self.df = 0.0
        
        # ECU measurement history
        self.a_his  = []
        self.df_his = []
        
        # time stamp
        self.t0         = t0
        self.time_his   = []
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0

    def ecu_callback(self,data):
        """Unpack message from sensor, ECU"""
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.a  = data.motor
        self.df = data.servo

        self.saveHistory()

    def saveHistory(self):
        self.time_his.append(self.curr_time)
        
        self.a_his.append(self.a)
        self.df_his.append(self.df)






if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
