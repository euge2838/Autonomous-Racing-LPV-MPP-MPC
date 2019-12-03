#!/usr/bin/env python
"""
    File name: Online Planner-LPV-MPC.py
    Author: Eugenio Alcala
    Email: eugenio.alcala@upc.edu.edu
    Date: 25/11/2019
    Python Version: 2.7.12
"""

import os
import sys
sys.path.append(sys.path[0]+'/ControllerObject')
sys.path.append(sys.path[0]+'/PlannerObject')
sys.path.append(sys.path[0]+'/Utilities')
import datetime
import rospy
from trackInitialization import Map, wrap
from barc.msg import pos_info, prediction, My_Planning, Racing_Info
import numpy as np
from numpy import hstack
import scipy.io as sio
import pdb
import pickle
from utilities import Regression, Curvature
from dataStructures import LMPCprediction, EstimatorData
from LPV_MPC_Planner import LPV_MPC_Planner
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from scipy import signal


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():

    # Initializa ROS node
    rospy.init_node("Trajectory_Planner")

    planning_refs   = rospy.Publisher('My_Planning', My_Planning, queue_size=1)

    map             = Map()  

    refs            = My_Planning()
    refs.x_d        = []
    refs.y_d        = []
    refs.psi_d      = []
    refs.vx_d       = []
    refs.curv_d     = [] 

    HW              = rospy.get_param("/TrajectoryPlanner/halfWidth")
    # HW              = 0.35 # It is a bit larger than the configured in the launch with the aim of improving results
    loop_rate       = rospy.get_param("/TrajectoryPlanner/Frecuency") # 20 Hz (50 ms)
    dt              = 1.0/loop_rate
    rate            = rospy.Rate(loop_rate)

    
    Testing = rospy.get_param("/TrajectoryPlanner/Testing")
    if Testing == 0:
        racing_info     = RacingDataClass()
        estimatorData   = EstimatorData()
        mode            = rospy.get_param("/control/mode")


    else:   # Testing mode
        mode            = "simulations"
        racing_info     = RacingDataClass()


    first_it        = 1

    Xlast           = 0.0
    Ylast           = 0.0
    Thetalast       = 0.0

    Counter         = 0

    ALL_LOCAL_DATA  = np.zeros((1000,7))       # [vx vy psidot ey psi udelta uaccel]
    References      = np.zeros((1000,5))

    ELAPSD_TIME     = np.zeros((1000,1))

    TimeCounter     = 0
    PlannerCounter  = 0
    start_LapTimer  = datetime.datetime.now()


    if rospy.get_param("/TrajectoryPlanner/Visualization") == 1:
        fig, axtr, line_tr, line_pred, line_trs, line_cl, line_gps_cl, rec, rec_sim = InitFigure_XY( map, mode, HW )


    #####################################################################

    # states = [vx vy w ey epsi]
    Q   = - np.diag( [ -0.000000000000088, -9.703658572659423, -0.5, 0.000000000213635, -0.153591566469547] )  
    L   = - np.array( [ 1.00702414775175, 0.187661946033823, -0.0,  0.0, -0.0329493219494661 ] )
    R   =   np.diag([0.8, 0.0])   # Input variables weight [delta, a]. El peso de "a" ha de ser 0.
    dR  =   np.array([6.0, 6.0])  # Input slew rate weight [d_delta, d_a]

    N   = rospy.get_param("/TrajectoryPlanner/N")

    planner_dt = dt
    # planner_dt = 0.05

    Planner  = LPV_MPC_Planner(Q, R, dR, L, N, planner_dt, map, "OSQP")

    #####################################################################


    # Filter to be applied.
    b_filter, a_filter = signal.ellip(4, 0.01, 120, 0.125)  


    LPV_X_Pred      = np.zeros((N,5))
    Xref            = np.zeros(N+1)
    Yref            = np.zeros(N+1)
    Thetaref        = np.zeros(N+1)
    xp              = np.zeros(N)
    yp              = np.zeros(N)
    yaw             = np.zeros(N)    

    SS              = np.zeros(N+1,) 

    GlobalState     = np.zeros(6)
    LocalState      = np.zeros(5)    

    while (not rospy.is_shutdown()):  
        
        # If inside the track publish input
        if (racing_info.LapNumber >= 1 or Testing == 1):

            startTimer = datetime.datetime.now()

            ###################################################################################################
            # GETTING INITIAL STATE:
            ###################################################################################################   

            if Testing == 0:
                # Read Measurements
                GlobalState[:] = estimatorData.CurrentState                 # The current estimated state vector [vx vy w x y psi]
                LocalState[:]  = np.array([ GlobalState[0], GlobalState[1], GlobalState[2], 0.0, 0.0 ]) # [vx vy w ey epsi]
                S_realDist, LocalState[4], LocalState[3], insideTrack = map.getLocalPosition(GlobalState[3], GlobalState[4], GlobalState[5])
            else:
                LocalState[:] = np.array([1.0, 0, 0, 0, 0])
                S_realDist, LocalState[4], LocalState[3], insideTrack = map.getLocalPosition(xp[0], yp[0], yaw[0])
                
            ###################################################################################################
            # OPTIMIZATION:
            ###################################################################################################             

            if first_it == 1:
                # Resolvemos la primera OP con los datos del planner no-lineal:
                # xx son los estados del modelo no lineal para la primera it.
                # delta es el steering angle del modelo no lineal para la primera it.

                x0 = LocalState[:]        # Initial planning state
                accel_rate = 0.2
                xx, uu = predicted_vectors_generation(N, x0, accel_rate, planner_dt)

                
                Planner.solve(x0, xx, uu, 0, 0, 0, first_it, HW)

                first_it += 1

                print "Dentro primera iter... = "
               
            else:                               

                # LPV_X_Pred, A_L, B_L, C_L = Planner.LPVPrediction( LocalState[:], SS[:], Planner.uPred ) 
                # Planner.solve( LocalState[:], 0, 0, A_L, B_L, C_L, first_it, HW)    

                # Planner.solve(Planner.xPred[1,:], LPV_X_Pred, Planner.uPred, A_L, B_L, C_L, first_it)

                LPV_X_Pred, A_L, B_L, C_L = Planner.LPVPrediction( Planner.xPred[1,:], SS[:], Planner.uPred )    
                Planner.solve(Planner.xPred[1,:], 0, 0, A_L, B_L, C_L, first_it, HW)



            ###################################################################################################
            ###################################################################################################

            # print "Ey: ", Planner.xPred[:,3] 

            # Saving current control actions to perform then the slew rate:
            Planner.OldSteering.append(Planner.uPred[0,0]) 
            Planner.OldAccelera.append(Planner.uPred[0,1])

            # pdb.set_trace()

            #####################################
            ## Getting vehicle position:
            #####################################
            Xref[0]     = Xlast
            Yref[0]     = Ylast
            Thetaref[0] = Thetalast

            # print "SS[0] = ", S_realDist

            # SS[0] = S_realDist
            for j in range( 0, N ):
                PointAndTangent = map.PointAndTangent         
                
                curv            = Curvature( SS[j], PointAndTangent )

                SS[j+1] = ( SS[j] + ( ( Planner.xPred[j,0]* np.cos(Planner.xPred[j,4])
                 - Planner.xPred[j,1]*np.sin(Planner.xPred[j,4]) ) / ( 1-Planner.xPred[j,3]*curv ) ) * planner_dt ) 

                Xref[j+1], Yref[j+1], Thetaref[j+1] = map.getGlobalPosition( SS[j+1], 0.0 )

            SS[0] = SS[1]


            Xlast = Xref[1]
            Ylast = Yref[1]
            Thetalast = Thetaref[1]

            for i in range(0,N):
                yaw[i]  = Thetaref[i] + Planner.xPred[i,4]
                xp[i]   = Xref[i] - Planner.xPred[i,3]*np.sin(yaw[i])
                yp[i]   = Yref[i] + Planner.xPred[i,3]*np.cos(yaw[i])        

            vel     = Planner.xPred[0:N,0]     
            curv    = Planner.xPred[0:N,2] / Planner.xPred[0:N,0]  


            endTimer    = datetime.datetime.now()
            deltaTimer  = endTimer - startTimer

            ELAPSD_TIME[Counter,:]      = deltaTimer.total_seconds()



            #####################################
            ## Plotting vehicle position:
            #####################################     

            if rospy.get_param("/TrajectoryPlanner/Visualization") == 1:
                line_trs.set_data(xp[0:N/2], yp[0:N/2])
                line_pred.set_data(xp[N/2:], yp[N/2:])
                l = 0.4; w = 0.2
                car_sim_x, car_sim_y = getCarPosition(xp[0], yp[0], yaw[0], w, l)
                # car_sim_x, car_sim_y = getCarPosition(xp[N-1], yp[N-1], yaw[N-1], w, l)
                rec_sim.set_xy(np.array([car_sim_x, car_sim_y]).T)
                fig.canvas.draw()

                StringValue = "vx = "+str(Planner.xPred[0,0])
                axtr.set_title(StringValue)


 


            #####################################
            ## Interpolating vehicle references:
            #####################################  
            interp_dt = 0.033
            time50ms = np.linspace(0, N*dt, num=N, endpoint=True)
            time33ms = np.linspace(0, N*dt, num=np.around(N*dt/interp_dt), endpoint=True)

            # X 
            f = interp1d(time50ms, xp, kind='cubic')
            X_interp = f(time33ms)  

            # Y
            f = interp1d(time50ms, yp, kind='cubic')
            Y_interp = f(time33ms)  

            # Yaw
            f = interp1d(time50ms, yaw, kind='cubic')
            Yaw_interp = f(time33ms)  

            # Velocity (Vx)
            f = interp1d(time50ms, vel, kind='cubic')
            Vx_interp = f(time33ms)

            # Curvature (K)
            f = interp1d(time50ms, curv, kind='cubic')
            Curv_interp = f(time33ms)     
            Curv_interp_filtered  = signal.filtfilt(b_filter, a_filter, Curv_interp, padlen=50)

            # plt.clf()
            # plt.figure(2)
            # plt.plot(Curv_interp, 'k-', label='input')
            # plt.plot(Curv_interp_filtered,  'c-', linewidth=1.5, label='pad')
            # plt.legend(loc='best')
            # plt.show()
            # plt.grid()

            # pdb.set_trace()



            #####################################
            ## Publishing vehicle references:
            #####################################   

            # refs.x_d        = xp
            # refs.y_d        = yp
            # refs.psi_d      = yaw
            # refs.vx_d       = vel 
            # refs.curv_d     = curv 
            refs.x_d        = X_interp
            refs.y_d        = Y_interp
            refs.psi_d      = Yaw_interp
            refs.vx_d       = Vx_interp 
            refs.curv_d     = Curv_interp_filtered             
            planning_refs.publish(refs)


            ALL_LOCAL_DATA[Counter,:]   = np.hstack(( Planner.xPred[0,:], Planner.uPred[0,:] ))
            References[Counter,:]       = np.hstack(( refs.x_d[0], refs.y_d[0], refs.psi_d[0], refs.vx_d[0], refs.curv_d[0] ))


            # Increase time counter and ROS sleep()
            TimeCounter     += 1
            PlannerCounter  += 1
            Counter         += 1


        rate.sleep()




    #############################################################
    day         = '30_7_19'
    num_test    = 'References'

    newpath     = '/home/euge/GitHub/barc/results_simu_test/Test_Planner/'+day+'/'+num_test+'/' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    np.savetxt(newpath+'/References.dat', References, fmt='%.5e')


    #############################################################
    # day         = '29_7_19'
    # num_test    = 'Test_1'

    # newpath     = '/home/euge/GitHub/barc/results_simu_test/Test_Planner/'+day+'/'+num_test+'/' 
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)

    # np.savetxt(newpath+'/ALL_LOCAL_DATA.dat', ALL_LOCAL_DATA, fmt='%.5e')
    # # np.savetxt(newpath+'/PREDICTED_DATA.dat', PREDICTED_DATA, fmt='%.5e')
    # # np.savetxt(newpath+'/GLOBAL_DATA.dat', GLOBAL_DATA, fmt='%.5e')
    # # np.savetxt(newpath+'/Complete_Vel_Vect.dat', Complete_Vel_Vect, fmt='%.5e')
    # # np.savetxt(newpath+'/References.dat', References, fmt='%.5e')
    # # np.savetxt(newpath+'/TLAPTIME.dat', TLAPTIME, fmt='%.5e')
    # np.savetxt(newpath+'/ELAPSD_TIME.dat', ELAPSD_TIME, fmt='%.5e')


    plt.close()

    # time50ms = np.linspace(0, (Counter-1)*dt, num=Counter-1, endpoint=True)
    # time33ms = np.linspace(0, (Counter-1)*dt, num=np.around((Counter-1)*dt/0.033), endpoint=True)
    # f = interp1d(time50ms, References[0:Counter-1,3], kind='cubic')

    plt.figure(2)
    plt.subplot(211)
    plt.plot(References[0:Counter-1,3], 'o')
    plt.legend(['Velocity'], loc='best')
    plt.grid()
    plt.subplot(212)
    # plt.plot(time33ms, f(time33ms), 'o')
    # plt.legend(['Velocity interpolated'], loc='best')    
    plt.plot(References[0:Counter-1,4], '-')
    plt.legend(['Curvature'], loc='best')    
    plt.show()
    plt.grid()
    # pdb.set_trace()


    quit() # final del while






# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

class RacingDataClass(object):
    """ Object collecting data from racing performance """

    def __init__(self):

        rospy.Subscriber('Racing_Info', Racing_Info, self.racing_info_callback, queue_size=1)

        self.LapNumber          = 0
        self.PlannerCounter     = 0

    def racing_info_callback(self,data):
        """ ... """      
        self.LapNumber          = data.LapNumber
        self.PlannerCounter     = data.PlannerCounter




# ===============================================================================================================================
# ==================================================== END OF MAIN ==============================================================
# ===============================================================================================================================

def InitFigure_XY(map, mode, HW):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    axtr = plt.axes()

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, HW)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -HW)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')


    line_cl,        = axtr.plot(xdata, ydata, '-k')
    line_gps_cl,    = axtr.plot(xdata, ydata, '--ob')
    line_tr,        = axtr.plot(xdata, ydata, '-or')
    line_trs,       = axtr.plot(xdata, ydata, '-og')
    line_pred,      = axtr.plot(xdata, ydata, '-or')
    
    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])

    rec = patches.Polygon(v, alpha=0.7,closed=True, fc='r', ec='k',zorder=10)
    # axtr.add_patch(rec)

    rec_sim = patches.Polygon(v, alpha=0.7,closed=True, fc='G', ec='k',zorder=10)

    if mode == "simulations":
        axtr.add_patch(rec_sim)

    plt.show()

    return fig, axtr, line_tr, line_pred, line_trs, line_cl, line_gps_cl, rec, rec_sim



def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l*np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y



def predicted_vectors_generation(Hp, x0, accel_rate, dt):

    Vx      = np.zeros((Hp+1, 1))
    Vx[0]   = x0[0]
    S       = np.zeros((Hp+1, 1))
    S[0]    = 0
    Vy      = np.zeros((Hp+1, 1))
    Vy[0]   = x0[1]
    W       = np.zeros((Hp+1, 1))
    W[0]    = x0[2]
    Ey      = np.zeros((Hp+1, 1))
    Ey[0]   = x0[3]
    Epsi    = np.zeros((Hp+1, 1))
    Epsi[0] = x0[4]

    Accel   = 0.1
    curv    = 0

    for i in range(0, Hp): 
        Vy[i+1]      = x0[1]  
        W[i+1]       = x0[2] 
        Ey[i+1]      = x0[3] 
        Epsi[i+1]    = x0[4] 

    Accel   = Accel + np.array([ (accel_rate * i) for i in range(0, Hp)])

    for i in range(0, Hp):
        Vx[i+1]    = Vx[i] + Accel[i] * dt
        S[i+1]      = S[i] + ( (Vx[i]*np.cos(Epsi[i]) - Vy[i]*np.sin(Epsi[i])) / (1-Ey[i]*curv) ) * dt

    # print "Vx = ", Vx
    # print "Vy = ", np.transpose(Vy)
    # print "W = ", W

    # pdb.set_trace()

    xx  = hstack([ Vx, Vy, W, Ey, Epsi, S])    

    uu = np.zeros(( Hp, 1 )) 

    return xx, uu





if __name__ == "__main__":

    try:    
        main()
        
    except rospy.ROSInterruptException:
        pass
