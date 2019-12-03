#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that (1) you retain this notice
# and (2) you provide clear attribution to UC Berkeley, including a link
# to http://barc-project.com
#
# Author: J. Noonan
# Email: jpnoonan@berkeley.edu
# 
# Updated by Eugenio Alcala
# Email: euge2838@gmail.com
#
# This code provides a way to see the car's trajectory, orientation, and velocity profile in 
# real time with referenced to the track defined a priori.
#
# ---------------------------------------------------------------------------
import sys
sys.path.append(sys.path[0]+'/ControllerObject')
sys.path.append(sys.path[0]+'/Utilities')

import rospy
from marvelmind_nav.msg import hedge_pos, hedge_imu_fusion
import numpy as np
from trackInitialization import Map
from barc.msg import pos_info, prediction, simulatorStates, My_Planning, Racing_Info
import matplotlib.pyplot as plt    
import pdb
import matplotlib.patches as patches

import scipy.io as sio


np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

# For printing with less decimals
# print "Vx = {0:0.2f}".format(estimatedStates[0])


def main():

    rospy.init_node("realTimePlotting")

    mode        = rospy.get_param("/control/mode") # testing planner
    plotGPS     = rospy.get_param("/visualization/plotGPS")   

    data        = Estimation_Mesures_Planning_Data(mode, plotGPS)        
    map         = Map()

    loop_rate   = 30.0
    rate        = rospy.Rate(loop_rate)

    vSwitch     = 1.3     
    psiSwitch   = 1.2    

    StateView   = False    

    if StateView == True:

        fig, linevx, linevy, linewz, lineepsi, lineey, line_tr, line_pred = _initializeFigure(map)
    else:

        Planning_Track = rospy.get_param("/TrajectoryPlanner/Planning_Track")
        # Planning_Track = 1
        planning_path   = '/home/euge/GitHub/barc/workspace/src/barc/src/data/Planner_Refs' 
        if Planning_Track == 1:
            X_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['pxp']
            Y_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['pyp']
        elif Planning_Track == 2:
            X_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['pxp']
            Y_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['pyp']
        elif Planning_Track == 3:
            X_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['pxp']
            Y_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['pyp']            
        elif Planning_Track == 4:
            X_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['pxp']
            Y_Planner_Pts  = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['pyp'] 

        ( fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_gps_cl, rec,
         rec_sim, rec_planning ) = _initializeFigure_xy(map, mode, X_Planner_Pts, Y_Planner_Pts)

    ClosedLoopTraj_gps_x = []
    ClosedLoopTraj_gps_y = []
    ClosedLoopTraj_x = []
    ClosedLoopTraj_y = []

    flagHalfLap = False

    while not rospy.is_shutdown():
        estimatedStates = data.readEstimatedData()
        s, ey, epsi, insideMap = map.getLocalPosition(estimatedStates[3], estimatedStates[4], estimatedStates[5])

        if s > map.TrackLength / 2:
            flagHalfLap = True

        if (s < map.TrackLength / 4) and (flagHalfLap == True): # New lap
            ClosedLoopTraj_gps_x = []
            ClosedLoopTraj_gps_y = []
            ClosedLoopTraj_x = []
            ClosedLoopTraj_y = []
            flagHalfLap = False

        x = estimatedStates[3]
        y = estimatedStates[4]

        if plotGPS == True:
            ClosedLoopTraj_gps_x.append(data.MeasuredData[0])
            ClosedLoopTraj_gps_y.append(data.MeasuredData[1])

        ClosedLoopTraj_x.append(x) 
        ClosedLoopTraj_y.append(y)

        psi = estimatedStates[5]
        l = 0.4; w = 0.2


        line_cl.set_data(ClosedLoopTraj_x, ClosedLoopTraj_y)

        if plotGPS == True:
            line_gps_cl.set_data(ClosedLoopTraj_gps_x, ClosedLoopTraj_gps_y)

        if StateView == True:
            linevx.set_data(0.0, estimatedStates[0])
            linevy.set_data(0.0, estimatedStates[1])
            linewz.set_data(0.0, estimatedStates[2])
            lineepsi.set_data(0.0, epsi)
            lineey.set_data(0.0, ey)

        # Plotting the estimated car:        
        line_tr.set_data(estimatedStates[3], estimatedStates[4])
        car_x, car_y = getCarPosition(x, y, psi, w, l)
        rec.set_xy(np.array([car_x, car_y]).T)        

        # Plotting the planner car and trajectory:
        # line_planning.set_data(data.x_d[:], data.y_d[:])
        # car_x, car_y = getCarPosition(data.x_d[0], data.y_d[0], data.psi_d[0], w, l)
        # rec_planning.set_xy(np.array([car_x, car_y]).T)        
        

        
        if mode == "simulations":
            x_sim   = data.sim_x[-1]
            y_sim   = data.sim_y[-1]
            psi_sim = data.sim_psi[-1]
            car_sim_x, car_sim_y = getCarPosition(x_sim, y_sim, psi_sim, w, l)
            rec_sim.set_xy(np.array([car_sim_x, car_sim_y]).T)

        # StringValue = "Vx = {0:0.2f}".format(estimatedStates[0]), " Vy = {0:0.2f}".format(estimatedStates[1]), "psiDot = {0:0.2f}".format(estimatedStates[2])

        StringValue = "Vx = {0:0.2f}".format(estimatedStates[0]), " Vx_Ref = {0:0.2f}".format(data.vx_d[0])
        axtr.set_title(StringValue)
        
        if insideMap == 1:
            fig.canvas.draw()

        rate.sleep()



def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y





class Estimation_Mesures_Planning_Data():
    """Object collecting closed loop data points
    Attributes:
        updateInitialConditions: function which updates initial conditions and clear the memory
    """
    def __init__(self, mode, plotGPS):

        if mode == "simulations":
            rospy.Subscriber("simulatorStates", simulatorStates, self.simState_callback)

        if plotGPS == True:
            rospy.Subscriber("hedge_pos", hedge_pos, self.gps_callback)

        rospy.Subscriber("pos_info", pos_info, self.pos_info_callback)
        rospy.Subscriber("OL_predictions", prediction, self.prediction_callback)

        rospy.Subscriber("My_Planning", My_Planning, self.My_Planning_callback)
        # rospy.Subscriber("Racing_Info", Racing_Info, self.Racing_Info_callback)


        N   = rospy.get_param("/TrajectoryPlanner/N")

        self.s      = []
        self.ey     = []
        self.epsi   = []

        self.x_d    = np.zeros(N+1) 
        self.y_d    = np.zeros(N+1) 
        self.psi_d  = np.zeros(N+1) 
        self.vx_d   = np.zeros(N+1) 
        
        self.LapNumber      = 0   
        self.PlannerCounter = 0  

        self.MeasuredData = [0.0, 0.0]
        self.EstimatedData= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.sim_x   = [0.0]
        self.sim_y   = [0.0]
        self.sim_psi = [0.0]

    def simState_callback(self, msg):
        self.sim_x.append(msg.x)
        self.sim_y.append(msg.y)
        self.sim_psi.append(msg.psi)

    def gps_callback(self, msg):
        self.MeasuredData = [msg.x_m, msg.y_m]

    def pos_info_callback(self, msg):
        self.EstimatedData = [msg.v_x, msg.v_y, msg.psiDot, msg.x, msg.y, msg.psi]

    def prediction_callback(self, msg):
        self.s      = msg.s
        self.ey     = msg.ey
        self.epsi   = msg.epsi 

    def My_Planning_callback(self, msg):
        self.x_d    = msg.x_d
        self.y_d    = msg.y_d
        self.psi_d  = msg.psi_d      
        self.vx_d   = msg.vx_d

    # def Racing_Info_callback(self, msg):
    #     self.LapNumber      = msg.LapNumber
    #     self.PlannerCounter = msg.PlannerCounter

    def readEstimatedData(self):
        return self.EstimatedData








    
# ===================================================================================================================================== #
# ============================================================= Internal Functions ==================================================== #
# ===================================================================================================================================== #







def _initializeFigure_xy(map, mode, X_Planner_Pts, Y_Planner_Pts):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    axtr = plt.axes()

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    # Points1 = np.zeros((Points, 2))
    # Points2 = np.zeros((Points, 2))
    # Points0 = np.zeros((Points, 2))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))    

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')

    # These lines plot the planned offline trajectory in the main figure:
    # plt.plot(X_Planner_Pts[0, 0:290], Y_Planner_Pts[0, 0:290], '--r')
    # plt.plot(X_Planner_Pts[0, 290:460], Y_Planner_Pts[0, 290:460], '--r')
    # plt.plot(X_Planner_Pts[0, :], Y_Planner_Pts[0, :], '--r')


    line_cl,        = axtr.plot(xdata, ydata, '-k')
    line_gps_cl,    = axtr.plot(xdata, ydata, '--ob')  # Plots the traveled positions
    line_tr,        = axtr.plot(xdata, ydata, '-or')       # Plots the current positions
    line_SS,        = axtr.plot(xdata, ydata, 'og')
    line_pred,      = axtr.plot(xdata, ydata, '-or')
    line_planning,  = axtr.plot(xdata, ydata, '-ok')
    
    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])

    rec = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10)
    axtr.add_patch(rec)

    # Vehicle:
    rec_sim = patches.Polygon(v, alpha=0.7, closed=True, fc='G', ec='k', zorder=10)

    if mode == "simulations":
        axtr.add_patch(rec_sim)    

    # Planner vehicle:
    rec_planning = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)
    # axtr.add_patch(rec_planning)



    plt.show()

    return fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_gps_cl, rec, rec_sim, rec_planning






def _initializeFigure(map):
    xdata = []; ydata = []
    plt.ion()
    fig = plt.figure(figsize=(40,20))

    axvx = fig.add_subplot(3, 2, 1)
    linevx, = axvx.plot(xdata, ydata, 'or-')
    axvx.set_ylim([0, 1.5])
    plt.ylabel("vx")
    plt.xlabel("t")

    axvy = fig.add_subplot(3, 2, 2)
    linevy, = axvy.plot(xdata, ydata, 'or-')
    plt.ylabel("vy")
    plt.xlabel("s")

    axwz = fig.add_subplot(3, 2, 3)
    linewz, = axwz.plot(xdata, ydata, 'or-')
    plt.ylabel("wz")
    plt.xlabel("s")

    axepsi = fig.add_subplot(3, 2, 4)
    lineepsi, = axepsi.plot(xdata, ydata, 'or-')
    axepsi.set_ylim([-np.pi/2,np.pi/2])
    plt.ylabel("epsi")
    plt.xlabel("s")

    axey = fig.add_subplot(3, 2, 5)
    lineey, = axey.plot(xdata, ydata, 'or-')
    axey.set_ylim([-map.width,map.width])
    plt.ylabel("ey")
    plt.xlabel("s")

    Points = np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    axtr = fig.add_subplot(3, 2, 6)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    line_tr, = axtr.plot(xdata, ydata, '-or')
    line_pred, = axtr.plot(xdata, ydata, '-or')
    
    plt.show()

    return fig, linevx, linevy, linewz, lineepsi, lineey, line_tr, line_pred



# ===================================================================================================================================== #
# ========================================================= End of Internal Functions ================================================= #
# ===================================================================================================================================== #

if __name__ == '__main__':
    try:
        main()

    except rospy.ROSInterruptException:
        pass
