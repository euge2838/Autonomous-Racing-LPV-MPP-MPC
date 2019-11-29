#!/usr/bin/env python
"""
    File name: TrajectoryPlanner.py
    Author: Eugenio Alcala
    Email: eugenio.alcala@upc.edu (euge2838@gmail.com)
    Date:18/07/2019
    Python Version: 2.7.12
"""

import os
import sys
import datetime
import rospy
import numpy as np
import scipy.io as sio
import pdb
import pickle
import matplotlib.pyplot as plt

sys.path.append(sys.path[0]+'/ControllerObject')
sys.path.append(sys.path[0]+'/Utilities')

from barc.msg import My_Planning, Racing_Info

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def main():

    # Initializa ROS node
    rospy.init_node("Trajectory_Planner")

    planning_refs   = rospy.Publisher('My_Planning', My_Planning, queue_size=1)

    refs            = My_Planning()
    racing_data     = RacingDataClass()

    loop_rate       = 30.0
    dt              = 1.0/loop_rate
    rate            = rospy.Rate(loop_rate)

    N               = rospy.get_param("/control/N")

    Planning_Track  = rospy.get_param("/TrajectoryPlanner/Planning_Track")

    planning_path   = '/home/euge/GitHub/barc/workspace/src/barc/src/data/Planner_Refs' 

    if Planning_Track == 1:
        ## TRACK 1 (L-shape, max Vel = 2.8 m/s) : (22 sept moorning)
        CURV_Planner    = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['pCURV']
        VEL_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['pnew_Vx']
        X_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['pxp']
        Y_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['pyp']
        PSI_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['pyaw']
        Lap_Lengths_Planner = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['lap_lengths']
        Zero_Cross_Index    = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_1_.mat')['zero_cross_index']
        length_track1       = Lap_Lengths_Planner[0][0]        
        TrackCounter        = Zero_Cross_Index[0][0]
       

    elif Planning_Track == 2:
        ## TRACK 2 (L-shape 2 laps, max Vel = 2.6 m/s) : (18 sept afternoon)
        CURV_Planner    = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['pCURV']
        VEL_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['pnew_Vx']
        X_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['pxp']
        Y_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['pyp']
        PSI_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['pyaw']
        Lap_Lengths_Planner = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_2_.mat')['lap_lengths']
        length_track1       = Lap_Lengths_Planner[0][0]
        TrackCounter    = 275  # Track 2


    elif Planning_Track == 3:
        ## TRACK 3 (L-shape 2 laps, max Vel = 2.7 m/s) : (19 sept afternoon)
        CURV_Planner    = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['pCURV']
        VEL_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['pnew_Vx']
        X_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['pxp']
        Y_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['pyp']
        PSI_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['pyaw']
        Lap_Lengths_Planner = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_3_.mat')['lap_lengths']
        length_track1       = Lap_Lengths_Planner[0][0]        
        TrackCounter    = 275  # Track 3


    elif Planning_Track == 4:
        ## TRACK 4 (L-shape 2 laps, max Vel = 2.4 m/s) : (20 sept morning)
        CURV_Planner    = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['pCURV']
        VEL_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['pnew_Vx']
        X_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['pxp']
        Y_Planner       = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['pyp']
        PSI_Planner     = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['pyaw']
        Lap_Lengths_Planner = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['lap_lengths']
        Zero_Cross_Index    = sio.loadmat(planning_path+'/GOOD_PLAN_REFS_4_.mat')['zero_cross_index']
        length_track1       = Lap_Lengths_Planner[0][0]        
        TrackCounter        = Zero_Cross_Index[0][0]

    # refs.x_d        = np.zeros((1,N+1)) 
    # refs.y_d        = np.zeros((1,N+1)) 
    # refs.psi_d      = np.zeros((1,N+1)) 
    # refs.vx_d       = np.zeros((1,N+1)) 
    # refs.curv_d     = np.zeros((1,N+1)) 
    refs.x_d        = []
    refs.y_d        = []
    refs.psi_d      = []
    refs.vx_d       = []
    refs.curv_d     = []    

    while not rospy.is_shutdown():

        # if racing_data.LapNumber >= 1:
        #     refs.x_d        = X_Planner[0, int(racing_data.PlannerCounter):int(racing_data.PlannerCounter)+N+1]
        #     # print "refs.x_d = ", refs.x_d
        #     refs.y_d        = Y_Planner[0, int(racing_data.PlannerCounter):int(racing_data.PlannerCounter)+N+1]
        #     refs.psi_d      = PSI_Planner[0, int(racing_data.PlannerCounter):int(racing_data.PlannerCounter)+N+1]
        #     refs.vx_d       = VEL_Planner[0, int(racing_data.PlannerCounter):int(racing_data.PlannerCounter)+N+1]
        #     refs.curv_d     = CURV_Planner[0, int(racing_data.PlannerCounter):int(racing_data.PlannerCounter)+N]



        # else:
        #     refs.x_d        = np.zeros((1,N+1)) 
        #     refs.y_d        = np.zeros((1,N+1)) 
        #     refs.psi_d      = np.zeros((1,N+1)) 
        #     refs.vx_d       = np.zeros((1,N+1)) 
        #     refs.curv_d     = np.zeros((1,N+1)) 

            # refs.x_d        = np.zeros(N+1,) 
            # refs.y_d        = np.zeros(N+1,) 
            # refs.psi_d      = np.zeros(N+1,) 
            # refs.vx_d       = np.zeros(N+1,) 
            # refs.curv_d     = np.zeros(N+1,) 
            
        planning_refs.publish(refs)

        rate.sleep()



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




if __name__ == "__main__":

    try:    
        main()
        
    except rospy.ROSInterruptException:
        pass
