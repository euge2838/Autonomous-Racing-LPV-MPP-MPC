#!/usr/bin/env python
"""
    File name: Learning-LPV-MPC.py
    Author: Eugenio Alcala
    Email: eugenio.alcala@upc.edu.edu
    Date: 09/30/2018
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

from trackInitialization import Map, wrap
from barc.msg import ECU, prediction, Racing_Info, My_Planning
from utilities import Regression, Curvature
from dataStructures import LMPCprediction, EstimatorData, PlanningData, ClosedLoopDataObj
from PathFollowingLPVMPC import PathFollowingLPV_MPC, ABC_computation_5SV

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    
    rospy.init_node("LPV-MPC")
    
    input_commands  = rospy.Publisher('ecu', ECU, queue_size=1)
    pred_treajecto  = rospy.Publisher('OL_predictions', prediction, queue_size=1)
    racing_info     = rospy.Publisher('Racing_Info', Racing_Info, queue_size=1)

    mode            = rospy.get_param("/control/mode")
    N               = rospy.get_param("/control/N")

    loop_rate       = 30.0
    dt              = 1.0/loop_rate
    rate            = rospy.Rate(loop_rate)


    ## TODO: change this in order to be taken from the launch file
    Steering_Delay  = 0 #3
    Velocity_Delay  = 0

    #Steering_Delay  = int(rospy.get_param("/simulator/delay_df")/dt)

    NN_LPV_MPC      = False

#########################################################
#########################################################

    # OFFLINE planning from LPV-MPP racing:
    planning_path   = '/home/euge/GitHub/barc/workspace/src/barc/src/data/Planner_Refs' 

    ## TRACK (OVAL, Racing Planner) : (30 July 19)
    CURV_Planner    = sio.loadmat(planning_path+'/LPV_MPC_PLANNING_1_.mat')['pCURV']
    VEL_Planner     = sio.loadmat(planning_path+'/LPV_MPC_PLANNING_1_.mat')['pnew_Vx']
    X_Planner       = sio.loadmat(planning_path+'/LPV_MPC_PLANNING_1_.mat')['pxp']
    Y_Planner       = sio.loadmat(planning_path+'/LPV_MPC_PLANNING_1_.mat')['pyp']
    PSI_Planner     = sio.loadmat(planning_path+'/LPV_MPC_PLANNING_1_.mat')['pyaw']    



#########################################################
#########################################################


    # Objects initializations

    OL_predictions  = prediction()
    cmd             = ECU()                                              # Command message
    rac_info        = Racing_Info()
    cmd.servo       = 0.0
    cmd.motor       = 0.0
    ClosedLoopData  = ClosedLoopDataObj(dt, 6000, 0)                     # Closed-Loop Data
    estimatorData   = EstimatorData()
    map             = Map()                                              # Map
    planning_data   = PlanningData()
    
    first_it        = 1
    NumberOfLaps    = 10    

    # Initialize variables for main loop
    GlobalState     = np.zeros(6)
    LocalState      = np.zeros(6)
    HalfTrack       = 0
    LapNumber       = 0
    RunController   = 1
    Counter         = 0
    CounterRacing   = 0
    uApplied        = np.array([0.0, 0.0])
    oldU            = np.array([0.0, 0.0])

    RMSE_ve         = np.zeros(NumberOfLaps)
    RMSE_ye         = np.zeros(NumberOfLaps)
    RMSE_thetae     = np.zeros(NumberOfLaps)
    RMSE_acc_y      = np.zeros(NumberOfLaps)
    RMSE_matrix     = np.zeros(NumberOfLaps)
    Norm_matrix     = np.zeros((NumberOfLaps,3))

    ALL_LOCAL_DATA  = np.zeros((2000,8))      # [vx vy psidot thetae s ye vxaccel vyaccel udelta uaccel]
    GLOBAL_DATA     = np.zeros((2000,3))       # [x y psi]
    PREDICTED_DATA  = np.zeros((2000,120))     # [vx vy psidot thetae s ye] presicted to N steps
    TLAPTIME        = np.zeros((30,1)) 
    ELAPSD_TIME     = np.zeros((2000,1))

    IDENT_DATA      = np.zeros((2000,5))
    Data_for_RMSE   = np.zeros((2000,4))

    References      = np.array([ 0.0, 0.0, 0.0, 1.0 ])

    # Loop running at loop rate
    TimeCounter     = 0
    PlannerCounter  = 0
    count           = True
    index           = 0
    start_LapTimer  = datetime.datetime.now()

    insideTrack     = 1

    rospy.sleep(1)   # Soluciona los problemas de inicializacion esperando a que el estimador se inicialice bien

    vel_ref         = 1
    Cf_new          = 60
    SS              = 0



###----------------------------------------------------------------###
    ### PATH TRACKING TUNING:
    ### 33 ms - 20 Hp 
    Q  = np.diag([100.0, 1.0, 1.0, 20.0, 0.0, 900.0])
    R  = 0.5*0.5 * np.diag([1.0, 1.0])  # delta, a
    dR = 1.5*25 * np.array([1.3, 1.0])  # Input rate cost u
    Controller  = PathFollowingLPV_MPC(Q, R, dR, N, vel_ref, dt, map, "OSQP", Steering_Delay, Velocity_Delay)

    # ### RACING TAJECTORY TRACKING TUNING:
    # ### 33 ms - 20 Hp 
    Q  = np.diag([400.0, 1.0, 1.0, 20.0, 0.0, 1100.0])
    R  = 0.0 * np.diag([1.0, 1.0])    # delta, a
    dR =  np.array([ 100.0, 45.0 ])  # Input rate cost u
    # dR =  np.array([ 10.0, 10.0 ])  # Input rate cost u
    Controller_TT  = PathFollowingLPV_MPC(Q, R, dR, N, vel_ref, dt, map, "OSQP", Steering_Delay, Velocity_Delay)

    ### RACING TAJECTORY TRACKING TUNING:
    ### 33 ms - 20 Hp 
    # Q  = np.diag([800.0, 1.0, 2.0, 10.0, 0.0, 1200.0])
    # R  = np.diag([0.0, 0.0])    # delta, a
    # # R  = np.array([146.0, 37.0])    # Input rate cost u
    # dR = np.array([20, 30.0])  # Input rate cost u
    # Controller_TT  = PathFollowingLPV_MPC(Q, R, dR, N, vel_ref, dt, map, "OSQP", Steering_Delay, Velocity_Delay)





###----------------------------------------------------------------###
###----------------------------------------------------------------###
###----------------------------------------------------------------###
###----------------------------------------------------------------###






    L_LPV_States_Prediction = np.zeros((N,6))
    LPV_States_Prediction   = np.zeros((N,6))

    while (not rospy.is_shutdown()) and RunController == 1:    
        # Read Measurements
        GlobalState[:] = estimatorData.CurrentState  # The current estimated state vector [vx vy w x y psi]
        LocalState[:]  = estimatorData.CurrentState  # [vx vy w x y psi]


        if LocalState[0] < 0.01:
            LocalState[0] = 0.01

        if LapNumber == 0: # Path tracking:
            # OUT: s, epsi, ey      IN: x, y, psi
            LocalState[4], LocalState[3], LocalState[5], insideTrack = map.getLocalPosition(GlobalState[3], GlobalState[4], GlobalState[5])

            # Check if the lap has finished
            if LocalState[4] >= 3*map.TrackLength/4:
                HalfTrack = 1

            new_Ref     = np.array([ 0.0, 0.0, 0.0, 1.0 ])
            References  = np.vstack(( References, new_Ref ))   # Simplemente se usa para guardar datos en fichero


        else: # Trajectory tracking:

            GlobalState[5] = GlobalState[5]-2*np.pi*LapNumber   ## BODY FRAME ERROR COMPUTATION:

            GlobalState[5] = wrap(GlobalState[5])

            PSI_Planner[0,PlannerCounter] = wrap(PSI_Planner[0,PlannerCounter])

            # Offline planning data:
            # OUT: s, ex, ey, epsi
            # LocalState[4], Xerror, LocalState[3], LocalState[5] = Body_Frame_Errors(GlobalState[3], 
            #      GlobalState[4], GlobalState[5], X_Planner[0,PlannerCounter], Y_Planner[0,PlannerCounter],
            #      PSI_Planner[0,PlannerCounter], SS, LocalState[0], LocalState[1], CURV_Planner[0, PlannerCounter], dt )

            # print "Desired yaw = ", PSI_Planner[0,PlannerCounter], " and Real yaw = ", GlobalState[5]



            # Online planning data:
            new_Ref     = np.array([ planning_data.x_d[0], planning_data.y_d[0], planning_data.psi_d[0], planning_data.vx_d[0] ])
            References  = np.vstack(( References, np.transpose(new_Ref) ))

            max_window = 0

            if index <= max_window:
                if index == 0:
                    X_REF_vector    = planning_data.x_d[0:N+max_window]
                    Y_REF_vector    = planning_data.y_d[0:N+max_window]
                    YAW_REF_vector  = planning_data.psi_d[0:N+max_window]
                    VEL_REF_vector  = planning_data.vx_d[0:N+max_window]
                    CURV_REF_vector = planning_data.curv_d[0:N+max_window]  

                x_ref       = X_REF_vector[index:index+N]
                y_ref       = Y_REF_vector[index:index+N]
                yaw_ref     = YAW_REF_vector[index:index+N]
                vel_ref     = VEL_REF_vector[index:index+N]
                curv_ref    = CURV_REF_vector[index:index+N]
                index += 1
            else:
                index = 0


            # OUT: s, ex, ey, epsi
            LocalState[4], Xerror, LocalState[5], LocalState[3] = Body_Frame_Errors(GlobalState[3], 
                 GlobalState[4], GlobalState[5], x_ref[0], y_ref[0], yaw_ref[0], SS, LocalState[0],
                 LocalState[1], curv_ref[0], dt )   

            SS = LocalState[4] 




        ### END OF THE LAP.
        # PATH TRACKING:
        if ( HalfTrack == 1 and (LocalState[4] <= map.TrackLength/4)):    
            HalfTrack       = 0
            LapNumber       += 1 
            SS              = 0
            PlannerCounter  = 0
            TimeCounter     = 0                      

            TotalLapTime = datetime.datetime.now() - start_LapTimer
            print 'Lap completed in',TotalLapTime , 'seconds. Starting lap:', LapNumber, '\n'
            TLAPTIME[LapNumber-1] = TotalLapTime.total_seconds()
            start_LapTimer = datetime.datetime.now()


        # RACING:
        elif ( (LapNumber >= 1) and (np.abs(GlobalState[3])<0.1) and (LocalState[4] >= (map.TrackLength - map.TrackLength/10)) ):  
            LapNumber       += 1 
            SS              = 0
            # PlannerCounter  = TrackCounter

            TotalLapTime = datetime.datetime.now() - start_LapTimer
            print 'Lap completed in',TotalLapTime , 'seconds. Starting lap:', LapNumber , '. PlannerCounter = ', PlannerCounter, '\n'
            TLAPTIME[LapNumber-1] = TotalLapTime.total_seconds()
            start_LapTimer = datetime.datetime.now()
            
            if LapNumber > NumberOfLaps:
                print('RunController = 0')
                RunController = 0





        startTimer = datetime.datetime.now()
                              
        oldU = uApplied
        uApplied = np.array([cmd.servo, cmd.motor])

        if LapNumber == 0:    
            Controller.OldSteering.append(cmd.servo) # meto al final del vector
            Controller.OldAccelera.append(cmd.motor)
            Controller.OldSteering.pop(0)
            Controller.OldAccelera.pop(0)    
        else:
            Controller_TT.OldSteering.append(cmd.servo) # meto al final del vector
            Controller_TT.OldAccelera.append(cmd.motor)
            Controller_TT.OldSteering.pop(0)
            Controller_TT.OldAccelera.pop(0)

        ### Publish input ###
        input_commands.publish(cmd)




        ###################################################################################################
        ###################################################################################################             


        if first_it < 10: 
            # print('Iteration = ', first_it)
            vel_ref     = np.ones(N)
            xx, uu      = predicted_vectors_generation(N, LocalState, uApplied, dt)
            Controller.solve(LocalState[0:6], xx, uu, NN_LPV_MPC, vel_ref, 0, 0, 0, first_it)
            first_it    = first_it + 1
            # print('---> Controller.uPred = ', Controller.uPred)
            

            Controller.OldPredicted = np.hstack((Controller.OldSteering[0:len(Controller.OldSteering)-1], Controller.uPred[Controller.steeringDelay:Controller.N,0]))
            Controller.OldPredicted = np.concatenate((np.matrix(Controller.OldPredicted).T, np.matrix(Controller.uPred[:,1]).T), axis=1)

        else:

            NN_LPV_MPC  = False
            if LapNumber == 0:      # PATH TRACKING : First lap at 1 m/s                
                vel_ref         = np.ones(N+1)
                curv_ref        = np.zeros(N)

                LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref, curv_ref, Cf_new, LapNumber)      

                Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, Controller.uPred, NN_LPV_MPC, vel_ref, A_L, B_L, C_L, first_it)


                # IDENT_DATA[Counter,:] = [LocalState[0], LocalState[1], LocalState[2], uApplied[0], uApplied[1]]

                Controller_TT.uPred = Controller.uPred

            else:    # TRAJECTORY TRACKING

                # Offline planning data:
                # vel_ref     = VEL_Planner[0, PlannerCounter:PlannerCounter+N+1]
                # curv_ref    = CURV_Planner[0, PlannerCounter:PlannerCounter+N]



                # VEL_REF_vector     = planning_data.vx_d[0:N+10]
                # CURV_REF_vector    = planning_data.curv_d[0:N+10]  

                # if index < 10:
                #     vel_ref     = VEL_REF_vector[index:index+N]
                #     curv_ref    = CURV_REF_vector[index:index+N]
                #     index += 1
                # else:
                #     index = 0


                # Online planning data:curv_ref
                # vel_ref     = planning_data.vx_d[0:N]
                # curv_ref    = planning_data.curv_d[0:N]              

                LPV_States_Prediction, A_L, B_L, C_L = Controller_TT.LPVPrediction(LocalState[0:6], Controller_TT.uPred, vel_ref, curv_ref, Cf_new, LapNumber)      

                Controller_TT.solve(LocalState[0:6], 0.0, Controller_TT.uPred, NN_LPV_MPC, vel_ref, A_L, B_L, C_L, first_it)



        ###################################################################################################
        ###################################################################################################
        





        if first_it > 19: 
            new_LPV_States_Prediction = LPV_States_Prediction[0, :]
            for i in range(1,N):
                new_LPV_States_Prediction = np.hstack((new_LPV_States_Prediction, LPV_States_Prediction[i,:]))
            PREDICTED_DATA[Counter,:] = new_LPV_States_Prediction

        if LapNumber == 0:    
            cmd.servo = Controller.uPred[0 + Controller.steeringDelay, 0]
            cmd.motor = Controller.uPred[0 + Controller.velocityDelay, 1]   
        else:
            cmd.servo = Controller_TT.uPred[0 + Controller.steeringDelay, 0]
            cmd.motor = Controller_TT.uPred[0 + Controller.velocityDelay, 1]

        #print Controller.uPred[0, :]

        endTimer = datetime.datetime.now(); 
        deltaTimer = endTimer - startTimer


        # ESTO HABRA QUE RESCATARLO PARA PRUEBAS FISICAS...
        # else:   # If car out of the track
        #     cmd.servo = 0
        #     cmd.motor = 0
        #     input_commands.publish(cmd)

        ## Record Prediction
        if LapNumber < 1:
            OL_predictions.s = Controller.xPred[:, 4]
            OL_predictions.ex = []
            OL_predictions.ey = Controller.xPred[:, 5]
            OL_predictions.epsi = Controller.xPred[:, 3]
        else:
            OL_predictions.s = Controller_TT.xPred[:, 4]
            OL_predictions.ex = []
            OL_predictions.ey = Controller_TT.xPred[:, 5]
            OL_predictions.epsi = Controller_TT.xPred[:, 3] 
            ALL_LOCAL_DATA[CounterRacing,:]   = np.hstack((LocalState, uApplied))
            GLOBAL_DATA[CounterRacing,:]      = [GlobalState[3], GlobalState[4], GlobalState[5]]
            CounterRacing += 1


        pred_treajecto.publish(OL_predictions)

        # ClosedLoopData.addMeasurement(GlobalState, LocalState, uApplied, Counter, deltaTimer.total_seconds())
        # Data_for_RMSE[TimeCounter,:] = [ LocalState[0], LocalState[5], LocalState[3], LocalState[7]]
        ELAPSD_TIME[Counter,:] = deltaTimer.total_seconds()


        # Publishing important info about the racing:
        rac_info.LapNumber      = LapNumber
        # rac_info.PlannerCounter = PlannerCounter
        racing_info.publish(rac_info)
        

        # Increase time counter and ROS sleep()

        # print PlannerCounter

        TimeCounter     += 1
        PlannerCounter  += 1
        # if PlannerCounter > 580: #offline racing planning
        if PlannerCounter > 999:
            cmd.servo = 0
            cmd.motor = 0
            input_commands.publish(cmd)
            day         = '31_7_19'
            num_test    = 'Test_1'
            newpath     = '/home/euge/GitHub/barc/results_simu_test/'+day+'/'+num_test+'/' 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            np.savetxt(newpath+'/ALL_LOCAL_DATA.dat', ALL_LOCAL_DATA, fmt='%.5e')
            np.savetxt(newpath+'/PREDICTED_DATA.dat', PREDICTED_DATA, fmt='%.5e')
            np.savetxt(newpath+'/GLOBAL_DATA.dat', GLOBAL_DATA, fmt='%.5e')
            np.savetxt(newpath+'/References.dat', References, fmt='%.5e')
            np.savetxt(newpath+'/TLAPTIME.dat', TLAPTIME, fmt='%.5e')
            np.savetxt(newpath+'/ELAPSD_TIME.dat', ELAPSD_TIME, fmt='%.5e')            
            quit()

        Counter  += 1
        rate.sleep()

    # END WHILE

    # Save Data
    # file_data = open(sys.path[0]+'/data/'+mode+'/ClosedLoopData'+"LPV_MPC"+'.obj', 'wb')
    # pickle.dump(ClosedLoopData, file_data)
    # pickle.dump(Controller, file_data)

#############################################################

    # file_data = open(newpath+'/ClosedLoopDataLPV_MPC.obj', 'wb')
    # pickle.dump(ClosedLoopData, file_data)
    # pickle.dump(Controller, file_data)    
    # file_data.close()

    # INFO_data = open(newpath+'/INFO.txt', 'wb')
    # INFO_data.write('Running in Track number ')
    # INFO_data.write('%d \n' % Planning_Track)    
    # #INFO_data.write(' ')
    # INFO_data.close()

#############################################################

    # plotTrajectory(map, ClosedLoopData, Complete_Vel_Vect)
    
    quit()










# ===============================================================================================================================
# ==================================================== END OF MAIN ==============================================================
# ===============================================================================================================================

def Body_Frame_Errors (x, y, psi, xd, yd, psid, s0, vx, vy, curv, dt):

    ex = (x-xd)*np.cos(psid) + (y-yd)*np.sin(psid)

    ey = -(x-xd)*np.sin(psid) + (y-yd)*np.cos(psid)

    epsi = wrap(psi - psid)

    #s = s0 + np.sqrt(vx*vx + vy*vy) * dt
    s = s0 + ( (vx*np.cos(epsi) - vy*np.sin(epsi)) / (1-ey*curv) ) * dt

    return s, ex, ey, epsi

  

def predicted_vectors_generation(Hp, LocalState, uu_Old, Ts):

    xx = np.array([[LocalState[0]+0.05, LocalState[1], LocalState[2], 0.0001, LocalState[4],      0.0001],
                   [LocalState[0]+0.2,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.01, 0.0001],
                   [LocalState[0]+0.4,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.02, 0.0001],
                   [LocalState[0]+0.6,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.04, 0.0001],
                   [LocalState[0]+0.7,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.07, 0.0001],
                   [LocalState[0]+0.8,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.1 , 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.14, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.18, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.23, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.55, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.66, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.77, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+0.89, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+1.00, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+1.19, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+1.39, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+1.59, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+1.79, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+1.89, 0.0001],
                   [LocalState[0]+0.9,  LocalState[1], LocalState[2], 0.0001, LocalState[4]+1.999, 0.0001]])
    uu = np.array([[0., 0.0],
                   [0., 0.3],
                   [0., 0.5],
                   [0., 0.7],
                   [0., 0.8],
                   [0., 0.9],
                   [0., 0.9],
                   [0., 0.9],
                   [0., 0.8],
                   [0., 0.7],
                   [0., 0.6],
                   [0., 0.5],
                   [0., 0.4],
                   [0., 0.30],
                   [0., 0.22],
                   [0., 0.18],
                   [0., 0.14],
                   [0., 0.1],
                   [0., 0.1],
                   [0., 0.1]])

    return xx, uu



def plotTrajectory(map, ClosedLoop, Complete_Vel_Vect):
    x = ClosedLoop.x
    x_glob = ClosedLoop.x_glob
    u = ClosedLoop.u
    time = ClosedLoop.SimTime
    it = ClosedLoop.iterations
    elapsedTime = ClosedLoop.elapsedTime
    #print elapsedTime

    # plt.figure(3)
    # plt.plot(time[0:it], elapsedTime[0:it, 0])
    # plt.ylabel('Elapsed Time')
    # ax = plt.gca()
    # ax.grid(True)

    plt.figure(2)
    plt.subplot(711)
    plt.plot(time[0:it], x[0:it, 0], color='b', label='Response')
    plt.plot(time[0:it], Complete_Vel_Vect[0:it], color='r', label='Reference')
    plt.ylabel('vx')
    ax = plt.gca()
    ax.legend()
    ax.grid(True)
    plt.subplot(712)
    plt.plot(time[0:it], x[0:it, 1])
    plt.ylabel('vy')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(713)
    plt.plot(time[0:it], x[0:it, 2])
    plt.ylabel('wz')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(714)
    plt.plot(time[0:it], x[0:it, 3],'k')
    plt.ylabel('epsi')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(715)
    plt.plot(time[0:it], x[0:it, 5],'k')
    plt.ylabel('ey')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(716)
    plt.plot(time[0:it], u[0:it, 0], 'r')
    plt.ylabel('steering')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(717)
    plt.plot(time[0:it], u[0:it, 1], 'r')
    plt.ylabel('acc')
    ax = plt.gca()
    ax.grid(True)
    plt.show()



if __name__ == "__main__":

    try:    
        main()
        
    except rospy.ROSInterruptException:
        pass
