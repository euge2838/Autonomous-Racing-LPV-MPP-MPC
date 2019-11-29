#########################################################################################
# Eugenio Alcala Plotting script
#########################################################################################

import sys
sys.path.append(sys.path[0]+'/../ControllersObject')
sys.path.append(sys.path[0]+'/../Utilities')
from dataStructures import LMPCprediction, EstimatorData, ClosedLoopDataObj, LMPCprediction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from LMPC import ControllerLMPC
#from ZeroStepLMPC import ControllerZeroStepLMPC

import sys
import pickle
import pdb
from trackInitialization import Map
from dataStructures import LMPCprediction, EstimatorData, ClosedLoopDataObj
import os


def main():
    homedir = os.path.expanduser("~")
    #file_data = open(homedir+'/barc_data/ClosedLoopDataLPV_MPC.obj', 'rb') 
    #file_data = open(sys.path[0]+'/../data/simulations/ClosedLoopDataLPV_MPC.obj', 'rb') 
    file_data = open(sys.path[0]+'/../data/experiments/ClosedLoopDataLPV_MPC.obj', 'rb') 
    ClosedLoopData = pickle.load(file_data)   
    Controller = pickle.load(file_data)
    file_data.close()

    map = Controller.map

    LapToPlot = range(0,2)

    plotTrajectorySpace(map, ClosedLoopData, LapToPlot)

    #plotTrajectoryTime(map, ClosedLoopData)




def plotTrajectoryTime(map, ClosedLoop):
    x = ClosedLoop.x
    x_glob = ClosedLoop.x_glob
    u = ClosedLoop.u
    time = ClosedLoop.SimTime
    it = ClosedLoop.iterations
    elapsedTime = ClosedLoop.elapsedTime

    plt.figure()
    plt.plot(time[0:it], 1000*elapsedTime[0:it, 0], '-o')
    plt.ylabel('Elapsed Time [ms]'),
    ax = plt.gca()
    ax.grid(True)

    plt.figure()
    plt.subplot(711)
    plt.plot(time[0:it], x[0:it, 0])
    plt.ylabel('vx [m/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(712)
    plt.plot(time[0:it], x[0:it, 1])
    plt.ylabel('vy [m/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(713)
    plt.plot(time[0:it], x[0:it, 2])
    plt.ylabel('wz [rad/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(714)
    plt.plot(time[0:it], x[0:it, 3],'k')
    plt.ylabel('epsi [rad]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(715)
    plt.plot(time[0:it], x[0:it, 5],'k')
    plt.ylabel('ey [m]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(716)
    plt.plot(time[0:it], u[0:it, 0], 'r')
    plt.ylabel('steering [rad]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(717)
    plt.plot(time[0:it], u[0:it, 1], 'r')
    plt.ylabel('acc [m/s^2]')
    ax = plt.gca()
    ax.grid(True)
    plt.show()



def plotTrajectorySpace(map, ClosedLoop, LapToPlot):
    x = ClosedLoop.x
    fin = np.argmin(x[:,4])
    zero_passes = np.where(x[0:fin,4] < 0.1)
    diff_zero_passes = np.diff(zero_passes[0])
    index_zero_passes = np.where(diff_zero_passes > 20)
    indexes = np.zeros(len(index_zero_passes[0]), dtype=int)
    for i in range(0,len(index_zero_passes[0])):
        indexes[i] = int(zero_passes[0][index_zero_passes[0][i]+1])
        
    if len(indexes)>=7:
        ini_for = len(indexes)-7
    else:
        ini_for = 0

    x_glob = ClosedLoop.x_glob
    u = ClosedLoop.u
    time = ClosedLoop.SimTime
    it = ClosedLoop.iterations
    elapsedTime = ClosedLoop.elapsedTime
    plotColors = ['c','g','y','b','m','k','r']
    
    Points = np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4]))
    Points1 = np.zeros((int(Points), 2))
    Points2 = np.zeros((int(Points), 2))
    Points0 = np.zeros((int(Points), 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.figure()
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')

    counter = 0
    for i in range(ini_for,len(indexes)):
        if i==0:
            vel = np.around( np.mean(x[indexes[0:indexes[0]], 0]), decimals=2)
            #plt.plot(x_glob[0:indexes[0], 4], x_glob[0:indexes[0], 5], '-o', color=plotColors[counter], label=vel)
            counter += 1
        else:
            vel = np.around( np.mean(x[indexes[i-1]:indexes[i], 0]), decimals=2)
            plt.plot(x_glob[indexes[i-1]:indexes[i], 4], x_glob[indexes[i-1]:indexes[i], 5], '-o', color=plotColors[counter], label=vel)
            counter += 1
            

    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    #plt.plot(x_glob[0:it, 4], x_glob[0:it, 5], '-r')

    plt.figure()
    plt.subplot(711)
    plt.plot(x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 4], x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 0], '-o')
    plt.ylabel('vx [m/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(712)
    plt.plot(x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 4], x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 1], '-o')
    plt.ylabel('vy [m/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(713)
    plt.plot(x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 4], x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 2], '-o')
    plt.ylabel('wz [rad/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(714)
    plt.plot(x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 4], x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 3], '-o')
    plt.ylabel('epsi [rad]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(715)
    plt.plot(x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 4], x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 5], '-o')
    plt.ylabel('ey [m]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(716)
    plt.plot(x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 4], x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 0], '-o', color='r')
    plt.ylabel('steering [rad]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(717)
    plt.plot(x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 4], x[indexes[len(indexes)-2]:indexes[len(indexes)-1], 1], '-o', color='r')
    plt.ylabel('acc [m/s^2]')
    ax = plt.gca()
    ax.grid(True)
    

    x = ClosedLoop.x
    x_glob = ClosedLoop.x_glob
    u = ClosedLoop.u
    time = ClosedLoop.SimTime
    it = ClosedLoop.iterations
    elapsedTime = ClosedLoop.elapsedTime

    plt.figure()
    plt.plot(time[0:it], 1000*elapsedTime[0:it, 0], '-o')
    plt.ylabel('Elapsed Time [ms]'),
    ax = plt.gca()
    ax.grid(True)

    plt.figure()
    plt.subplot(711)
    plt.plot(time[0:it], x[0:it, 0])
    plt.ylabel('vx [m/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(712)
    plt.plot(time[0:it], x[0:it, 1])
    plt.ylabel('vy [m/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(713)
    plt.plot(time[0:it], x[0:it, 2])
    plt.ylabel('wz [rad/s]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(714)
    plt.plot(time[0:it], x[0:it, 3],'k')
    plt.ylabel('epsi [rad]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(715)
    plt.plot(time[0:it], x[0:it, 5],'k')
    plt.ylabel('ey [m]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(716)
    plt.plot(time[0:it], u[0:it, 0], 'r')
    plt.ylabel('steering [rad]')
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(717)
    plt.plot(time[0:it], u[0:it, 1], 'r')
    plt.ylabel('acc [m/s^2]')
    ax = plt.gca()
    ax.grid(True)
    plt.show()







def plotClosedLoopColorLMPC(LMPController, map, LapToPlot):
    SS_glob = LMPController.SS_glob
    LapCounter  = LMPController.LapCounter
    SS      = LMPController.SS
    uSS     = LMPController.uSS

    TotNumberIt = LMPController.it

    print "Number iterations: ", TotNumberIt
    Points = np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4]))
    Points1 = np.zeros((int(Points), 2))
    Points2 = np.zeros((int(Points), 2))
    Points0 = np.zeros((int(Points), 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.figure()
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')

    xPlot = []
    yPlot = []
    Color = []
    for i in LapToPlot:
        for j in range(0, len(SS_glob[0:LapCounter[i], 4, i].tolist())):
            xPlot.append(SS_glob[0:LapCounter[i], 4, i].tolist()[j])
            yPlot.append(SS_glob[0:LapCounter[i], 5, i].tolist()[j])
            Color.append(np.sqrt( (SS_glob[0:LapCounter[i], 0, i].tolist()[j])**2 +  (SS_glob[0:LapCounter[i], 0, i].tolist()[j]) ) )

    plt.scatter(xPlot, yPlot, alpha=1.0, c = Color, s = 100)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # plt.scatter(SS_glob[0:LapCounter[i], 4, i], SS_glob[0:LapCounter[i], 5, i], alpha=0.8, c = SS_glob[0:LapCounter[i], 0, i])
    plt.colorbar()



main()