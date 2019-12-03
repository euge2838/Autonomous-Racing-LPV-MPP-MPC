#!/usr/bin/env python
"""
    File name: vehicleSimulator.py
    Author: Shuqi Xu
    Email: shuqixu@kth.se
    Updated by Eugenio Alcala
    Email: euge2838@gmail.com
    Python Version: 2.7.12
"""
# ---------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that (1) you retain this notice
# and (2) you provide clear attribution to UC Berkeley, including a link
# to http://barc-project.com
#
# Attibution Information: The barc project ROS code-base was developed
# at UC Berkeley in the Model Predictive Control (MPC) lab by Jon Gonzales
# (jon.gonzales@berkeley.edu). The cloud services integation with ROS was developed
# by Kiet Lam  (kiet.lam@berkeley.edu). The web-server app Dator was
# based on an open source project by Bruce Wootton
#----------------------------------------------------------------------------

import sys
sys.path.append(sys.path[0]+'/ControllersObject')
sys.path.append(sys.path[0]+'/Utilities')
import rospy
import geometry_msgs.msg
from barc.msg import ECU, pos_info, Vel_est, simulatorStates, My_IMU
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from marvelmind_nav.msg import hedge_imu_fusion, hedge_pos
from numpy import tan, arctan, cos, sin, pi
from numpy.random import randn
from tf import transformations
import pdb

def main():
    rospy.init_node("simulator")
    gps_freq_update = rospy.get_param("simulator/gps_freq_update")
    simulator_dt    = rospy.get_param("simulator/dt")
    lowLevelDyn     = rospy.get_param("simulator/lowLevelDyn")
    
    print "[SIMULATOR] Is Low Level Dynamics active?: ", lowLevelDyn
    
    sim = Simulator()
    imu = ImuClass()
    gps = GpsClass(gps_freq_update, simulator_dt)
    enc = EncClass()
    ecu = EcuClass()

    counter = 0

    a_his 	= [0.0]*int(rospy.get_param("simulator/delay_a")/rospy.get_param("simulator/dt"))
    df_his 	= [0.0]*int(rospy.get_param("simulator/delay_df")/rospy.get_param("simulator/dt"))

    
    pub_simulatorStates = rospy.Publisher('simulatorStates', simulatorStates, queue_size=1)
    simStates = simulatorStates()

    print('[SIMULATOR] The simulator is running!')
    print('\n')
    
    servo_inp = 0.0
    T  = simulator_dt
    Tf = 0.07

    while not rospy.is_shutdown():
    	# Simulator delay

		a_his.append(ecu.u[0])
		df_his.append(ecu.u[1])

		if lowLevelDyn == True:
			servo_inp = (1 - T / Tf) * servo_inp + ( T / Tf )*df_his.pop(0)
			u = [a_his.pop(0), servo_inp]
		else:
			u = [a_his.pop(0), df_his.pop(0)] # EA: remove the first element of the array and return it to you.
			#print "Applyied the first of this vector: ",df_his ,"\n "

		sim.f(u)

		simStates.x      = sim.x
		simStates.y      = sim.y
		simStates.vx     = sim.vx
		simStates.vy     = sim.vy
		simStates.psi    = sim.yaw
		simStates.psiDot = sim.psiDot
		
		# Publish input
		pub_simulatorStates.publish(simStates)

		imu.update(sim)
		gps.update(sim)
		enc.update(sim)

		sim.saveHistory()

		gps.gps_pub()
		imu.imu_pub()
		enc.enc_pub()

		sim.rate.sleep()





class Simulator(object):
	""" Object collecting GPS measurement data
    Attributes:
    	Model params:
    		1.L_f 2.L_r 3.m(car mass) 3.I_z(car inertial) 4.c_f(equivalent drag coefficient)
        States:
            1.x 2.y 3.vx 4.vy 5.ax 6.ay 7.psiDot
        States history:
            1.x_his 2.y_his 3.vx_his 4.vy_his 5.ax_his 6.ay_his 7.psiDot_his
        Simulator sampling time:
        	1. dt
        Time stamp:
        	1. time_his
	Methods:
		f(u):
			System model used to update the states
		pacejka(ang):
			Pacejka lateral tire modeling
    """
	def __init__(self):

		self.L_f 	= rospy.get_param("lf")
		self.L_r 	= rospy.get_param("lr")
		self.m 		= rospy.get_param("m")
		self.I_z 	= rospy.get_param("Iz")
		self.c_f 	= rospy.get_param("simulator/c_f")

		self.B = rospy.get_param("simulator/B")
		self.C = rospy.get_param("simulator/C")
		self.mu= rospy.get_param("simulator/mu")
		self.g = 9.81

		self.x 		= 0.01
		self.y 		= 0.0
		self.vx 	= rospy.get_param("simulator/init_vx")
		self.vy 	= 0.0
		self.ax 	= 0.0
		self.ay 	= 0.0

		self.yaw 	= 0.0

		self.psiDot = 0.0

		self.x_his 		= []
		self.y_his 		= []
		self.vx_his 	= []
		self.vy_his 	= []
		self.ax_his 	= []
		self.ay_his 	= []
		self.psiDot_his = []

		self.dt 		= rospy.get_param("simulator/dt")
		self.rate 		= rospy.Rate(1.0/self.dt)
		self.time_his 	= []


	def f(self,u):
		a_F = 0.0
		a_R = 0.0

		if abs(self.vx) > 0.2:
			a_F = u[1] - arctan((self.vy + self.L_f*self.psiDot)/abs(self.vx))
			a_R = arctan((- self.vy + self.L_r*self.psiDot)/abs(self.vx))

		# FyF = self.pacejka(a_F)
		# FyR = self.pacejka(a_R)
		FyF = 60*a_F
		FyR = 60*a_R


		if abs(a_F) > 30.0/180.0*pi or abs(a_R) > 30.0/180.0*pi:
			print "WARNING: Large slip angles in simulation"

		x 	= self.x
		y 	= self.y
		ax 	= self.ax
		ay 	= self.ay
		vx 	= self.vx
		vy 	= self.vy
		yaw = self.yaw
		psiDot = self.psiDot
                
		self.x 		+= self.dt*(cos(yaw)*vx - sin(yaw)*vy)
		self.y 		+= self.dt*(sin(yaw)*vx + cos(yaw)*vy)
		self.vx 	+= self.dt*(ax + psiDot*vy)
		self.vy 	+= self.dt*(ay - psiDot*vx)
		self.ax 	 = u[0] - self.mu*vx - FyF/self.m*sin(u[1])
		self.ay 	 = 1.0/self.m*(FyF*cos(u[1])+FyR)
		self.yaw 	+= self.dt*(psiDot)                                        
		self.psiDot += self.dt*(1.0/self.I_z*(self.L_f*FyF*cos(u[1]) - self.L_r*FyR))

		self.vx = abs(self.vx)
	

	def pacejka(self,ang):
		D = self.c_f*self.m*self.g/2 	# Friction force/2
		Stiffness_Force = D*sin(self.C*arctan(self.B*ang))
		return Stiffness_Force

	def saveHistory(self):
		self.x_his.append(self.x)
		self.y_his.append(self.y)
		self.vx_his.append(self.vx)
		self.vy_his.append(self.vy)
		self.ax_his.append(self.ax)
		self.ay_his.append(self.ay)
		self.psiDot_his.append(self.psiDot)
		self.time_his.append(rospy.get_rostime().to_sec()) 






class ImuClass(object):
	def __init__(self):

		# self.pub  = rospy.Publisher("imu/data", Imu, queue_size=1)
		self.pub  = rospy.Publisher("imu/data", My_IMU, queue_size=1)

		self.ax 	= 0.0
		self.ay 	= 0.0
		self.yaw 	= 0.0
		self.psiDot = 0.0
		self.ax_std 	= rospy.get_param("simulator/ax_std")
		self.ay_std 	= rospy.get_param("simulator/ay_std")
		self.psiDot_std = rospy.get_param("simulator/psiDot_std")
		self.psi_std    = rospy.get_param("simulator/psi_std")
		self.n_bound 	= rospy.get_param("simulator/n_bound")


		# self.msg = Imu()
		self.msg = My_IMU()

	def update(self,sim):
		n = max(-self.ax_std*self.n_bound, min(self.ax_std*randn(), self.ax_std*self.n_bound))
		self.ax = sim.ax + n

		n = max(-self.ay_std*self.n_bound, min(self.ay_std*randn(), self.ay_std*self.n_bound))
		self.ay = sim.ay + n

		n = max(-self.psiDot_std*self.n_bound, min(self.psiDot_std*randn(), self.psiDot_std*self.n_bound))
		self.psiDot = sim.psiDot + n

		n = max(-self.psi_std*self.n_bound, min(self.psi_std*randn(), self.psi_std*self.n_bound))
		self.yaw = sim.yaw + n

	def imu_pub(self):
		# self.msg.linear_acceleration.x = self.ax
		# self.msg.linear_acceleration.y = self.ay
		# self.msg.angular_velocity = Vector3(0,0,self.psiDot)
		# self.orientation = geometry_msgs.msg.Quaternion(0, 0, sin(self.yaw/2), cos(self.yaw/2))
		# self.pub.publish(self.msg)
		self.msg.ax = self.ax
		self.msg.ay = self.ay
		self.msg.psiDot = self.psiDot
		self.msg.yaw = self.yaw
		self.msg.roll = 0
		self.msg.pitch = 0
		self.pub.publish(self.msg)		




class GpsClass(object):
	def __init__(self, gps_freq_update, simulator_dt):
		self.pub  = rospy.Publisher("hedge_pos", hedge_pos, queue_size=1)
		self.x = 0.0
		self.y = 0.0
		self.x_std 	 = rospy.get_param("simulator/x_std")
		self.y_std 	 = rospy.get_param("simulator/y_std")
		self.n_bound = rospy.get_param("simulator/n_bound")

		self.msg = hedge_pos()
		self.counter  = 0
		self.thUpdate = (1.0 /  gps_freq_update) / simulator_dt

	def update(self,sim):
		n = max(-self.x_std*self.n_bound, min(self.x_std*randn(), self.x_std*self.n_bound))
		self.x = sim.x + n

		n = max(-self.y_std*self.n_bound, min(self.y_std*randn(), self.y_std*self.n_bound))
		self.y = sim.y + n

	def gps_pub(self):
		if self.counter > self.thUpdate:
			self.counter = 0
			self.msg.x_m = self.x
			self.msg.y_m = self.y
			self.pub.publish(self.msg)
			# print "Update GPS"
		else:
			# print "Not update GPS"
			self.counter = self.counter + 1





class EncClass(object):
	def __init__(self):
		self.pub  = rospy.Publisher("vel_est", Vel_est, queue_size=1)
		self.v = 0.0
		self.v_std 	 = rospy.get_param("simulator/v_std")
		self.n_bound = rospy.get_param("simulator/n_bound")

		self.msg = Vel_est()

	def update(self,sim):
		n = max(-self.v_std*self.n_bound, min(self.v_std*randn(), self.v_std*self.n_bound))
		self.v = (sim.vx**2+sim.vy**2)**0.5 + n

	def enc_pub(self):
		self.msg.vel_fl  = self.v  
		self.msg.vel_fr  = self.v
		self.msg.vel_bl  = self.v
		self.msg.vel_br  = self.v
		self.msg.vel_est = self.v
		self.pub.publish(self.msg)




class EcuClass(object):
	def __init__(self):
		self.sub = rospy.Subscriber("ecu", ECU, self.ecu_callback, queue_size=1)
		self.u = [0.0, 0.0]

	def ecu_callback(self,data):
		self.u = [data.motor, data.servo]

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
		pass
