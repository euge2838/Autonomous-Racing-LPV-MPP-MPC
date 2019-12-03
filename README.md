# LPV-MPP-MPC
Planning and control for autonomous racing vehicles. This project allows you to solve the autonomous racing driving problem using advanced control theory. 
Particularly, here it is presented a collaborative work using optimal strategies. The Model Predictive Control (MPC) strategy is used online for computing the optimal trajectory maximizing vehicle velocity but also for computing the optimal control actions that make the vehicle to follow the computed references.
All the algorithms are solved in real time employing the Operator Splitting Quadratic Program (OSQP) solver.

### The vehicle model
The planning-control-estimation diagram is shown in the following figure

<!--![](https://github.com/euge2838/LPV-MPP-MPC/blob/master/Berkeley_control_planning_diagram.png) -->
<img src="https://github.com/euge2838/LPV-MPP-MPC/blob/master/Berkeley_control_planning_diagram.png" alt="alt text" width="520" height="314">



### LPV modeling
The LPV paradigm allows to represent a given non-linear representation into a pseudo-linear form as

<img src="https://github.com/euge2838/LPV-MPP-MPC/blob/master/vehicle_modeling.png" alt="alt text" width="521" height="259">
<!-- ![](https://github.com/euge2838/LPV-MPP-MPC/blob/master/vehicle_modeling.png) -->

### The vehicle model
The model used in planning, control and estimation algorithms is the bicycle representation where the inputs are the front steering angle and the rear wheel linear acceleration

![](https://github.com/euge2838/LPV-MPP-MPC/blob/master/variables_representation.png)


### MPC for planning
The trajectory planning for racing is solved using the MPC technique. The cost function adresses the lap time minimization as well as the smothness of the lateral motion by reducing as much as possible the understeer and oversteer behaviours.
This algorithm is launched every 33 ms.

### MPC for control
At this point an MPC is built and solved at every control iteration (33 ms) for figure out the optimal control actions (steering and rear wheel acceleration).


## Running the tests
The codes are in Python 2.7 and the structure is made in ROS. Hence, for running this project you have to do:
```
  - catkin_make (to create build and devel folders)
  - source devel/setup.bash
  - roslaunch barc MAIN_LAUNCH.launch
  
```


## Resulting video
[![IMAGE ALT TEXT HERE](Kazam_screenshot_00000.png)](https://www.youtube.com/watch?v=NrFt6ZmRRY0)


## References
* Trajectory planner presented in:  Alcala, E., Puig, V. & Quevedo, J. (2019). LPV-MP Planning for Autonomous Racing Vehicles considering Obstacles. Robotics and Autonomous Systems.

* Controller presented in: Alcala, E., Puig, V., Quevedo, J., & Rosolia, U. (2019). Autonomous Racing using Linear Parameter Varying - Model Predictive Control (LPV-MPC). Control engineering practice.

* Estimator presented in: Alcala, E., Puig, V., Quevedo, J., & Escobet, T. (2018). Gain-scheduling LPV control for autonomous vehicles including friction force estimation and compensation mechanism. IET Control Theory & Applications, 12(12), 1683-1693. (https://digital-library.theiet.org/content/journals/10.1049/iet-cta.2017.1154).



