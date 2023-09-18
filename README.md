# MobileElastic TaskSuite

## Name
Mobile elastic manipulation task suite

In this benchmark, there are five tasks involving elastic objects and mobile manipulators.

## Installation

1. Install OmniIsaacGymEnvs for Isaac Gym(https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)

2. Run the existing example to test  OmniIsaacGymEnvs

3. Clone MoelaSuite 
```
git clone https://github.com/carolzyy/MoelaTasksuite.git
cd MoelaTasksuite
```
4. Training task
```
/path/to/isaacsim/isaac_sim-2022.2.0/python.sh /path/to/MoelaTasksuite/scripts/rlgames_train.py  task=Taskname
```

## Description

PlaceTask  

&emsp;&emsp;Describtion:  
&emsp;&emsp;&emsp;&emsp;Put the long elastic rod on the table  

&emsp;&emsp;Metrics:  
&emsp;&emsp;&emsp;&emsp;The distance between the table surface middle point and the rod endpoint  

&emsp;&emsp;Example:  

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="Pics/Place.gif" width="350"/>  

BendTask  
&emsp;&emsp;Description:  
&emsp;&emsp;&emsp;&emsp;end the rod with the help of the wall and go through the corner  

&emsp;&emsp;Metrics:  
&emsp;&emsp;&emsp;&emsp;The distance between the red target cube and the end point of the rod  

&emsp;&emsp;Example:  

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="Pics/Bend.gif" width="350"/>  

TransportTask  
&emsp;&emsp;Description:  
&emsp;&emsp;&emsp;&emsp;Transport the rod to the target point by passing the obstacle in the middle of the corridor  
 
&emsp;&emsp;Metrics:   
&emsp;&emsp;&emsp;&emsp;The distance between the target point and the end point of the rod  

&emsp;&emsp;Example:  

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="Pics/Transport.gif" width="350"/>  

DragTask  
Description:  
&emsp;&emsp;&emsp;&emsp;Drag the rod to move it to the other side of the obstacle  

Metrics:  
&emsp;&emsp;&emsp;&emsp;The distance between the middle point of the rod and the green target point  

&emsp;&emsp;Example:  

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="Pics/Drag.gif" width="350"/>  

LiftTask  

&emsp;&emsp;Description:  
&emsp;&emsp;&emsp;&emsp;The belt hangs on as a curtain, the robot needs to lift the belt and then arrive at the target point

&emsp;&emsp;Metrics:  
&emsp;&emsp;&emsp;&emsp;The distance between the middle point of the belt and the target point, the distance from the robot to the final red target point  

&emsp;&emsp;Example:  

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="Pics/Lift.gif" width="350"/>


## Roadmap
1. add image observation
2. add new task which include new shape of elastic objects
