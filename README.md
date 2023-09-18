# MobileDeformBench

## Name
Mobile elastic(deformable) manipulation benchmark

## Description
In this benchmark, there are five tasks involving elastic objects and mobile manipulators.

## Visuals

PlaceTask  
	<img src="Pics/Place.gif" width="250"/>  

BendTask  
    <img src="Pics/Bend.gif" width="250"/>  

TransportTask  
    <img src="Pics/Transport.gif" width="250"/>  

DragTask  
    <img src="Pics/Drag.gif" width="250"/>  

LiftTask  
    <img src="Pics/Lift.gif" width="250"/>

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

## Roadmap

