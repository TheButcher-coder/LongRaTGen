# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:33:47 2025

@author: c8501100
"""

import numpy as np
from exudyn.utilities import *
from exudyn.rigidBodyUtilities import *
from exudyn.graphicsDataUtilities import *
from exudyn.robotics import *   # to import  robotics core functions
from exudyn.rigidBodyUtilities import HT2rotationMatrix, HT2translation, Skew, HTtranslate, InverseHT,\
                                      HT0, HTrotateY, HTrotateX, RigidBodyInertia
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#**function: generate puma560 manipulator as myRobot dictionary, settings are done in function 
#**output: myRobot dictionary
#**author: Martin Sereinig
#**notes: std DH-parameters: [theta, d, a, alpha], according to P. Corke page 138, 
#       puma p560 limits, taken from Corke Visual Control of Robots 

def ManipulatorTx2_90L():
    # 
    thetaList = [0, 0,0,0,0,0] # q(2)-pi/2 q(3)+pi/2 q(4) q(5) q(6)]' # offsets?
    # thetaList = [0, -np.pi/2, -np.pi/2, 0,np.pi,0] * 0
    # offset= [0]*6
    offset = [0, -np.pi/2, -np.pi/2, 0,np.pi,0] 
    dList = [0.478, 0.05, 0, 0.550, 0, 0.1]
    aList= [0.05, 0.50, 0, 0, 0, 0,]
    alphaList= [-np.pi/2, 0, -np.pi/2, -np.pi/2, -np.pi/2, 0]
    
    rotLimit = np.array([[-180,180], 
                [-130, 145.5], 
                [-145, 145], 
                [-270, 270], 
                [-115, 140], 
                [-270, 270], ])*np.pi/180 # convert to rad
    velLimit = np.array([400,350,410,540,475,760])*np.pi/180 # rad/s
    
    massList=[36.67, 41.45, 15.97, 22.43, 0.443, 0.032]
    r_com=[[-0.13, 0.47,   -0.18], # %(rx, ry, rz) link 1
           [ 0,       0,         0],  #%(rx, ry, rz) link 2
           [ 0.05,	-0.55,	 -0.14],  # %(rx, ry, rz) link 3
           [ 0,       0.004,       0], #%(rx, ry, rz) link 4
           [ 0.01,       0,           0], #%(rx, ry, rz) link 5
           [ 0,       0,         0]] #%(rx, ry, rz) link 6
    # %Inertia matrices of each link with respect to its D-H reference system.
    # % Ixx	Iyy	Izz	Ixy	Iyz	Ixz, for each row
    InertiaList =[[1.158, -0.031, -0.224, 1.42, -0.038, 1.469],
                  [7.277, 0, 0, 7.641, 0.011, 0.671],
                  [0.344, 0.007, -0.007, 0.292, -0.001, 0.329],
                  [1.909, 0, 0.059, 1.893, 0, 0.237],
                  [0.001, 0, 0, 0.001, 0, 0.001],
                  [0, 0, 0, 0, 0, 0]]
    
    # stiffness and torque is just a "guess"
    stiffnessList = [1e6, 1e6, 1e5, 1e4, 1e3, 1e3]
    torqueList = [100, 100, 50, 40, 20, 10] 
    
    # todo: calculate inertia in COM !
    print('WARNING: inertia of robot is given relative to D-H reference system, not COM, leading to wrong dynamics!')
    linkList = []
    for i in range(len(dList)): 
        linkList +=[
               {'stdDH':[thetaList[i],dList[i],aList[i],alphaList[i]], 
               # 'modDHKK':[0,0,0,0],# not set
               'mass':massList[i],  #not needed!
               'inertia':np.diag(InertiaList[i][0:3]), # should be w.r.t. COM - currently in wrong system!
               'jointStiffness':stiffnessList[i], # Values from literature described in KIM1995 Puma Joint Stiffness
               'jointTorqueMax': torqueList[i],  # maximum joint torques Puma560, taken from taken from Corke Visual Control of Robots, p58 table2.21
               'jointLimits': rotLimit[i,:], # taken from Corke Visual Control of Robots 
               'COM':r_com[i]} # w.r.t. stdDH joint coordinatesystem
                ]
    
    Tmax=[]
    JointStiffness=[]
    for link in linkList:
         JointStiffness += [link['jointStiffness']]
         Tmax += [link['jointTorqueMax']]             


    #this is the global myRobot structure
    myRobot={'links':linkList,
           'jointType':[1,1,1,1,1,1], #1=revolute, 0=prismatic
           'jointStiffnessMatrix':   np.diag(JointStiffness),
           'joinTorqueMaxMatrix':    np.diag(Tmax),
           'base':{'HT':HTtranslate([0,0,0])},
           'tool':{'HT':HTtranslate([0,0,0])},
           'gravity':[0,0,-9.81],
           'referenceConfiguration':[0]*6, #reference configuration for bodies; at which the myRobot is built
           'dhMode': 'stdDH', #this mode prescribes the default DH mode to be used; 
           'Pcontrol': np.array([200000, 100000, 40000, 1000, 300, 50]), #some assumed values, not taken from real robot
           'Dcontrol': np.array([4000,   4000,   1000,   10,   5,   1]),#some assumed values, not taken from real robot
           'offset': offset,
           } 
    return myRobot



def createRobot(mbs, baseMarker, q0 = [0]*6, TBase = HT(np.eye(3), [0]*3)): 
    import roboticstoolbox as rtb
    
    robotDef = ManipulatorTx2_90L()
    graphicsBaseList = [graphics.FromSTLfile('TX2_90L/link0.stl', color=graphics.color.yellow), 
                        graphics.Basis(length=0.7)]
    robot = Robot(gravity=[0,0,-9.81],
                  base = RobotBase(HT=TBase, visualization=VRobotBase(graphicsData=graphicsBaseList)),
                  tool = RobotTool(HT=HTtranslate([0]*3), visualization=VRobotTool(graphicsData=[])),
                  referenceConfiguration = q0) #referenceConfiguration created with 0s automatically
    nLinks = len(robotDef['links'])
    
    links_rtb = []
    for cnt, link in enumerate(robotDef['links']):
        # print('create link: ', cnt, link)
        RotLink = np.eye(3)
        if cnt == 2: 
            RotLink = RotXYZ2RotationMatrix([0,0,np.pi])
        vLink = [graphics.FromSTLfile('TX2_90L/link{}.stl'.format(cnt+1),color=graphics.color.yellow, 
                                    Aoff = RotLink)]
                                     # ,scale=0.25,pOff=[0.35,0,0], Aoff=turtleRot) 
        if cnt == 5: 
            vLink += [graphics.Basis(length=0.2)]
        robot.AddLink(RobotLink(mass=link['mass'], 
                                   COM=link['COM'], 
                                   inertia=link['inertia'], 
                                    localHT=StdDH2HT(link['stdDH']),
                                    # localHT=StdDH2HT(link['modKKDH']),
                                   PDcontrol=(robotDef['Pcontrol'][cnt], robotDef['Dcontrol'][cnt]),
                                   visualization=VRobotLink(graphicsData=vLink)
                                   ))
        links_rtb += [rtb.RevoluteDH(d=link['stdDH'][1], a=link['stdDH'][2], alpha=link['stdDH'][3],
                                     offset=robotDef['referenceConfiguration'][cnt]*0, qlim=link['jointLimits'])]
        
    robotDict = robot.CreateRedundantCoordinateMBS(mbs, baseMarker=baseMarker, 
                                                      createJointTorqueLoads=False,
                                                      )
    oMotorList = robotDict['springDamperList']
    for i in range(6): 
        mbs.SetObjectParameter(oMotorList[i], 'offset', q0[i])
        # mbs.SetObjectParameter(oMotorList[i], 'activeConnector', False)
        
    
    robot_rtb = rtb.DHRobot(links_rtb)
    return robotDict, robot, robot_rtb
        
if __name__ == '__main__': 
    
    

    from numpy import linalg as LA
    from math import pi
    
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()
    sensorWriteToFile = False
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #configurations and trajectory
    q0 = [0,0,0,0,0,0] #zero angle configuration
    

    #this set also works with load control:
    q1 = [0, pi/8, pi*0.5, 0,pi/8,0] #configuration 1
    q2 = [0.8*pi,-0.8*pi, -pi*0.5,0.75*pi,-pi*0.4,pi*0.4] #configuration 2
    q3 = [0.5*pi,0,-0.25*pi,0,0,0] #zero angle configuration
    
    #trajectory generated with optimal acceleration profiles:
    trajectory = Trajectory(initialCoordinates=q1, initialTime=0)
    # trajectory.Add(ProfileConstantAcceleration(q3,0.25))
    trajectory.Add(ProfileConstantAcceleration(q1,0.25))
    # trajectory.Add(ProfileConstantAcceleration(q2,0.25))
    # trajectory.Add(ProfileConstantAcceleration(q0,0.25))
   
     
        
    
    #++++++++++++++++++++++++++++++++++++++++++++++++
    #base, graphics, object and marker:
    objectGround = mbs.AddObject(ObjectGround(# referencePosition=HT2translation([0,0,0]), 
                                          #visualization=VObjectGround(graphicsData=graphicsBaseList)
                                              ))
    
    

        
    #baseMarker; could also be a moving base!
    baseMarker = mbs.AddMarker(MarkerBodyRigid(bodyNumber=objectGround, localPosition=[0,0,0]))
    
    robotDict, robot, robot_rtb = createRobot(mbs, baseMarker, q0=q1)
    # robot.GetBaseHT()
    
    T = HT(np.eye(3), [0,0,1.0])
    robot_rtb.ikine_LM(T)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #build mbs robot model:
    
    jointList = robotDict['jointList'] #must be stored there for the load user function

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #user function which is called only once per step, speeds up simulation drastically
    def PreStepUF(mbs, t):
        staticTorques = np.zeros(len(jointList))
            
        [u,v,a] = trajectory.Evaluate(t)
    
        #compute load for joint number
        for i in range(len(robot.links)):
            joint = jointList[i]
            phi = mbs.GetObjectOutput(joint, exu.OutputVariableType.Rotation)[2] #z-rotation
            omega = mbs.GetObjectOutput(joint, exu.OutputVariableType.AngularVelocityLocal)[2] #z-angular velocity
            tsd = torsionalSDlist[i]
            mbs.SetObjectParameter(tsd, 'offset', u[i] +robot.referenceConfiguration[i] )
            mbs.SetObjectParameter(tsd, 'velocityOffset', v[i])
            mbs.SetObjectParameter(tsd, 'torque', staticTorques[i]) #additional torque from given velocity 

        
        return True
    
    mbs.SetPreStepUserFunction(PreStepUF)
    
    sListJointAngles = []
    sListTorques = []
    nJoints = len(jointList)

    torsionalSDlist = robotDict['springDamperList']
    sJointRotComponents = [0]*nJoints #only one component
    sTorqueComponents = [0]*nJoints   #only one component

    
    mbs.Assemble()
    #mbs.systemData.Info()
    
    SC.visualizationSettings.connectors.showJointAxes = True
    SC.visualizationSettings.connectors.jointAxesLength = 0.02
    SC.visualizationSettings.connectors.jointAxesRadius = 0.002
    
    SC.visualizationSettings.nodes.showBasis = True
    SC.visualizationSettings.nodes.basisSize = 0.1
    SC.visualizationSettings.loads.show = False
    
    SC.visualizationSettings.openGL.multiSampling=4
        
    # tEnd = 1.25
    # h = 0.002
    tEnd = 1.25
    h = 0.001#*0.1*0.01
    
    #mbs.WaitForUserToContinue()
    simulationSettings = exu.SimulationSettings() #takes currently set values or default values
    
    simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
    simulationSettings.timeIntegration.endTime = tEnd
    simulationSettings.solutionSettings.solutionWritePeriod = 0.005
    simulationSettings.solutionSettings.sensorsWritePeriod = 0.005
    simulationSettings.solutionSettings.binarySolutionFile = True
    #simulationSettings.solutionSettings.writeSolutionToFile = False
    # simulationSettings.timeIntegration.simulateInRealtime = True
    # simulationSettings.timeIntegration.realtimeFactor = 0.25
    
    simulationSettings.timeIntegration.verboseMode = 1
    # simulationSettings.displayComputationTime = True
    simulationSettings.displayStatistics = True
    simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
    
    #simulationSettings.timeIntegration.newton.useModifiedNewton = True
    simulationSettings.timeIntegration.newton.useModifiedNewton = True
    
    simulationSettings.timeIntegration.generalizedAlpha.computeInitialAccelerations=True
    SC.visualizationSettings.general.autoFitScene=False
    SC.visualizationSettings.window.renderWindowSize=[1920,1200]
    useGraphics = True
    
    # q_new = [np.pi/4,-np.pi, np.pi, 0, -np.pi, 0]
    # exu.StartRenderer()
    # for i, tsd in enumerate(torsionalSDlist): 
    #     mbs.SetObjectParameter(tsd, 'offset', q_new[i])
    #     print('joint ', i, ', angle: ', q_new[i])
    # mbs.SolveStatic()
    
    if useGraphics:
        exu.StartRenderer()
        
        robot_rtb.plot(q0)
        # sys.exit()
        
        
        
        if 'renderState' in exu.sys:
            SC.SetRenderState(exu.sys['renderState'])
        # mbs.WaitForUserToContinue()
        
    mbs.SolveDynamic(simulationSettings, 
                     # solverType=exu.DynamicSolverType.TrapezoidalIndex2, 
                     showHints=True)
    #explicit integration also possible for KinematicTree:
    # mbs.SolveDynamic(simulationSettings, 
    #                  solverType=exu.DynamicSolverType.RK33,
    #                  showHints=True)
    
    
    if useGraphics:
        SC.visualizationSettings.general.autoFitScene = False
        # exu.StopRenderer()
        