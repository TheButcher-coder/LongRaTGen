#
 
"""

* title:        Draft: Satellite deflections
* autor:        Peter Manzl, UIBK
* description:  

todo: 
    * Done! add FFRF body from satelite
        * optional: split into several bodies and connect them
    * Done! add cage of robot 
    * Done! Move satelite from [p0, R0] to  [p1, R1] using a constraint
    * add Stäubli Robot for visualization
    * create collision free trajectories
    
"""

#%% import libraries

import exudyn as exu
from exudyn.FEM import *

from exudyn.utilities import * #includes itemInterface and rigidBodyUtilities
import exudyn.graphics as graphics 
from exudyn.lieGroupBasics import LogSE3, ExpSE3, LogSO3, ExpSO3

import numpy as np
import time
import sys

from model_staeubli import createRobot
from exudyn.robotics import InverseKinematicsNumerical
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP
from spatialmath import SO3, SE3


#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
#Use FEMinterface to import FEM model and create FFRFreducedOrder object

SC = exu.SystemContainer()
mbs = SC.AddSystem()
fileName = 'satelite_V1.npy' # file into which the modal reduction is saved
createFile = True
flagRobot = True

graphicsGround = graphics.CheckerBoard(point=[0,0,1e-3], size=6)
graphicsCage = graphics.FromSTLfile('kaefig_tx2.stl', color=[0.2, 0.2, 0.2, 0.2])
oGround = mbs.AddObject(ObjectGround(referencePosition= [0,0,0], 
                                     visualization=VObjectGround(graphicsData=[graphicsGround])
                                     ))


mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0,0,0]))

if flagRobot: 
    # q0 = [np.pi/2, -np.pi/2, 0, 0, np.pi/2, 0]
    mGroundRobot = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0,0,0.35*0]))
    TBase = HT(np.eye(3), [0]*3) #HT(RotXYZ2RotationMatrix([0,0,-np.pi/2]), [0,0,0.35])
    robotDict, robot,robot_rtb = createRobot(mbs, mGroundRobot, TBase = TBase)
    # oMotorList = robotDict['springDamperList']
    # for i in range(6): 
        # mbs.SetObjectParameter(oMotorList[i], 'offset', q0[i])
    # mbs.Assemble()
    # myIkine = InverseKinematicsNumerical(robot, useRenderer=False)
    mFlangeRobot = mbs.AddMarker(MarkerBodyRigid(bodyNumber=robotDict['bodyList'][-1], localPosition=[0,0,0]))


p_0 = [0.0, 0.0, 1]
phi_0 = [0,0,0]

p_1 = [0.05, -0.3, 1+0.15]
phi_1 = [0.,0,0]

p_2 = [0.05, -0.3, 1.55]
phi_2 = [0,0,0]


if flagRobot: 
    q_0 = [0]*6
    T_0 = SE3(TBase) * robot_rtb.fkine([0]*6)
    p_0 = T_0.t
    phi_0 = [0,0,0] # RotationMatrix2RotXYZ(T_0.R) @ np.linalg.inv(TBase[0:3,0:3])
    T_1 = HT(RotXYZ2RotationMatrix(phi_1), p_1)
    q_1 = robot_rtb.ikine_LM(T_1, q_0).q
    
    T_2 = HT(RotXYZ2RotationMatrix(phi_2), p_2)
    q_2 = robot_rtb.ikine_LM(T_2, q_1).q
    
    trajectory = Trajectory(initialCoordinates=q_0, initialTime=0)
    #trajectory.Add(ProfileConstantAcceleration(q_1,1))
    #for i in range(4):
    #    trajectory.Add(ProfileConstantAcceleration(q_2,0.2))
    #    trajectory.Add(ProfileConstantAcceleration(q_1,0.2))
    #Test für noise ring:
    import sys
    sys.path.insert(0, '..\\..\\Code')
    import RaTGen as rt
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Initialisierung
    rat = rt.RaTGen()
    rat.dt = 1
    rat.mean = .1
    rat.std = .01
    # Punkte generieren
    x = rat.generate_sin(.5, 1, 0, 0, 4 * np.pi)
    y = rat.generate_cos(.5, 1, 0, 0, 4 * np.pi)
    #z = rat.generate_noise(0, 2 * np.pi)
    z = np.ones([len(x)])

    # Rotation (angenommen: rot ist eine Liste von 3x3-Rotationsmatrizen)
    rot = rat.generate_rot_X(0, 4*np.pi)

    # Punktwolke p
    p = np.array([x, y, z])

    #get homogenous Transform
    T = []
    for i in range(len(rot)):
        T.append(HT(rot[i], np.array([x[i], y[i], z[i]])))
    #T = HT(rot, p)
    #calculate inverse kin
    Q = []
    t_last = T[0]
    q0 = np.zeros(robot_rtb.n)  # or any reasonable initial guess
    for t in T:
        ik_solution = robot_rtb.ikine_LM(t, q0)
        Q.append(ik_solution.q)
        q0 = ik_solution.q  # use previous solution as next initial guess

    for q in Q:
        trajectory.Add(ProfileConstantAcceleration(q, 1))

#%% 

nModes = 48

pFlange_Abaqus = [0, -2e-3,0]
normalFlange_Abaqus = [0,1,0]

fem = FEMinterface()

if createFile: 
    nodes=fem.ImportFromAbaqusInputFile('sat_draft.inp', typeName='Instance', name='satelite')
    
    fem.ReadMassMatrixFromAbaqus('sat_draft_MASS.mtx')
    fem.ReadStiffnessMatrixFromAbaqus('sat_draft_STIF.mtx')
    
    if True:
        exu.Print('size of nodes:', sys.getsizeof(np.array(fem.nodes['Position'])) )
        exu.Print('size of elements:', sys.getsizeof(fem.elements[0]) )
        exu.Print('size of massMatrix:', sys.getsizeof(fem.massMatrix) )
        exu.Print('size of stiffnessMatrix:', sys.getsizeof(fem.stiffnessMatrix) )
        exu.Print('size of modeBasis:', sys.getsizeof(fem.modeBasis) )
        #print('size of postProcessingModes:', sys.getsizeof(fem.postProcessingModes['matrix']) )
        exu.Print('===================')
    
    # read from exported abaqus model: flange is in negative y direction. In course model 
    # the number of nodes can also be checked for a "sanity check"
    nodesFlange  = fem.GetNodesInPlane(pFlange_Abaqus, normalFlange_Abaqus) 
    
    exu.Print("nNodes=",fem.NumberOfNodes())
    exu.Print("compute HCB modes... ")
    start_time = time.time()
    fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=[nodesFlange], 
                                  nEigenModes=nModes, 
                                  useSparseSolver=False, #sparse solver gives non-repeatable results ...
                                  computationMode = HCBstaticModeSelection.RBE2)
    
    exu.Print("HCB modes needed %.3f seconds" % (time.time() - start_time))
    
    cms = ObjectFFRFreducedOrderInterface(fem)
    fem.SaveToFile(fileName)
else: 
    fem.LoadFromFile(fileName)
    cms = ObjectFFRFreducedOrderInterface(fem)

R_0 = RotXYZ2RotationMatrix(phi_0)
R_Obj2 = RotXYZ2RotationMatrix(phi_1)
R_Model = RotXYZ2RotationMatrix([np.pi/2, 0, 0])

R_Obj = R_Model @ R_0
objFFRF = cms.AddObjectFFRFreducedOrder(mbs, positionRef=p_0, 
                                        rotationMatrixRef = R_Obj,
                                        initialVelocity=[0,0,0], 
                                        initialAngularVelocity=[0,0,0],
                                        gravity=[0,0,-9.81],
                                        stiffnessProportionalDamping=0.01,
                                        color=[0.1,0.9,0.1,1.])


p_Flange = p_0 + np.reshape(R_Obj @ np.array([[0,0,-2e-3]]).T, -1)
print('pos flange: ', p_Flange)

nodesFlange  = fem.GetNodesInPlane(pFlange_Abaqus, normalFlange_Abaqus) 
centerFlange = fem.GetNodePositionsMean(nodesFlange)
# centerPointAverage1 = fem.GetNodePositionsMean(nodesPlane1)

lenNodesFlange = len(nodesFlange)
weightsFlange = np.array((1./lenNodesFlange)*np.ones(lenNodesFlange))

mFlange = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'], 
                                      meshNodeNumbers=np.array(nodesFlange), #these are the meshNodeNumbers
                                      weightingFactors=weightsFlange, 
                                      offset=centerFlange))



mRB = mbs.AddMarker(MarkerNodeRigid(nodeNumber=objFFRF['nRigidBody']))

# oFlangeConstraint = ObjectGround(referencePosition= p_0))

if flagRobot: 
       mbs.AddObject(GenericJoint(markerNumbers=[mFlangeRobot, mFlange],  
                             rotationMarker0=R_Obj,
                             ))
       # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mFlangeRobot, mRB], 
                                           # stiffness=np.diag([1e3, 1e3, 1e5, 1e1, 1e1, 1e3]), 
                                           # damping=np.diag([1e2, 1e2, 1e2, 0.1e1, 0.1e1, 1e1]), 
                                           # rotationMarker0=R_Obj,
                                           # ))
       
else: 
    # use a moving constraint instead of robot to move the satelite
    mFlangeConstraint = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=p_0))
    mbs.variables['mFlangeConstraint'] = mFlangeConstraint
    mbs.variables['cFlange'] = mbs.AddObject(GenericJoint(markerNumbers=[mFlangeConstraint, mFlange],  
                           rotationMarker0=R_Obj,
                           ))
                           

# mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mFlangeConstraint, mRB], 
#                                     stiffness=np.diag([1e3, 1e3, 1e5, 1e1, 1e1, 1e3]), 
#                                     damping=np.diag([1e2, 1e2, 1e2, 0.1e1, 0.1e1, 1e1]), 
#                                     rotationMarker0=R_Obj,
#                                     ))
sCorners = []
for __x in [0.1, -0.1]: 
    for __z in [0.715, -0.715]:
        nCorner = fem.GetNodeAtPoint([__z,0, __x])
        
        sCorner = mbs.AddSensor(SensorSuperElement(bodyNumber=objFFRF['oFFRFreducedOrder'], 
                                  meshNodeNumber=nCorner, #meshnode number!
                                  outputVariableType = exu.OutputVariableType.Position, 
                                  storeInternal=True ))
        
        sCorners += [sCorner]
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
sFlange=mbs.AddSensor(SensorMarker(markerNumber=mFlange, #meshnode number!
                         storeInternal=True,
                         outputVariableType = exu.OutputVariableType.Position))



oGround2 = mbs.AddObject(ObjectGround(referencePosition= [0,0,0], 
                                     visualization=VObjectGround(graphicsData=[graphicsCage])
                                     ))


mbs.Assemble()

mbs.variables['p0'] = np.array(p_0)
mbs.variables['p1'] = np.array(p_1)
mbs.variables['rot0'] = Skew2Vec(LogSO3(R_Obj))
mbs.variables['rot1'] = Skew2Vec(LogSO3(R_Obj2))

if flagRobot: 
    def PreStepRobotMotion(mbs, t): 
        u,v,a = trajectory.Evaluate(t)
        for i in range(6): 
            mbs.SetObjectParameter(robotDict['springDamperList'][i], 'offset', u[i])
        return True
    PreStepFunction = PreStepRobotMotion
    

else: 
    def PreStepConstraintMotion(mbs, t): 
        p = SmoothStep(t, 0, 0.2, mbs.variables['p0'], mbs.variables['p1'])#
        mActor = mbs.variables['mFlangeConstraint']
        mbs.SetMarkerParameter(mActor, 'localPosition', p)
    
        rot = SmoothStep(t, 0, 0.2, mbs.variables['rot0'], mbs.variables['rot1'])
        mbs.SetObjectParameter(mbs.variables['cFlange'], 'rotationMarker0', ExpSO3(rot))
        
        return True
    PreStepFunction = PreStepConstraintMotion


mbs.SetPreStepUserFunction(PreStepFunction)

simulationSettings = exu.SimulationSettings()

# SC.visualizationSettings.nodes.defaultSize = nodeDrawSize
SC.visualizationSettings.nodes.drawNodesAsPoint = False
SC.visualizationSettings.bodies.deformationScaleFactor = 1e4 #use this factor to scale the deformation of modes
SC.visualizationSettings.openGL.initialCenterPoint = p_0
SC.visualizationSettings.loads.drawSimplified = False


SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
SC.visualizationSettings.contour.outputVariableComponent = -1 #y-component
SC.visualizationSettings.bodies.deformationScaleFactor = 1
# simulationSettings.solutionSettings.solutionInformation = "ObjectFFRFreducedOrder test"

h=5e-4
tEnd = 1 #at almost max. of deflection under gravity
if flagRobot: 
    tEnd = trajectory[-1]['finalTime'] + 0.5

simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.solutionSettings.solutionWritePeriod = h
simulationSettings.timeIntegration.verboseMode = 1
#simulationSettings.timeIntegration.verboseModeFile = 3
simulationSettings.timeIntegration.newton.useModifiedNewton = True

simulationSettings.solutionSettings.sensorsWritePeriod = h
# simulationSettings.solutionSettings.coordinatesSolutionFileName = "solution/satelite.txt"
simulationSettings.solutionSettings.writeSolutionToFile=True

simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.5 #SHOULD work with 0.9 as well

exu.StartRenderer()
if 'renderState' in exu.sys:
    SC.SetRenderState(exu.sys['renderState']) #load last model view
    exu.sys['renderState']['centerPoint'] = p_0
mbs.WaitForUserToContinue() #press space to continue

mbs.SolveDynamic(simulationSettings)


SC.WaitForRenderEngineStopFlag()
exu.StopRenderer() #safely close rendering window!
lastRenderState = SC.GetRenderState() #store model view for next simulation

import matplotlib.pyplot as plt

# create 3D plot with trajectories of cornerpoints using sensors
ax = plt.figure().add_subplot(projection='3d')
dataFlange = mbs.GetSensorStoredData(sFlange)
ax.plot(dataFlange[:,1], dataFlange[:,2],dataFlange[:,3])
for i in range(len(sCorners)): 
    dataCorner = mbs.GetSensorStoredData(sCorners[i])
    ax.plot(dataCorner[:,1], dataCorner[:,2], dataCorner[:,3])


