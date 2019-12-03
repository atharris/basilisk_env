''' '''
'''
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

'''

import math
import numpy as np

from Basilisk.fswAlgorithms import (hillPoint, inertial3D, attTrackingError, MRP_Feedback,
                                    rwMotorTorque, fswMessages, opNavPoint, velocityPoint,
                                    sunSafePoint, cssWlsEst, headingSuKF, limbFinding, horizonOpNav,
                                    centerRadiusCNN, faultDetection)
from Basilisk.fswAlgorithms import pixelLineConverter, houghCircles, relativeODuKF, pixelLineBiasUKF #FSW for OpNav
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import fswSetupRW, unitTestSupport, orbitalMotion, macros
from Basilisk import __path__
bskPath = __path__[0]

class BSKFswModels():
    def __init__(self, SimBase, fswRate):
        # Define process name and default time-step for all FSW tasks defined later on
        self.processName = SimBase.FSWProcessName
        self.processTasksTimeStep = macros.sec2nano(fswRate)  # 0.5
        
        # Create module data and module wraps
        self.inertial3DData = inertial3D.inertial3DConfig()
        self.inertial3DWrap = SimBase.setModelDataWrap(self.inertial3DData)
        self.inertial3DWrap.ModelTag = "inertial3D"
        
        self.hillPointData = hillPoint.hillPointConfig()
        self.hillPointWrap = SimBase.setModelDataWrap(self.hillPointData)
        self.hillPointWrap.ModelTag = "hillPoint"
        
        self.sunSafePointData = sunSafePoint.sunSafePointConfig()
        self.sunSafePointWrap = SimBase.setModelDataWrap(self.sunSafePointData)
        self.sunSafePointWrap.ModelTag = "sunSafePoint"

        self.opNavPointData = opNavPoint.OpNavPointConfig()
        self.opNavPointWrap = SimBase.setModelDataWrap(self.opNavPointData)
        self.opNavPointWrap.ModelTag = "opNavPoint"
        
        self.velocityPointData = velocityPoint.velocityPointConfig()
        self.velocityPointWrap = SimBase.setModelDataWrap(self.velocityPointData)
        self.velocityPointWrap.ModelTag  = "velocityPoint"
        
        self.cssWlsEstData = cssWlsEst.CSSWLSConfig()
        self.cssWlsEstWrap = SimBase.setModelDataWrap(self.cssWlsEstData)
        self.cssWlsEstWrap.ModelTag = "cssWlsEst"
        
        self.trackingErrorData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorWrap = SimBase.setModelDataWrap(self.trackingErrorData)
        self.trackingErrorWrap.ModelTag = "trackingError"

        self.trackingErrorCamData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorCamWrap = SimBase.setModelDataWrap(self.trackingErrorCamData)
        self.trackingErrorCamWrap.ModelTag = "trackingErrorCam"

        self.mrpFeedbackControlData = MRP_Feedback.MRP_FeedbackConfig()
        self.mrpFeedbackControlWrap = SimBase.setModelDataWrap(self.mrpFeedbackControlData)
        self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"

        self.mrpFeedbackRWsData = MRP_Feedback.MRP_FeedbackConfig()
        self.mrpFeedbackRWsWrap = SimBase.setModelDataWrap(self.mrpFeedbackRWsData)
        self.mrpFeedbackRWsWrap.ModelTag = "mrpFeedbackRWs"
        
        self.rwMotorTorqueData = rwMotorTorque.rwMotorTorqueConfig()
        self.rwMotorTorqueWrap = SimBase.setModelDataWrap(self.rwMotorTorqueData)
        self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"

        self.imageProcessing = houghCircles.HoughCircles()
        self.imageProcessing.ModelTag = "houghCircles"

        self.opNavCNN = centerRadiusCNN.CenterRadiusCNN()
        self.opNavCNN.ModelTag = "opNavCNN"

        self.pixelLineData = pixelLineConverter.PixelLineConvertData()
        self.pixelLineWrap = SimBase.setModelDataWrap(self.pixelLineData)
        self.pixelLineWrap.ModelTag = "pixelLine"

        self.opNavFaultData = faultDetection.FaultDetectionData()
        self.opNavFaultWrap = SimBase.setModelDataWrap(self.opNavFaultData)
        self.opNavFaultWrap.ModelTag = "OpNav_Fault"

        self.limbFinding = limbFinding.LimbFinding()
        self.limbFinding.ModelTag = "limbFind"

        self.horizonNavData = horizonOpNav.HorizonOpNavData()
        self.horizonNavWrap = SimBase.setModelDataWrap(self.horizonNavData)
        self.horizonNavWrap.ModelTag = "limbNav"

        self.relativeODData = relativeODuKF.RelODuKFConfig()
        self.relativeODWrap = SimBase.setModelDataWrap(self.relativeODData)
        self.relativeODWrap.ModelTag = "relativeOD"

        self.pixelLineFilterData = pixelLineBiasUKF.PixelLineBiasUKFConfig()
        self.pixelLineFilterWrap = SimBase.setModelDataWrap(self.pixelLineFilterData)
        self.pixelLineFilterWrap.ModelTag = "pixelLineFilter"

        self.headingUKFData = headingSuKF.HeadingSuKFConfig()
        self.headingUKFWrap = SimBase.setModelDataWrap(self.headingUKFData)
        self.headingUKFWrap.ModelTag = "headingUKF"

        # Initialize all modules
        self.InitAllFSWObjects(SimBase)
        
        # Create tasks
        SimBase.fswProc.addTask(SimBase.CreateNewTask("hillPointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavPointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("headingPointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavPointLimbTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavAttODLimbTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavPointTaskCheat", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("mrpFeedbackTask", self.processTasksTimeStep), 15)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("mrpFeedbackRWsTask", self.processTasksTimeStep), 15)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavODTask", self.processTasksTimeStep), 5)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("imageProcTask", self.processTasksTimeStep), 9)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavODTaskLimb", self.processTasksTimeStep), 15)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavODTaskB", self.processTasksTimeStep), 9)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavAttODTask", self.processTasksTimeStep), 9)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("cnnAttODTask", self.processTasksTimeStep), 9)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("opNavFaultDet", self.processTasksTimeStep), 9)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("attODFaultDet", self.processTasksTimeStep), 9)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("cnnFaultDet", self.processTasksTimeStep), 9)

        SimBase.AddModelToTask("hillPointTask", self.hillPointWrap, self.hillPointData, 10)
        SimBase.AddModelToTask("hillPointTask", self.trackingErrorCamWrap, self.trackingErrorCamData, 9)

        SimBase.AddModelToTask("opNavPointTask", self.imageProcessing, None, 15)
        SimBase.AddModelToTask("opNavPointTask", self.pixelLineWrap, self.pixelLineData, 12)
        SimBase.AddModelToTask("opNavPointTask", self.opNavPointWrap, self.opNavPointData, 9)

        SimBase.AddModelToTask("headingPointTask", self.imageProcessing, None, 15)
        SimBase.AddModelToTask("headingPointTask", self.pixelLineWrap, self.pixelLineData, 12)
        SimBase.AddModelToTask("headingPointTask", self.headingUKFWrap, self.headingUKFData, 10)
        SimBase.AddModelToTask("headingPointTask", self.opNavPointWrap, self.opNavPointData, 9)

        SimBase.AddModelToTask("opNavPointLimbTask", self.limbFinding, None, 25)
        SimBase.AddModelToTask("opNavPointLimbTask", self.horizonNavWrap, self.horizonNavData, 12)
        SimBase.AddModelToTask("opNavPointLimbTask", self.opNavPointWrap, self.opNavPointData, 10)

        SimBase.AddModelToTask("opNavAttODLimbTask", self.limbFinding, None, 25)
        SimBase.AddModelToTask("opNavAttODLimbTask", self.horizonNavWrap, self.horizonNavData, 12)
        SimBase.AddModelToTask("opNavAttODLimbTask", self.opNavPointWrap, self.opNavPointData, 10)
        SimBase.AddModelToTask("opNavAttODLimbTask", self.relativeODWrap, self.relativeODData, 9)

        SimBase.AddModelToTask("opNavODTaskLimb", self.limbFinding, None, 25)
        SimBase.AddModelToTask("opNavODTaskLimb", self.horizonNavWrap, self.horizonNavData, 22)
        SimBase.AddModelToTask("opNavODTaskLimb", self.relativeODWrap, self.relativeODData, 20)

        SimBase.AddModelToTask("opNavPointTaskCheat", self.hillPointWrap, self.hillPointData, 10)
        SimBase.AddModelToTask("opNavPointTaskCheat", self.trackingErrorCamWrap, self.trackingErrorCamData, 9)

        SimBase.AddModelToTask("opNavODTask", self.imageProcessing, None, 15)
        SimBase.AddModelToTask("opNavODTask", self.pixelLineWrap, self.pixelLineData, 14)
        SimBase.AddModelToTask("opNavODTask", self.relativeODWrap, self.relativeODData, 13)

        SimBase.AddModelToTask("opNavODTaskB", self.imageProcessing, None, 15)
        SimBase.AddModelToTask("opNavODTaskB", self.pixelLineFilterWrap, self.pixelLineFilterData, 13)

        SimBase.AddModelToTask("imageProcTask", self.imageProcessing, None, 15)

        SimBase.AddModelToTask("opNavAttODTask", self.imageProcessing, None, 15)
        SimBase.AddModelToTask("opNavAttODTask", self.pixelLineWrap, self.pixelLineData, 14)
        SimBase.AddModelToTask("opNavAttODTask", self.opNavPointWrap, self.opNavPointData, 10)
        # SimBase.AddModelToTask("opNavAttODTask", self.pixelLineFilterWrap, self.pixelLineFilterData, 9)
        SimBase.AddModelToTask("opNavAttODTask", self.relativeODWrap, self.relativeODData, 9)

        SimBase.AddModelToTask("cnnAttODTask", self.opNavCNN, None, 15)
        SimBase.AddModelToTask("cnnAttODTask", self.pixelLineWrap, self.pixelLineData, 14)
        SimBase.AddModelToTask("cnnAttODTask", self.opNavPointWrap, self.opNavPointData, 10)
        # SimBase.AddModelToTask("opNavAttODTask", self.pixelLineFilterWrap, self.pixelLineFilterData, 9)
        SimBase.AddModelToTask("cnnAttODTask", self.relativeODWrap, self.relativeODData, 9)

        SimBase.AddModelToTask("mrpFeedbackTask", self.mrpFeedbackControlWrap, self.mrpFeedbackControlData, 10) #used for external torque

        SimBase.AddModelToTask("mrpFeedbackRWsTask", self.mrpFeedbackRWsWrap, self.mrpFeedbackRWsData, 9)
        SimBase.AddModelToTask("mrpFeedbackRWsTask", self.rwMotorTorqueWrap, self.rwMotorTorqueData, 8)

        SimBase.AddModelToTask("attODFaultDet", self.limbFinding, None, 25)
        SimBase.AddModelToTask("attODFaultDet", self.horizonNavWrap, self.horizonNavData, 20)
        SimBase.AddModelToTask("attODFaultDet", self.imageProcessing, None, 18)
        SimBase.AddModelToTask("attODFaultDet", self.pixelLineWrap, self.pixelLineData, 16)
        SimBase.AddModelToTask("attODFaultDet", self.opNavFaultWrap, self.opNavFaultData, 14)
        SimBase.AddModelToTask("attODFaultDet", self.opNavPointWrap, self.opNavPointData, 10)
        SimBase.AddModelToTask("attODFaultDet", self.relativeODWrap, self.relativeODData, 9)

        SimBase.AddModelToTask("opNavFaultDet", self.limbFinding, None, 25)
        SimBase.AddModelToTask("opNavFaultDet", self.horizonNavWrap, self.horizonNavData, 20)
        SimBase.AddModelToTask("opNavFaultDet", self.imageProcessing, None, 18)
        SimBase.AddModelToTask("opNavFaultDet", self.pixelLineWrap, self.pixelLineData, 16)
        SimBase.AddModelToTask("opNavFaultDet", self.opNavFaultWrap, self.opNavFaultData, 14)
        SimBase.AddModelToTask("opNavFaultDet", self.relativeODWrap, self.relativeODData, 9)

        SimBase.AddModelToTask("cnnFaultDet", self.opNavCNN, None, 25)
        SimBase.AddModelToTask("cnnFaultDet", self.pixelLineWrap, self.pixelLineData, 20)
        SimBase.AddModelToTask("cnnFaultDet", self.imageProcessing, None, 18)
        SimBase.AddModelToTask("cnnFaultDet", self.pixelLineWrap, self.pixelLineData, 16)
        SimBase.AddModelToTask("cnnFaultDet", self.opNavFaultWrap, self.opNavFaultData, 14)
        SimBase.AddModelToTask("cnnFaultDet", self.opNavPointWrap, self.opNavPointData, 10)
        SimBase.AddModelToTask("cnnFaultDet", self.relativeODWrap, self.relativeODData, 9)

        # Create events to be called for triggering GN&C maneuvers
        SimBase.fswProc.disableAllTasks()

        SimBase.createNewEvent("initiateStandby", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'standby'"],
                               ["self.fswProc.disableAllTasks()",
                                ])

        SimBase.createNewEvent("prepOpNav", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'prepOpNav'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointTaskCheat')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("imageGen", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'imageGen'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('imageProcTask')",
                                "self.enableTask('opNavPointTaskCheat')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("pointOpNav", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'pointOpNav'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointTask')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("pointHead", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'pointHead'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('headingPointTask')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("pointLimb", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'pointLimb'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointLimbTask')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("OpNavOD", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'OpNavOD'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointTaskCheat')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.enableTask('opNavODTask')"])

        SimBase.createNewEvent("DoubleOD", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'DoubleOD'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointTaskCheat')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.enableTask('opNavODTaskLimb')",
                                "self.enableTask('opNavODTask')"])

        SimBase.createNewEvent("OpNavODLimb", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'OpNavODLimb'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointTaskCheat')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.enableTask('opNavODTaskLimb')"])

        SimBase.createNewEvent("OpNavODB", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'OpNavODB'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointTaskCheat')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.enableTask('opNavODTaskB')"])

        SimBase.createNewEvent("OpNavAttOD", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'OpNavAttOD'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavAttODTask')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("OpNavAttODLimb", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'OpNavAttODLimb'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavAttODLimbTask')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("CNNAttOD", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'CNNAttOD'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('cnnAttODTask')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("FaultDet", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'FaultDet'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('attODFaultDet')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

        SimBase.createNewEvent("ODFaultDet", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'ODFaultDet'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('opNavPointTaskCheat')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.enableTask('opNavFaultDet')"])

        SimBase.createNewEvent("FaultDetCNN", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'FaultDetCNN'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('cnnFaultDet')",
                                "self.enableTask('mrpFeedbackRWsTask')"])

    # ------------------------------------------------------------------------------------------- #
    # These are module-initialization methods
    def SetHillPointGuidance(self, SimBase):
        self.hillPointData.outputDataName = "att_reference"
        self.hillPointData.inputNavDataName = SimBase.DynModels.SimpleNavObject.outputTransName
        self.hillPointData.inputCelMessName = "mars barycenter_ephemeris_data"

    def SetOpNavPointGuidance(self, SimBase):
        self.opNavPointData.attGuidanceOutMsgName = "att_guidance"
        self.opNavPointData.imuInMsgName = SimBase.DynModels.SimpleNavObject.outputAttName
        self.opNavPointData.cameraConfigMsgName = "camera_config_data"
        self.opNavPointData.opnavDataInMsgName = "output_nav_msg" #"heading_filtered"
        self.opNavPointData.smallAngle = 0.001*np.pi/180.
        self.opNavPointData.timeOut = 1000 # Max time in sec between images before engaging search
        # self.opNavPointData.opNavAxisSpinRate = 0.1*np.pi/180.
        self.opNavPointData.omega_RN_B = [0.001, 0.0, -0.001]
        self.opNavPointData.alignAxis_C = [0.,0.,1]

    def SetHeadingUKF(self):
        self.headingUKFData.opnavOutMsgName = "heading_filtered"
        self.headingUKFData.filtDataOutMsgName = "heading_filter_data"
        self.headingUKFData.opnavDataInMsgName = "output_nav_msg"
        # self.headingUKFData.cameraConfigMsgName = "camera_config_data"

        self.headingUKFData.alpha = 0.02
        self.headingUKFData.beta = 2.0
        self.headingUKFData.kappa = 0.0

        # filterObject.state = [0.0, 0., 0., 0., 0.]
        self.headingUKFData.stateInit = [0.0, 0.0, 1.0, 0.0, 0.0]
        self.headingUKFData.covarInit = [0.2, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.2, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.2, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.005, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.005]

        qNoiseIn = np.identity(5)
        qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3] * 1E-6 * 1E-6
        qNoiseIn[3:5, 3:5] = qNoiseIn[3:5, 3:5] * 1E-6 * 1E-6
        self.headingUKFData.qNoise = qNoiseIn.reshape(25).tolist()
        # self.headingUKFData.qObsVal = 0.001

    def SetAttitudeTrackingError(self, SimBase):
        self.trackingErrorData.inputNavName = SimBase.DynModels.SimpleNavObject.outputAttName
        # Note: SimBase.DynModels.SimpleNavObject.outputAttName = "simple_att_nav_output"
        self.trackingErrorData.inputRefName = "att_reference"
        self.trackingErrorData.outputDataName = "att_guidance"

    ## Celestial point to Mars
    def SetCelTwoBodyMarsPoint(self):
        self.celTwoBodyMarsData.inputNavDataName = "simple_trans_nav_output"
        self.celTwoBodyMarsData.inputCelMessName = "mars barycenter_ephemeris_data"
        self.celTwoBodyMarsData.outputDataName = "att_ref_output"
        self.celTwoBodyMarsData.singularityThresh = 1.0 * math.pi / 180.0

    def SetAttTrackingErrorCam(self, SimBase):
        self.trackingErrorCamData.inputRefName = "att_reference"
        self.trackingErrorCamData.inputNavName = "simple_att_nav_output"
        self.trackingErrorCamData.outputDataName = "att_guidance"

        M2 =  rbk.euler2(90 * macros.D2R) #rbk.euler2(-90 * macros.D2R) #
        M3 =  rbk.euler1(90 * macros.D2R) #rbk.euler3(90 * macros.D2R) #
        M_cam = rbk.MRP2C(SimBase.DynModels.cameraMRP_CB)

        MRP = rbk.C2MRP(np.dot(np.dot(M3, M2), M_cam)) # This assures that the s/c does not control to the hill frame, but to a rotated frame such that the camera is pointing to the planet
        self.trackingErrorCamData.sigma_R0R = MRP
        # self.trackingErrorCamData.sigma_R0R = [1./3+0.1, 1./3-0.1, 0.1-1/3]

    def SetCSSWlsEst(self, SimBase):
        cssConfig = fswMessages.CSSConfigFswMsg()
        totalCSSList = []
        nHat_B_vec = [
            [0.0, 0.707107, 0.707107],
            [0.707107, 0., 0.707107],
            [0.0, -0.707107, 0.707107],
            [-0.707107, 0., 0.707107],
            [0.0, -0.965926, -0.258819],
            [-0.707107, -0.353553, -0.612372],
            [0., 0.258819, -0.965926],
            [0.707107, -0.353553, -0.612372]
        ]
        for CSSHat in nHat_B_vec:
            CSSConfigElement = fswMessages.CSSUnitConfigFswMsg()
            CSSConfigElement.CBias = 1.0
            CSSConfigElement.nHat_B = CSSHat
            totalCSSList.append(CSSConfigElement)
            cssConfig.cssVals = totalCSSList

        cssConfig.nCSS = len(SimBase.DynModels.CSSConstellationObject.sensorList)
        cssConfigSize = cssConfig.getStructSize()
        SimBase.TotalSim.CreateNewMessage("FSWProcess", "css_config_data", cssConfigSize, 2, "CSSConstellation")
        SimBase.TotalSim.WriteMessageData("css_config_data", cssConfigSize, 0, cssConfig)

        self.cssWlsEstData.cssDataInMsgName = SimBase.DynModels.CSSConstellationObject.outputConstellationMessage
        self.cssWlsEstData.cssConfigInMsgName = "css_config_data"
        self.cssWlsEstData.navStateOutMsgName = "sun_point_data"

    def SetMRPFeedbackControl(self, SimBase):
        self.mrpFeedbackControlData.inputGuidName = "att_guidance"
        self.mrpFeedbackControlData.vehConfigInMsgName = "adcs_config_data"
        self.mrpFeedbackControlData.outputDataName = SimBase.DynModels.extForceTorqueObject.cmdTorqueInMsgName
        # Note: SimBase.DynModels.extForceTorqueObject.cmdTorqueInMsgName = "extTorquePntB_B_cmds"
        
        self.mrpFeedbackControlData.K = 3.5
        self.mrpFeedbackControlData.Ki = -1.0 # Note: make value negative to turn off integral feedback
        self.mrpFeedbackControlData.P = 30.0
        self.mrpFeedbackControlData.integralLimit = 2. / self.mrpFeedbackControlData.Ki * 0.1


    def SetMRPFeedbackRWA(self):
        self.mrpFeedbackRWsData.K = 3.5
        self.mrpFeedbackRWsData.Ki = -1  # Note: make value negative to turn off integral feedback
        self.mrpFeedbackRWsData.P = 30.0
        self.mrpFeedbackRWsData.integralLimit = 2. / self.mrpFeedbackRWsData.Ki * 0.1

        self.mrpFeedbackRWsData.vehConfigInMsgName = "adcs_config_data"
        self.mrpFeedbackRWsData.inputRWSpeedsName = "reactionwheel_output_states"
        self.mrpFeedbackRWsData.rwParamsInMsgName = "rwa_config_data"
        self.mrpFeedbackRWsData.inputGuidName = "att_guidance"
        self.mrpFeedbackRWsData.outputDataName = "controlTorqueRaw"


    def SetVehicleConfiguration(self, SimBase):
        vehicleConfigOut = fswMessages.VehicleConfigFswMsg()
        # use the same inertia in the FSW algorithm as in the simulation
        vehicleConfigOut.ISCPntB_B = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]
        unitTestSupport.setMessage(SimBase.TotalSim,
                                   SimBase.FSWProcessName,
                                    "adcs_config_data",
                                    vehicleConfigOut)

    def SetRWConfigMsg(self, SimBase):
        # Configure RW pyramid exactly as it is in the Dynamics (i.e. FSW with perfect knowledge)
        rwElAngle = np.array([40.0, 40.0, 40.0, 40.0]) * macros.D2R
        rwAzimuthAngle = np.array([45.0, 135.0, 225.0, 315.0]) * macros.D2R
        wheelJs = 50.0 / (6000.0 * math.pi * 2.0 / 60)
        
        fswSetupRW.clearSetup()
        for elAngle, azAngle in zip(rwElAngle, rwAzimuthAngle):
            gsHat = (rbk.Mi(-azAngle, 3).dot(rbk.Mi(elAngle, 2))).dot(np.array([1, 0, 0]))
            fswSetupRW.create(gsHat,  # spin axis
                              wheelJs,  # kg*m^2
                              0.2)  # Nm        uMax
        
        fswSetupRW.writeConfigMessage("rwa_config_data", SimBase.TotalSim, SimBase.FSWProcessName)


    def SetRWMotorTorque(self, SimBase):
        controlAxes_B = [
        1.0, 0.0, 0.0
        , 0.0, 1.0, 0.0
        , 0.0, 0.0, 1.0
        ]
        self.rwMotorTorqueData.controlAxes_B = controlAxes_B
        self.rwMotorTorqueData.inputVehControlName = "controlTorqueRaw"
        self.rwMotorTorqueData.outputDataName = SimBase.DynModels.rwStateEffector.InputCmds  # "reactionwheel_cmds"
        self.rwMotorTorqueData.rwParamsInMsgName = "rwa_config_data"

    def SetCNNOpNav(self):
        self.opNavCNN.imageInMsgName = "opnav_image"
        self.opNavCNN.opnavCirclesOutMsgName = "circles_data"
        self.opNavCNN.pixelNoise = [5,5,5]
        self.opNavCNN.pathToNetwork = bskPath + "/../../src/fswAlgorithms/imageProcessing/centerRadiusCNN/position_net2_trained_11-14.onnx"

    def SetImageProcessing(self):
        self.imageProcessing.imageInMsgName = "opnav_image"
        self.imageProcessing.opnavCirclesOutMsgName = "circles_data"

        self.imageProcessing.saveImages = 0
        self.imageProcessing.expectedCircles = 1
        self.imageProcessing.cannyThresh = 200
        self.imageProcessing.voteThresh = 25
        self.imageProcessing.houghMinDist = 50
        self.imageProcessing.houghMinRadius = 20
        self.imageProcessing.blurrSize = 9
        self.imageProcessing.noiseSF = 1
        self.imageProcessing.dpValue = 1
        self.imageProcessing.saveDir = 'Test/'
        self.imageProcessing.houghMaxRadius = 0#int(512 / 1.25)

    def SetPixelLineConversion(self):
        self.pixelLineData.circlesInMsgName = "circles_data"
        self.pixelLineData.cameraConfigMsgName = "camera_config_data"
        self.pixelLineData.attInMsgName = "simple_att_nav_output"
        self.pixelLineData.planetTarget = 2
        self.pixelLineData.opNavOutMsgName = "output_nav_msg"

    def SetLimbFinding(self):
        self.limbFinding.imageInMsgName = "opnav_image"
        self.limbFinding.opnavLimbOutMsgName = "limb_data"

        self.limbFinding.saveImages = 0
        self.limbFinding.cannyThreshLow = 50
        self.limbFinding.cannyThreshHigh = 100
        self.limbFinding.blurrSize = 5
        self.limbFinding.limbNumThresh = 0

    def SetHorizonNav(self):
        self.horizonNavData.limbInMsgName = "limb_data"
        self.horizonNavData.cameraConfigMsgName = "camera_config_data"
        self.horizonNavData.attInMsgName = "simple_att_nav_output"
        self.horizonNavData.planetTarget = 2
        self.horizonNavData.noiseSF = 1 #2 should work though
        self.horizonNavData.opNavOutMsgName = "output_nav_msg"

    def SetRelativeODFilter(self):
        self.relativeODData.navStateOutMsgName = "relod_state_estimate"
        self.relativeODData.filtDataOutMsgName = "relod_filter_data"
        self.relativeODData.opNavInMsgName = "output_nav_msg"

        self.relativeODData.planetIdInit = 2
        self.relativeODData.alpha = 0.02
        self.relativeODData.beta = 2.0
        self.relativeODData.kappa = 0.0
        self.relativeODData.noiseSF = 7.5

        mu = 42828.314 * 1E9  # m^3/s^2
        elementsInit = orbitalMotion.ClassicElements()
        elementsInit.a = 10000 * 1E3  # m
        elementsInit.e = 0.2
        elementsInit.i = 10 * macros.D2R
        elementsInit.Omega = 25. * macros.D2R
        elementsInit.omega = 10. * macros.D2R
        elementsInit.f = 40 * macros.D2R
        r, v = orbitalMotion.elem2rv(mu, elementsInit)

        self.relativeODData.stateInit = r.tolist() + v.tolist()
        self.relativeODData.covarInit = [1. * 1E6, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 1. * 1E6, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 1. * 1E6, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.02 * 1E6, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.02 * 1E6, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.02 * 1E6]

        qNoiseIn = np.identity(6)
        qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3] * 1E-3 * 1E-3
        qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6] * 1E-4 * 1E-4
        self.relativeODData.qNoise = qNoiseIn.reshape(36).tolist()

    def SetFaultDetection(self):
        self.opNavFaultData.navMeasPrimaryMsgName = "primary_opnav"
        self.opNavFaultData.navMeasSecondaryMsgName = "secondary_opnav"
        self.opNavFaultData.cameraConfigMsgName = "camera_config_data"
        self.opNavFaultData.attInMsgName = "simple_att_nav_output"
        self.opNavFaultData.opNavOutMsgName = "output_nav_msg"
        self.opNavFaultData.sigmaFault = 0.3
        self.opNavFaultData.faultMode = 0

    def SetPixelLineFilter(self):
        self.pixelLineFilterData.navStateOutMsgName = "pixelLine_state_estimate"
        self.pixelLineFilterData.filtDataOutMsgName = "pixelLine_filter_data"
        self.pixelLineFilterData.circlesInMsgName = "circles_data"
        self.pixelLineFilterData.cameraConfigMsgName = "camera_config_data"
        self.pixelLineFilterData.attInMsgName = "simple_att_nav_output"

        self.pixelLineFilterData.planetIdInit = 2
        self.pixelLineFilterData.alpha = 0.02
        self.pixelLineFilterData.beta = 2.0
        self.pixelLineFilterData.kappa = 0.0
        self.pixelLineFilterData.gamma = 0.9

        mu = 42828.314 * 1E9  # m^3/s^2
        elementsInit = orbitalMotion.ClassicElements()
        elementsInit.a = 10000 * 1E3  # m
        elementsInit.e = 0.2
        elementsInit.i = 10 * macros.D2R
        elementsInit.Omega = 25. * macros.D2R
        elementsInit.omega = 10. * macros.D2R
        elementsInit.f = 40 * macros.D2R
        r, v = orbitalMotion.elem2rv(mu, elementsInit)
        bias = [1, 1, 2]

        self.pixelLineFilterData.stateInit = r.tolist() + v.tolist() + bias
        self.pixelLineFilterData.covarInit = [10. * 1E6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 10. * 1E6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 10. * 1E6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.01 * 1E6, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.01 * 1E6, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.01 * 1E6, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1]


        qNoiseIn = np.identity(9)
        qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3] * 1E-3 * 1E-3
        qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6] * 1E-4 * 1E-4
        qNoiseIn[6:9, 6:9] = qNoiseIn[6:9, 6:9] * 1E-8 * 1E-8
        self.pixelLineFilterData.qNoise = qNoiseIn.reshape(9 * 9).tolist()


    # Global call to initialize every module
    def InitAllFSWObjects(self, SimBase):
        self.SetHillPointGuidance(SimBase)
        self.SetCSSWlsEst(SimBase)
        self.SetAttitudeTrackingError(SimBase)
        self.SetMRPFeedbackControl(SimBase)
        self.SetVehicleConfiguration(SimBase)
        self.SetRWConfigMsg(SimBase)
        self.SetMRPFeedbackRWA()
        self.SetRWMotorTorque(SimBase)
        self.SetAttTrackingErrorCam(SimBase)
        self.SetImageProcessing()
        self.SetPixelLineConversion()

        self.SetCNNOpNav()
        self.SetRelativeODFilter()
        self.SetFaultDetection()
        # J. Christian methods
        self.SetLimbFinding()
        self.SetHorizonNav()

        self.SetOpNavPointGuidance(SimBase)
        self.SetHeadingUKF()
        self.SetPixelLineFilter()


#BSKFswModels()
