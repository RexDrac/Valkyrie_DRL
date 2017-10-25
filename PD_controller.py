import numpy as np

Kp_hip = 1080.0  # 680  # 700
Ki_hip = 0.0  # 50
Kd_hip = 70.0  # 5  # 10

Kp_knee = 2580.0  # 1380  # 1379
Ki_knee = 0.0  # 100
Kd_knee = 150.0  # 30  # 25

Kp_ankle = 3160.0  # 2460  # 2500
Ki_ankle = 0.0  # 200
Kd_ankle = 300.0  # 50  # 45

Kp_waist = 720.0  # 520  # 520
Ki_waist = 0.0  # 50
Kd_waist = 60.0  # 5  # 10

TORQUE_HIP=350.0*2
TORQUE_KNEE=350.0*2
TORQUE_ANKLE=205.0*2
TORQUE_WAIST=150.0

class PDController:
    def __init__(self):
        self.Kp_hip=1080
        self.Ki_hip=0
        self.Kd_hip=70

        self.Kp_knee=2580
        self.Ki_knee=0
        self.Kd_knee=150

        self.Kp_ankle=3160
        self.Ki_ankle=0
        self.Kd_ankle=300

        self.Kp_waist=720
        self.Ki_waist=0
        self.Kd_waist=60

        self.hip_e_i=0.0
        self.knee_e_i=0.0
        self.ankle_e_i=0.0
        self.waist_e_i=0.0

        self.TORQUE_HIP=350*2
        self.TORQUE_KNEE=350*2
        self.TORQUE_ANKLE=205*2
        self.TORQUE_WAIST=150

        self.HIP_RANGE=np.asarray([-1.62,2.42])*1.
        self.KNEE_RANGE=np.asarray([-2.06,0.08])*1.

        self.ANKLE_RANGE=np.asarray([-0.65,0.93])*1.
        self.WAIST_RANGE=np.asarray([-0.67,0.13])*1.


    def controller(self,SP, PV):
        action=[0,0,0,0]
        #SP=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        SP[0]=self.translate(SP[0],[-1.0,1.0],self.HIP_RANGE)
        SP[1]=self.translate(SP[1],[-1.0,1.0],self.KNEE_RANGE)
        SP[2]=self.translate(SP[2],[-1.0,1.0],self.ANKLE_RANGE)
        SP[3]=self.translate(SP[3],[-1.0,1.0],self.WAIST_RANGE)

        #print(SP)

        #SP[0]=self.translate(SP[0],[-1.0,1.0],[-3.14,3.14])
        #SP[1]=self.translate(SP[1],[-1.0,1.0],[-3.14,3.14])
        #SP[2]=self.translate(SP[2],[-1.0,1.0],[-3.14,3.14])
        #SP[3]=self.translate(SP[3],[-1.0,1.0],[-3.14,3.14])

        hip_e=float(SP[0]-PV[0])
        knee_e=float(SP[1]-PV[1])
        ankle_e=float(SP[2]-PV[2])
        waist_e=float(SP[3]-PV[3])

        hip_de=float(SP[4]-PV[4])
        knee_de=float(SP[5]-PV[5])
        ankle_de=float(SP[6]-PV[6])
        waist_de=float(SP[7]-PV[7])

        self.hip_e_i = hip_e + self.hip_e_i
        self.knee_e_i = knee_e + self.knee_e_i
        self.ankle_e_i = ankle_e + self.ankle_e_i
        self.waist_e_i = waist_e + self.waist_e_i

        self.hip_e_i=float(max(min(self.hip_e_i,31.4),-31.4))
        self.knee_e_i=float(max(min(self.knee_e_i,31.4),-31.4))
        self.ankle_e_i=float(max(min(self.ankle_e_i,31.4),-31.4))
        self.waist_e_i=float(max(min(self.waist_e_i,31.4),-31.4))

        action[0] = Kp_hip * (hip_e) + Kd_hip * hip_de# + Ki_hip * self.hip_e_i
        #        hip_todo[1] = Kp_hip*(hip_e[1]) + Kd_hip*hip_de[1]
        action[1] = Kp_knee * (knee_e) + Kd_knee * knee_de# + Ki_knee * self.knee_e_i
        #        knee_todo[1] = Kp_knee*(knee_e[1]) + Kd_knee*knee_de[1]
        action[2] = Kp_ankle * (ankle_e) + Kd_ankle * ankle_de# + Ki_ankle * self.ankle_e_i
        #        ankle_todo[1] = Kp_ankle*(ankle_e[1]) + Kd_ankle*ankle_de[1]
        action[3] = Kp_waist * (waist_e) + Kd_waist * waist_de# + Ki_waist * self.waist_e_i

        action[0]=action[0]/TORQUE_HIP
        action[1]=action[1]/TORQUE_KNEE
        action[2]=action[2]/TORQUE_ANKLE
        action[3]=action[3]/TORQUE_WAIST
        #print(action)

        return action

    def reset(self):
        self.hip_e_i=0.0
        self.knee_e_i=0.0
        self.ankle_e_i=0.0
        self.waist_e_i=0.0

    def translate(self,value,old_range,new_range):
        OldRange=old_range[1]-old_range[0]
        NewRange=new_range[1]-new_range[0]
        NewValue=(value-old_range[0])*NewRange/OldRange+new_range[0]
        return NewValue

    def translate_network_output(self,value):
        NewValue=[0,0,0,0]
        NewValue[0] = self.translate(value[0],[-1,1],self.HIP_RANGE)
        NewValue[1] = self.translate(value[1], [-1, 1], self.KNEE_RANGE)
        NewValue[2] = self.translate(value[2], [-1, 1], self.ANKLE_RANGE)
        NewValue[3] = self.translate(value[3], [-1, 1], self.WAIST_RANGE)
        return NewValue

    def translate_PD_input(self,value):
        NewValue=[0,0,0,0]
        NewValue[0] = self.translate(value[0],self.HIP_RANGE,[-1,1])
        NewValue[1] = self.translate(value[1], self.KNEE_RANGE, [-1, 1])
        NewValue[2] = self.translate(value[2], self.ANKLE_RANGE, [-1, 1])
        NewValue[3] = self.translate(value[3], self.WAIST_RANGE, [-1, 1])
        return NewValue

    def set_PD_parameters(self,P,I,D):
        self.Kp_hip=P[0]
        self.Ki_hip=I[0]
        self.Kd_hip=D[0]

        self.Kp_knee=P[1]
        self.Ki_knee=I[1]
        self.Kd_knee=D[1]

        self.Kp_ankle=P[2]
        self.Ki_ankle=I[2]
        self.Kd_ankle=D[2]

        self.Kp_waist=P[3]
        self.Ki_waist=I[3]
        self.Kd_waist=D[3]

    def get_PD_parameters(self):
        P = [self.Kp_hip,self.Kp_knee,self.Kp_ankle,self.Kp_waist]
        I = [self.Ki_hip, self.Ki_knee, self.Ki_ankle, self.Ki_waist]
        D = [self.Kd_hip, self.Kd_knee, self.Kd_ankle, self.Kd_waist]

        return [P,I,D]
