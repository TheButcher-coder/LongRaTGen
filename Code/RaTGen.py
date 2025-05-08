import numpy as np
import pandas as pd
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import scipy as sc


class RaTGen:
    def __init__(self):
        self.traj = []
        self.max_q = None       #Contains every joint's max angle(s)
        self.max_accel = 1   #Contains max acceleration
        self.max_vel = 10    #Contains max velocity
        self.dt = .1          #Contains time diff
        self.bot = None    #Contains robot object for inverse Kinematics
        self.nullpos = [[],[],[],[]]   #nullposition
        self.mean = np.array(0)
        self.std_dev = np.array(.1)
        self.rng = OrnsteinUhlenbeckActionNoise(self.mean, self.std_dev)  #Noise generator

    def make_traj(self, p, rot):
        # Ensure p and rot are numpy arrays
        p = np.array(p)
        rot = np.array(rot)

        # Determine the length of the trajectory
        length = max(len(p), len(rot))

        # Initialize the transformation matrix
        temp = np.zeros([length, 4, 4])

        # Assign the position and rotation to the transformation matrix
        for i in range(length):
            if len(rot) > len(p):
                temp[i, :3, :3] = rot[i] if i < len(rot) else np.eye(3)
                temp[i, :3, 3] = p
                temp[i, 3, 3] = 1
            elif len(p) == len(rot):
                temp[i, :3, :3] = rot[i]
                temp[i, :3, 3] = p[i]
                temp[i, 3, 3] = 1
            else:
                temp[i, :3, :3] = rot
                temp[i, :3, 3] = p[i] if i < len(p) else np.zeros(3)
                temp[i, 3, 3] = 1

        self.traj.append(temp)

    def generate_rot_X(self, t0, tmax):
        t = np.arange(t0, tmax, self.dt)
        return np.array([
            [[1, 0, 0],
             [0, np.cos(ti), -np.sin(ti)],
             [0, np.sin(ti), np.cos(ti)]]
            for ti in t
        ])

    def generate_rot_Y(self, t0, tmax):
        t = np.arange(t0, tmax, self.dt)
        return np.array([
            [[np.cos(ti), 0, np.sin(ti)],
             [0, 1, 0],
             [-np.sin(ti), 0, np.cos(ti)]]
            for ti in t
        ])

    def generate_rot_Z(self, t0, tmax):
        t = np.arange(t0, tmax, self.dt)
        return np.array([
            [[np.cos(ti), -np.sin(ti), 0],
             [np.sin(ti), np.cos(ti), 0],
             [0, 0, 1]]
            for ti in t
        ])

    def generate_sin(self, amp, freq, phase=0, t0=0, tmax=2*np.pi):    #Generates a sin trajectory
        return amp * np.sin(freq * np.arange(t0, tmax, self.dt) + phase)

    def generate_cos(self, amp, freq, phase=0, t0=0, tmax=2*np.pi):    #Generates a cos trajectory
        return amp * np.cos(freq * np.arange(t0, tmax, self.dt) + phase)

    def generate_custom(self, fun, t0, tmax):
        t = np.arange(t0, tmax, self.dt)
        #self.traj.append(fun(t))
        return fun(t)

    def generate_punch(self, t0, tmax, a, b, c):     #Whats required?
        #https: // www.desmos.com / calculator / z8txe5wywh?lang = de
        t = np.linspace(t0, tmax, int((tmax-t0)/self.dt))
        return ( a*t^(2*b)*np.exp(-c*t))  # not implemented


    def add_traj(self, num1, num2):
        len1 = len(self.traj[num1])
        len2 = len(self.traj[num2])
        if len1 > len2:
            self.traj[num2] = np.append(self.traj[num2], np.zeros(len1-len2))
        else:
            self.traj[num1] = np.append(self.traj[num1], np.zeros(len2-len1))

        self.traj[num1] += self.traj[num2]

    def generate_noise(self, t0, tmax):
        t = np.arange(t0, tmax, self.dt)

        values = []
        for _ in t:
            value = self.rng()
            values.append(value)
        return values

    def del_traj(self, num):    #Deletes trajectory
        del self.traj[num]

    def set_max_q(self, max_q):     #Sets max_q
        self.max_q = max_q
    def get_max_q(self):        #Returns max_q
        return self.max_q

    def set_max_accel(self, max_accel):     #Sets max_accel
        self.max_accel = max_accel*self.dt
    def get_max_accel(self):        #Returns max_accel
        return self.max_accel

    def set_max_vel(self, max_vel):     #Sets max_vel
        self.max_vel = max_vel*self.dt
    def get_max_vel(self):        #Returns max_vel
        return self.max_vel

    def get_traj_q(self, kart):
        return self.bot.ik_LM(kart)  # not implemented


    def set_dt(self, dt):     #Sets dt
        self.dt = dt
    def get_dt(self):        #Returns dt
        return self.dt

    def trans_kart_to_angles(self, kart):
        return -69  # not implemented

    #IO Functions
    def get_traj(self):     #Returns trajectory
        return self.traj
    def write_csv(self, outfile):      #Writes to file-> q as csv
        return -69  # not implemented
    def read_csv(self, infile):        #Reads from file
        return -69  # not implemented


    # Setter & Getter für mean
    def set_mean(self, mean):
        self.mean = np.array(mean)
        self.rng = OrnsteinUhlenbeckActionNoise(self.mean, self.std_dev)

    def get_mean(self):
        return self.mean

    # Setter & Getter für std_dev
    def set_std_dev(self, std_dev):
        self.std_dev = np.array(std_dev)
        self.rng = OrnsteinUhlenbeckActionNoise(self.mean, self.std_dev)

    def get_std_dev(self):
        return self.std_dev

    def smooth(self, traj):   #Smooth trajectory with specified acceleration
        t0 = traj[0]

        dt = np.diff(traj)
        v0 = dt[0]
        dt[dt > self.max_vel] = self.max_vel
        dt[dt < -self.max_vel] = -self.max_vel

        ddt = np.diff(dt)
        ddt[ddt > self.max_accel] = self.max_accel
        ddt[ddt < -self.max_accel] = -self.max_accel

        dt_new = np.cumsum(ddt) + v0
        traj_new = np.cumsum(dt_new)

        return traj_new


    def smooth_add_noise(self, traj):
        t0 = traj[0]

        dt = np.diff(traj)/self.dt

        ddt = np.diff(dt)/self.dt

        for i in range(len(ddt)):
            ddt[i] += self.rng()
        ddt[ddt > self.max_accel] = self.max_accel
        ddt[ddt < -self.max_accel] = -self.max_accel


        dt_new = np.cumsum(ddt)*self.dt

        dt_new[dt_new > self.max_vel] = self.max_vel
        dt_new[dt_new < -self.max_vel] = -self.max_vel

        traj_new = np.cumsum(dt_new)*self.dt + t0
        return traj_new

    def smooth_add_noise2(self, traj):
        t0 = traj[0]

        dt = np.diff(traj)/self.dt

        ddt = np.diff(dt)/self.dt

        ddt[ddt > self.max_accel] = self.max_accel
        ddt[ddt < -self.max_accel] = -self.max_accel

        dt_new = np.cumsum(ddt)*self.dt

        for i in range(len(dt_new)):
            dt_new[i] += self.rng()

        dt_new[dt_new > self.max_vel] = self.max_vel
        dt_new[dt_new < -self.max_vel] = -self.max_vel

        traj_new = np.cumsum(dt_new)*self.dt + t0
        return traj_new

    def set_nullpos(self, nullpos):
        self.nullpos = nullpos
    def get_nullpos(self):
        return self.nullpos