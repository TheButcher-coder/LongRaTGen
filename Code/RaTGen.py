import numpy as np
import pandas as pd

class RaTGen:
    def __init__(self):
        self.traj = []       #maybe Pandas?
        self.max_q = None       #Contains every joint's max angle
        self.max_accel = None   #Contains max acceleration
        self.max_vel = None     #Contains max velocity
        self.dt = .1          #Contains time diff


    def make_traj(self, x, y, z, alpha, beta, gamma):
        #construct 3d Trajectory with 6DOF and append to traj
        if len(x) == len(y) == len(z) == len(alpha) == len(beta) == len(gamma):
            temp = np.zeros(4)
            temp[3, :3] = np.array([x, y, z])

            self.traj.append(np.array([x, y, z, alpha, beta, gamma]))

    def generate_sin(self, amp, freq, phase=0, t0=0, tmax=2*np.pi):    #Generates a sin trajectory
        return amp * np.sin(2 * np.pi * freq * np.arange(t0, tmax, self.dt) + phase)

    def generate_cos(self, amp, freq, phase=0, t0=0, tmax=2*np.pi):    #Generates a cos trajectory
        return amp * np.cos(2 * np.pi * freq * np.arange(t0, tmax, self.dt) + phase)

    def generate_custom(self, fun, t0, tmax):
        t = np.arange(t0, tmax, self.dt)
        #self.traj.append(fun(t))
        return fun(t)

    def generate_punch(self, t0, tmax, a, b, c):     #Whats required?
        #https: // www.desmos.com / calculator / z8txe5wywh?lang = de
        t = np.linspace(t0, tmax, int((tmax-t0)/self.dt))
        return ( a*t^(2*b)*np.exp(-c*t))  # not implemented


    #def generate_movement(self, input):    #Hard coded movement as list of Transforms
     #   self.traj.append(input)
    #def add_traj(self, traj1, traj2, t_diff=0):   #Add two trajectories
    #    len1 = len(traj1)
    #    len2 = len(traj2)
    #    if len1 > len2:
    #        traj2 = np.append(traj2, np.zeros(len1-len2))
    #    else:
    #        traj1 = np.append(traj1, np.zeros(len2-len1))
#
     #   self.traj.append(traj1 + traj2)

    def add_traj(self, num1, num2):
        len1 = len(self.traj[num1])
        len2 = len(self.traj[num2])
        if len1 > len2:
            self.traj[num2] = np.append(self.traj[num2], np.zeros(len1-len2))
        else:
            self.traj[num1] = np.append(self.traj[num1], np.zeros(len2-len1))

        self.traj[num1] += self.traj[num2]

    def add_random_noise(self, amp, t0, tmax):
        t = np.arange(t0, tmax, self.dt)
        noise = amp * np.random.randn(len(t))
        return noise

    def del_traj(self, num):    #Deletes trajectory
        del self.traj[num]

    def set_max_q(self, max_q):     #Sets max_q
        self.max_q = max_q
    def get_max_q(self):        #Returns max_q
        return self.max_q

    def set_max_accel(self, max_accel):     #Sets max_accel
        self.max_accel = max_accel
    def get_max_accel(self):        #Returns max_accel
        return self.max_accel

    def set_max_vel(self, max_vel):     #Sets max_vel
        self.max_vel = max_vel
    def get_max_vel(self):        #Returns max_vel
        return self.max_vel

    def set_dt(self, dt):     #Sets dt
        self.dt = dt
    def get_dt(self):        #Returns dt
        return self.dt

    #IO Functions
    def get_traj(self):     #Returns trajectory
        return self.traj
    def write_csv(self, outfile):      #Writes to file-> q as csv
        return -69  # not implemented
    def read_csv(self, infile):        #Reads from file
        return -69  # not implemented