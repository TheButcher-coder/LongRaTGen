import numpy as np
import pandas as pd

class RaTGen:
    def __init__(self):
        self.traj = pd.DataFrame()        #maybe Pandas?
        self.max_q = None       #Contains every joint's max angle
        self.max_accel = None   #Contains max acceleration
        self.max_vel = None     #Contains max velocity
        self.dt = .1          #Contains time diff


    def generate_sin(self, amp, freq, phase=0, t0=0, tmax=2*np.pi):    #Generates a sin trajectory
        self.traj['sin_1'] = amp * np.sin(2 * np.pi * freq * np.arange(t0, tmax, self.dt) + phase)

    def generate_custom(self, fun, t0, tmax):
        t = np.linspace(t0, tmax, int((tmax-t0)/self.dt))
        self.traj['custom'] = fun(t)


    def generate_punch(self, force):     #Whats required?
        return -69  # not implemented
    def generate_movement(self, ):    #Hard coded movement as list of Transforms
        return -69  # not implemented
    def add_traj(self, traj1, traj2, t_diff=0):   #Add two trajectories
        return -69  # not implemented
    def add_random_noise(self, noise):  #Adds random noise to trajectory    maybe custom Noise function?
        return -69  # not implemented


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