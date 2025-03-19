class RaTGen:
    def __init__(self):
        self.traj = None        #maybe Pandas?
        self.max_q = None       #Contains every joint's max angle
        self.max_accel = None   #Contains max acceleration
        self.max_vel = None     #Contains max velocity


    def generate_sin(self, amp, freq, phase=0):

    def generate_custom(self, fun, t0, tmax, dt):

    def generate_punch(self, ):

    def generate_movement(self):

    def add_traj(self, traj1, traj2):

    def write_traj(self, outfile):

    def read_traj(self, infile):