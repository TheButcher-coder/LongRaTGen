class RaTGen:
    def __init__(self):
        self.traj = None        #maybe Pandas?
        self.max_q = None       #Contains every joint's max angle
        self.max_accel = None   #Contains max acceleration
        self.max_vel = None     #Contains max velocity


    def generate_sin(self, amp, freq, phase=0):

    def generate_custom(self, fun, t0, tmax, dt):

    def generate_punch(self, ):     #Whats required?

    def generate_movement(self, ):    #Hard coded movement as list of Transforms

    def add_traj(self, traj1, traj2):   #Add two trajectories

    def add_random_noise(self, noise):  #Adds random noise to trajectory    maybe custom Noise function?

    def write_csv(self, outfile):      #Writes to file-> q as csv

    def read_csv(self, infile):        #Reads from file