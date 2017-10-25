import pickle

class Configuration:
    def __init__(self):
        self.conf={}
        self.conf['env-id'] = 'HumanoidBalanceFilter-v0'
        self.conf['render-eval'] = False

        self.conf['joint-interpolation'] = True
        self.conf['action-bounds'] = [[-0.13,-2.42,-0.08,-0.93],[0.67,1.62,2.06,0.65]] #waist,hip,knee,ankle
        self.conf['LLC-frequency'] =500
        self.conf['HLC-frequency'] = 25

        self.conf['batch-size'] = 64

        self.conf['critic-layer-norm'] = False
        self.conf['critic-observation-norm'] = False #use batch norm to normalize observations
        self.conf['critic-l2-reg'] = 1e-2
        self.conf['critic-lr'] = 1e-3
        self.conf['critic-layer-size'] = [512,256]
        self.conf['critic-activation-fn'] = 'leaky_relu'

        self.conf['actor-layer-norm'] = False
        self.conf['actor-observation-norm'] = False #use batch norm to normalize observations
        self.conf['actor-lr'] = 1e-4
        self.conf['actor-layer-size'] = [512,256]
        self.conf['render'] = False
        self.conf['normalize-returns'] = False
        self.conf['normalize-observations'] = True
        self.conf['actor-activation-fn'] = 'leaky_relu'

        self.conf['tau'] = 0.001
        self.conf['gamma'] = 0.99
        self.conf['popart'] = False

        self.conf['prioritized-exp-replay'] = True
        self.conf['replay-buffer-size'] = 100000
        self.conf['replay-start-size'] = 1000

        self.conf['reward-scale'] = 1.0
        self.conf['epoch-num'] = 2000
        self.conf['epoch-step-num'] = 2000
        self.conf['total-step-num'] = 2500000
        self.conf['test-num'] = 1

        self.conf['param-noise'] = False
        self.conf['param-noise-settings'] = [0.1,0.1,1.01] #initial_stddev, desired_action_stddev, adoption_coefficient [0.1,0.1,1.01]
        self.conf['OU-noise'] = True
        self.conf['OU-noise-settings'] = [0.0,0.15,0.2] #mu, theta, sigma

    def save_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/configuration.obj', 'wb')
        pickle.dump(self.conf, output)
        output.close()

    def load_configuration(self,dir):
        # write python dict to a file
        pkl_file = open(dir + '/configuration.obj', 'rb')
        conf_temp = pickle.load(pkl_file)
        self.conf=conf_temp
        pkl_file.close()

    def record_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/readme.txt', 'w')
        for key in self.conf:
            output.write("{}: {}\n".format(key,self.conf[key]))

    def print_configuration(self):
        for key in self.conf:
            print(key + ': ' + str(self.conf[key]))