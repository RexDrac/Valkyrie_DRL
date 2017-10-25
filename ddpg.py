import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from replay_buffer import ReplayBuffer
from grad_inverter import grad_inverter
from prioritized_replay import *
from configuration import *
from normalize2 import *

class DDPG:
    """docstring for DDPG"""

    def __init__(self, env, config):
        self.name = 'DDPG'  # name for uploading results
        self.environment = env
        self.config = config
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env._observationDim
        self.action_dim = 4 # waist, hip, knee, ankle env._actionDim

        self.replay_buffer_size = config.conf['replay-buffer-size']
        self.replay_start_size = config.conf['replay-start-size']
        self.batch_size = config.conf['batch-size']
        self.gamma = config.conf['gamma']
        self.action_bounds = config.conf['action-bounds']

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, config)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim, config)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.memory = Memory(capacity=self.replay_buffer_size)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        self.grad_inv = grad_inverter(self.action_bounds, self.sess)

        self.actor_network.perturb_policy()

        self.ob_normalize1 = BatchNormallize(self.state_dim, config.conf['replay-buffer-size'])#TODO test observation normalization
        self.ob_normalize2 = OnlineNormalize(self.state_dim, config.conf['replay-buffer-size'])#TODO test observation normalization


    def train(self):
        train_num = 1
        for i in range(0, train_num):
            # Sample a random minibatch of N transitions from replay buffer
            tree_idx, batch_memory, ISWeights = [],[],[]
            if self.config.conf['prioritized-exp-replay'] ==True:
                tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
            else:
                batch_memory = self.replay_buffer.get_batch(self.batch_size)

            state_batch = np.asarray([data[0] for data in batch_memory])
            action_batch = np.asarray([data[1] for data in batch_memory])
            reward_batch = np.asarray([data[2] for data in batch_memory])
            next_state_batch = np.asarray([data[3] for data in batch_memory])
            done_batch = np.asarray([data[4] for data in batch_memory])

            self.ob_normalize1.update(state_batch)#TODO test observation normalization
            if self.config.conf['normalize-observations']:
                state_batch = self.ob_normalize1.normalize(state_batch)
                next_state_batch = self.ob_normalize1.normalize(next_state_batch)

            # for action_dim = 1
            action_batch = np.resize(action_batch, [self.batch_size, self.action_dim])

            # Calculate y_batch

            next_action_batch = self.actor_network.actions_target(next_state_batch)
            q_value_batch = self.critic_network.q_value_target(next_state_batch, next_action_batch)
            y_batch = []
            for i in range(self.batch_size):
                if done_batch[i]:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + self.gamma * q_value_batch[i])
            y_batch = np.resize(y_batch, [self.batch_size, 1])
            # Update critic by minimizing the loss L
            #print(self.memory.tree.data_pointer)
            #print(tree_idx)
            if self.config.conf['prioritized-exp-replay'] ==True:
                #print(ISWeights)
                ISWeights = np.asarray(ISWeights)
                ISWeights = np.resize(ISWeights, [self.batch_size, 1])
                #print(np.size(ISWeights))
                abs_errors = self.critic_network.train(y_batch, state_batch, action_batch, ISWeights)
                #print(abs_errors)
                self.memory.batch_update(tree_idx, abs_errors)
            else:
                self.critic_network.train(y_batch, state_batch, action_batch, [])

            # Update the actor policy using the sampled gradient:
            action_batch_for_gradients = self.actor_network.actions(state_batch)
            q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)
            q_gradient_batch = self.grad_inv.invert(q_gradient_batch,action_batch_for_gradients)

            self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def action_noise(self, state, epsilon=1, lamda=1):
        # Select action a_t according to the current policy and exploration noise
        if self.config.conf['param-noise'] == True:
            action = self.actor_network.action_noise(state)
            #print(self.actor_network.action_noise(state)-self.actor_network.action(state))
        else:
            action = self.actor_network.action(state)
        # print(self.exploration_noise.noise(action))
        if self.config.conf['OU-noise'] == True:
            ou_noise = epsilon * lamda * self.exploration_noise.noise(action)
            action = action + ou_noise
        return action

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def actions_target(self, state):
        action = self.actor_network.actions_target(state)
        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.store_transition(state, action, reward, next_state, done)

        self.ob_normalize2.update(np.resize(state, [1, self.state_dim]))#TODO test observation normalization
        if self.config.conf['prioritized-exp-replay'] == True:
            # Store transitions to replay start size then start training
            if self.memory.tree.data_pointer > self.replay_start_size:
                self.train()
        else:
            if self.replay_buffer.count() > self.replay_start_size:
                self.train()

    def reset(self):
        # Re-iniitialize the random process when an episode ends
        if self.config.conf['param-noise'] == True:
            self.param_noise()
            #print(conf['param-noise'] == True)
        if self.config.conf['OU-noise'] == True:
            self.exploration_noise.reset()

    def save_weight(self, time_step, dir_path):
        print("Now we save model")
        self.actor_network.save_network(time_step, dir_path)
        self.critic_network.save_network(time_step, dir_path)

    def load_weight(self, dir_path):
        # Now load the weight
        print("Now we load the weight")
        self.actor_network.load_network(dir_path)
        self.critic_network.load_network(dir_path)

    def save_memory(self,filename):
        self.replay_buffer.save_menory(filename)

    def load_memory(self,filename):
        self.replay_buffer.load_memory(filename)

    def param_noise(self):
        #update parameter noise spec
        #self.actor_network.perturb_policy()
        if self.config.conf['prioritized-exp-replay'] == True:
            if self.memory.tree.data_pointer > self.replay_start_size:
                batch_memory = self.memory.sample_random(self.batch_size)
                #tree_idx, batch_memory, ISWeights = self.memory.sample(BATCH_SIZE)

                state_batch = np.asarray([data[0] for data in batch_memory])
                distance = self.actor_network.adapt_param_noise(state_batch)

        else:
            if self.replay_buffer.count() > self.replay_start_size:
                batch_memory = self.replay_buffer.get_batch(self.batch_size)

                state_batch = np.asarray([data[0] for data in batch_memory])
                distance = self.actor_network.adapt_param_noise(state_batch)

        #print(distance)
        #print(self.actor_network.param_noise.current_stddev)
        self.actor_network.perturb_policy()

    def store_transition(self, s, a, r, s_, d):
        if self.config.conf['prioritized-exp-replay'] == True:
            transition=(s,a,r,s_,d)
            self.memory.store(transition)    # have high priority for newly arrived transition
            #print(self.memory.tree.data_pointer)
        else:
            self.replay_buffer.add(s, a, r, s_, d)
            #print(self.replay_buffer.count())


