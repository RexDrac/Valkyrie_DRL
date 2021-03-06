import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from parameter_noise import AdaptiveParamNoise
import math
from configuration import *

class ActorNetwork:
    """docstring for ActorNetwork"""

    def __init__(self, sess, state_dim, action_dim, config):
        self.param_noise = AdaptiveParamNoise()
        self.layer1_size = config.conf['actor-layer-size'][0]
        self.layer2_size = config.conf['actor-layer-size'][1]
        self.learning_rate = config.conf['actor-lr']
        self.tau = config.conf['tau']
        self.is_param_noise = config.conf['param-noise']
        self.is_layer_norm = config.conf['actor-layer-norm']
        self.is_observation_norm = config.conf['actor-observation-norm']
        if config.conf['actor-activation-fn'] == "relu":
            self.activation_fn = tf.nn.relu
        elif config.conf['actor-activation-fn'] == "elu":
            self.activation_fn = tf.nn.elu
        else:
            self.activation_fn = None

        #self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        self.actor_network = {}
        self.state_input, \
        self.action_output, \
        self.actor_network['vars'], \
        self.actor_network['trainable_vars'], \
        self.actor_network['perturbable_vars'],\
        self.is_training = self.create_network(state_dim, action_dim, 'actor_network')

        # create target actor network
        self.target_actor_network = {}
        self.target_state_input, \
        self.target_action_output, \
        self.target_actor_network['vars'], \
        self.target_actor_network['trainable_vars'], \
        self.target_actor_network['perturbable_vars'],\
        self.target_is_training = self.create_network(state_dim, action_dim, 'target_actor_network')

        # create perturbed actor network
        self.perturbed_actor_network = {}
        self.perturbed_state_input, \
        self.perturbed_action_output, \
        self.perturbed_actor_network['vars'], \
        self.perturbed_actor_network['trainable_vars'], \
        self.perturbed_actor_network['perturbable_vars'],\
        self.perturbed_is_training = self.create_network(state_dim, action_dim, 'perturbed_actor_network')

        # define training rules
        self.create_training_method()
        self.setup_target_network_updates()
        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.target_init_updates)
        self.param_noise_stddev = self.setup_param_noise()

        # self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.actor_network['trainable_vars'], -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.parameters_gradients, self.actor_network['trainable_vars']))

    def create_network(self, state_dim, action_dim, name):
        layer1_size = self.layer1_size
        layer2_size = self.layer1_size

        with tf.variable_scope(name) as scope:
            state_input = tf.placeholder("float", [None, state_dim])
            is_training = tf.placeholder(tf.bool, None)

            W1 = self.variable([state_dim, layer1_size], state_dim)
            b1 = self.variable([layer1_size], state_dim)
            W2 = self.variable([layer1_size, layer2_size], layer1_size)
            b2 = self.variable([layer2_size], layer1_size)
            W3 = tf.Variable(tf.random_uniform([layer2_size, action_dim], -3e-3, 3e-3))
            b3 = tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3))
            if self.is_layer_norm == True:
                layer1 = tf.matmul(state_input, W1) + b1
                layer1_norm = tc.layers.layer_norm(layer1, center=True, scale=True)#, activation_fn=tf.nn.relu)
                #layer1_norm = tf.nn.relu(layer1_norm)
                layer1_norm = self.activation_fn(layer1_norm)
                layer2 = tf.matmul(layer1_norm, W2) + b2
                layer2_norm = tc.layers.layer_norm(layer2, center=True, scale=True)#, activation_fn=tf.nn.relu)
                #layer2_norm = tf.nn.relu(layer2_norm)
                layer2_norm = self.activation_fn(layer2_norm)
                action_output = tf.identity(tf.matmul(layer2_norm, W3) + b3)
            else:
                layer1 = tf.matmul(state_input, W1) + b1
                layer1 = self.activation_fn(layer1)#tf.nn.relu(layer1)
                layer2 = tf.matmul(layer1, W2) + b2
                layer2 = self.activation_fn(layer2)#tf.nn.relu(layer2)
                action_output = tf.identity(tf.matmul(layer2, W3) + b3)

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        perturbable_vars = [var for var in trainable_vars if 'LayerNorm' not in var.name]

        return state_input, action_output, vars, trainable_vars, perturbable_vars, is_training

    def update_target(self):
        self.sess.run(self.target_soft_updates)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch,
            self.is_training: True
        })

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch,
            self.is_training: False
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state],
            self.is_training: False
        })[0]

    def action_noise(self, state):
        return self.sess.run(self.perturbed_action_output, feed_dict={
            self.perturbed_state_input: [state],
            self.perturbed_is_training: False
        })[0]

    def actions_target(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_is_training: False
        })

    def get_target_updates(self, vars, target_vars, tau):
        soft_updates = []
        init_updates = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            init_updates.append(tf.assign(target_var, var))
            soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
        assert len(init_updates) == len(vars)
        assert len(soft_updates) == len(vars)
        return tf.group(*init_updates), tf.group(*soft_updates)

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = self.get_target_updates(self.actor_network['vars'], self.target_actor_network['vars'], self.tau)
        self.target_init_updates = actor_init_updates
        self.target_soft_updates = actor_soft_updates

    def get_perturbed_actor_updates(self, actor, perturbed_actor, param_noise_stddev):
        assert len(actor['vars']) == len(perturbed_actor['vars'])
        assert len(actor['perturbable_vars']) == len(perturbed_actor['perturbable_vars'])

        updates = []
        for var, perturbed_var in zip(actor['vars'], perturbed_actor['vars']):
            if var in actor['perturbable_vars']:
                updates.append(
                    tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
            else:
                updates.append(tf.assign(perturbed_var, var))
        assert len(updates) == len(actor['vars'])
        return tf.group(*updates)

    def setup_param_noise(self):
        param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        # Configure perturbed actor.
        perturbed_actor_init_updates, perturbed_actor_soft_updates = self.get_target_updates(self.actor_network['vars'], self.perturbed_actor_network['vars'], self.tau)
        self.sess.run(perturbed_actor_init_updates)

        self.perturb_policy_ops = self.get_perturbed_actor_updates(self.actor_network, self.perturbed_actor_network, param_noise_stddev)
        self.sess.run(self.perturb_policy_ops, feed_dict={
            param_noise_stddev: self.param_noise.current_stddev,
        })

        # Configure separate copy for stddev adoption.
        self.perturb_adaptive_policy_ops = self.get_perturbed_actor_updates(self.actor_network, self.perturbed_actor_network, param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.action_output - self.perturbed_action_output)))
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            param_noise_stddev: self.param_noise.current_stddev,
        })
        return param_noise_stddev

    def adapt_param_noise(self, state_batch):
        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        # measure the distance
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.state_input: state_batch,
            self.perturbed_state_input: state_batch,
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        #mean_distance = mpi_mean(distance)
        self.param_noise.adapt(distance)
        return distance

    def perturb_policy(self):
        self.sess.run(self.perturb_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.cond(training_phase,
                       lambda: tc.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope_bn, decay=0.9, epsilon=1e-5),
                       lambda: tc.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope_bn, decay=0.9, epsilon=1e-5))

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
        #return tf.Variable(tf.random_normal(shape))

    def add_layer(inputs, in_size, out_size, normalization=None, activation_fn=None, dropout=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        #biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        biases = tf.Variable(tf.random_normal([1, out_size]))
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_fn is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_fn(Wx_plus_b)

        return outputs

    def load_network(self, dir_path):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(dir_path + '/saved_actor_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:" + checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, dir_path):
        print('save actor-network...' + str(time_step))
        self.saver.save(self.sess, dir_path+'/saved_actor_networks/' + 'actor-network')  # , global_step = time_step)
