import os
import pickle
from logger import logger
from ddpg import *
from PD_controller import *
import gc
import math
import numpy as np
from datetime import datetime
from Interpolate import *
from configuration import *
from valkyrie_gym_env import Valkyrie

gc.enable()



def main():
    config = Configuration()
    ENV_NAME = config.conf['env-id'] # 'HumanoidBalanceFilter-v0'#'HumanoidBalance-v0'
    EPISODES = config.conf['epoch-num']
    TEST = config.conf['test-num']
    step_lim =config.conf['total-step-num']

    episode_count = config.conf['epoch-num']
    action_bounds = config.conf['action-bounds']

    PD_frequency = config.conf['LLC-frequency']
    network_frequency = config.conf['HLC-frequency']
    sampling_skip = int(PD_frequency/network_frequency)

    reward_decay=1.0
    reward_scale=0.05#Normalizing the scale of reward to 10#0.1#1.0/sampling_skip#scale down the reward
    max_steps = int(16*network_frequency)
    BEST_REWARD = 0

    env = Valkyrie(max_time=16, renders=False, initial_gap_time = 1)

    agent = DDPG(env,config)

    dir_path = 'record/' + datetime.now().strftime('%Y_%m_%d_%H.%M.%S') + '/no_force'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(dir_path+'/saved_actor_networks'):
        os.makedirs(dir_path+'/saved_actor_networks')
    if not os.path.exists(dir_path+'/saved_critic_networks'):
        os.makedirs(dir_path+'/saved_critic_networks')
    logging = logger(dir_path)
    config.save_configuration(dir_path)
    config.record_configuration(dir_path)
    config.print_configuration()
    agent.load_weight(dir_path)


    step_count=0
    #env.monitor.start('experiments/' + ENV_NAME,force=True)

    prev_action = np.zeros((4,))

    if config.conf['joint-interpolation'] == True:
        hip_interpolate = JointTrajectoryInterpolate()
        knee_interpolate = JointTrajectoryInterpolate()
        ankle_interpolate = JointTrajectoryInterpolate()
        waist_interpolate = JointTrajectoryInterpolate()

    for episode in range(EPISODES):
        state = env.reset()

        # Train
        action = np.zeros((4,))#4 dimension output of actor network, hip, knee, waist, ankle
        total_reward=0


        control_action = np.zeros((7,))#duplicate action for two legs
        next_state, reward, done, _ = env._step(control_action)
        #next_state = Valkyrie.getExtendedObservation()

        agent.reset()

        for step in range(max_steps):
            step_count+=1 #counting total steps during training

            prev_action = action
            #update action
            state = env.getExtendedObservation()
            if agent.config.conf['normalize-observations']:
                state_norm = agent.ob_normalize1.normalize(np.asarray(state))
                state_norm = np.reshape(state_norm, (agent.state_dim))  # reshape intp(?,)
            else:
                state_norm = state
            action = agent.action_noise(state_norm)
            action = np.clip(action,action_bounds[0],action_bounds[1])

            #print(action)
            #env.render()
            reward_add=0
            if config.conf['joint-interpolation'] == True:
                waist_interpolate.cubic_interpolation_setup(prev_action[0], 0, action[0], 0, 1.0 / float(network_frequency))
                hip_interpolate.cubic_interpolation_setup(prev_action[1], 0, action[1], 0, 1.0 / float(network_frequency))
                knee_interpolate.cubic_interpolation_setup(prev_action[2], 0, action[2], 0, 1.0 / float(network_frequency))
                ankle_interpolate.cubic_interpolation_setup(prev_action[3], 0, action[3], 0, 1.0 / float(network_frequency))

            for i in range(sampling_skip):
                if config.conf['joint-interpolation'] == True:
                    action = [  waist_interpolate.interpolate(1.0 / PD_frequency),\
                                hip_interpolate.interpolate(1.0 / PD_frequency), \
                                knee_interpolate.interpolate(1.0 / PD_frequency), \
                                ankle_interpolate.interpolate(1.0 / PD_frequency)]

                #env.render()
                control_action[0:4] = action
                control_action[4:7] = action[1:4]#duplicate leg control signals
                next_state, reward, done, _ = env._step(control_action)
                reward_add=reward+reward_decay*reward_add

            reward=reward_add*reward_scale#/sampling_skip
            agent.perceive(state,action,reward,next_state,done)

            if done:
                break

        if episode % 10 == 0 and episode > 1:
            total_reward = 0
            for i in range(TEST):
                _ = env._reset()

                action = np.zeros((4,))
                control_action = np.zeros((7,))  # duplicate action for two legs
                state, reward, done, _ = env._step(control_action)


                for j in range(max_steps):
                    prev_action = action

                    state = env.getExtendedObservation()
                    if agent.config.conf['normalize-observations']:
                        state_norm = agent.ob_normalize1.normalize(np.asarray(state))
                        state_norm = np.reshape(state_norm, (agent.state_dim))  # reshape intp(?,)
                    else:
                        state_norm = state
                    action = agent.action(state_norm) # direct action for test
                    action = np.clip(action, action_bounds[0], action_bounds[1])

                    reward_add = 0
                    if config.conf['joint-interpolation'] == True:
                        waist_interpolate.cubic_interpolation_setup(prev_action[0], 0, action[0], 0, 1.0 / float(network_frequency))
                        hip_interpolate.cubic_interpolation_setup(prev_action[1], 0, action[1], 0, 1.0 / float(network_frequency))
                        knee_interpolate.cubic_interpolation_setup(prev_action[2], 0, action[2], 0, 1.0 / float(network_frequency))
                        ankle_interpolate.cubic_interpolation_setup(prev_action[3], 0, action[3], 0, 1.0 / float(network_frequency))

                    #env.render()
                    for i in range(sampling_skip):
                        #if(sampling_skip%10==0):
                            #env.render()
                        if config.conf['joint-interpolation'] == True:
                            action = [  waist_interpolate.interpolate(1.0 / PD_frequency), \
                                        hip_interpolate.interpolate(1.0 / PD_frequency), \
                                        knee_interpolate.interpolate(1.0 / PD_frequency), \
                                        ankle_interpolate.interpolate(1.0 / PD_frequency)]

                        control_action[0:4] = action
                        control_action[4:7] = action[1:4]  # duplicate leg control signals
                        _, reward, done, _ = env._step(control_action)
                        reward_add = reward+reward_decay*reward_add

                    reward = reward_add*reward_scale# / sampling_skip
                    total_reward += reward
                    if done:
                        break

            ave_reward = total_reward/TEST
            if BEST_REWARD<ave_reward and episode>100: #save training data
                BEST_REWARD=ave_reward
                agent.save_weight(step, dir_path)
            print('episode:'+str(episode)+' step:'+str(step_count)+' Evaluation Average Reward:'+str(ave_reward))
            logging.add_train(episode, step_count, ave_reward)
            logging.save_train()
            agent.ob_normalize1.save_normalization(dir_path)#TODO test observation normalization
            agent.ob_normalize2.save_normalization(dir_path)  # TODO test observation normalization
        if step_count>step_lim:
            break
    #agent.save_weight(step, dir_path)
    logging.save_train()
    agent.save_memory("replay_buffer.txt")
if __name__ == '__main__':
    main()
