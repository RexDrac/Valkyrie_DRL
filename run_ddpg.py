from ddpg import *
from gym import wrappers
import gc
import numpy as np
import time
from PD_controller import *
from logger import logger
from Interpolate import *
from valkyrie_gym_env import Valkyrie
gc.enable()


dir_path = 'record/2017_10_25_15.28.26/no_force'#'2017_05_29_18.23.49/with_force'
MONITOR_DIR = dir_path
def main():
	config = Configuration()
	config.load_configuration(dir_path)
	config.print_configuration()

	ENV_NAME = config.conf['env-id']  # 'HumanoidBalanceFilter-v0'#'HumanoidBalance-v0'
	EPISODES = config.conf['epoch-num']
	TEST = config.conf['test-num']
	step_lim = config.conf['total-step-num']

	episode_count = config.conf['epoch-num']
	action_bounds = config.conf['action-bounds']

	PD_frequency = config.conf['LLC-frequency']
	network_frequency = config.conf['HLC-frequency']
	sampling_skip = int(PD_frequency / network_frequency)

	reward_decay = 1.0
	reward_scale = 0.05  # Normalizing the scale of reward to 10#0.1#1.0/sampling_skip#scale down the reward
	max_steps = int(16 * network_frequency)
	BEST_REWARD = 0

	EPISODES = 1
	STEPS = 2500000

	force = 700
	impulse = 0.01
	force_chest = [0, 0]  # max(0,force_chest[1]-300*1.0 / EXPLORE)]
	force_pelvis = [0, 0]
	force_period = [5 * PD_frequency, (5 + 0.1) * PD_frequency]  # impulse / force * FPS

	env = Valkyrie(max_time=16, renders=True, initial_gap_time=1)
	agent = DDPG(env,config)
	agent.load_weight(dir_path)
	agent.ob_normalize1.load_normalization(dir_path)#TODO test observation normalization
	agent.ob_normalize1.print_normalization()#TODO test observation normalization
	agent.ob_normalize2.load_normalization(dir_path)  # TODO test observation normalization
	agent.ob_normalize2.print_normalization()  # TODO test observation normalization

	step_count = 0

	total_reward = 0
	force_chest = [0, 0]
	force_pelvis = [0, 0]

	logging=logger(dir_path)

	t_max = 0#timer
	t_min = 100
	t_total=[]
	prev_action = []

	if config.conf['joint-interpolation'] == True:
		hip_interpolate = JointTrajectoryInterpolate()
		knee_interpolate = JointTrajectoryInterpolate()
		ankle_interpolate = JointTrajectoryInterpolate()
		waist_interpolate = JointTrajectoryInterpolate()

	for i in range(TEST):
		step_count = 0
		total_reward = 0

		_ = agent.reset()

		action = np.zeros((4,))  # 4 dimension output of actor network, hip, knee, waist, ankle
		control_action = np.zeros((7,))  # duplicate action for two legs
		state, reward, done, _ = env._step(control_action)

		for j in range(max_steps):
			#print(env.COM_pos_local)
			#print(env.COM_pos)
			#print(env.linkCOMPos['rightAnklePitch'])
			step_count += 1  # counting total steps during training

			prev_action = action

			#t0 = time.time()
			#update action
			state = env.getExtendedObservation()
			if agent.config.conf['normalize-observations']:
				state_norm = agent.ob_normalize1.normalize(np.asarray(state))
				state_norm = np.reshape(state_norm, (agent.state_dim))  # reshape intp(?,)
			else:
				state_norm = state
			action = agent.action(state_norm)
			action = np.clip(action, action_bounds[0], action_bounds[1])

			#t1 = time.time()

			#total = t1 - t0
			#t_total.append(total)
			#prev_action = action
			reward_add = 0
			env.render()
			if config.conf['joint-interpolation'] == True:
				waist_interpolate.cubic_interpolation_setup(prev_action[0], 0, action[0], 0,1.0 / float(network_frequency))
				hip_interpolate.cubic_interpolation_setup(prev_action[1], 0, action[1], 0, 1.0 / float(network_frequency))
				knee_interpolate.cubic_interpolation_setup(prev_action[2], 0, action[2], 0, 1.0 / float(network_frequency))
				ankle_interpolate.cubic_interpolation_setup(prev_action[3], 0, action[3], 0, 1.0 / float(network_frequency))

			action_org = action

			for i in range(sampling_skip):
				step_count += 1
				if ((step_count >= force_period[0]) and (step_count < force_period[1])):
					force_chest = [0, 0]
					force_pelvis = [force, 0]
				else:
					force_chest = [0, 0]
					force_pelvis = [0, 0]
				if ((step_count >= force_period[0]-250) and (step_count < force_period[1]+250)):
					text = ''#''600N applied on pelvis for 0.1s'
				else:
					text = ''

				if config.conf['joint-interpolation'] == True:
					action = [waist_interpolate.interpolate(1.0 / PD_frequency), \
							  hip_interpolate.interpolate(1.0 / PD_frequency), \
							  knee_interpolate.interpolate(1.0 / PD_frequency), \
							  ankle_interpolate.interpolate(1.0 / PD_frequency)]

				# env.render()
				control_action[0:4] = action
				control_action[4:7] = action[1:4]  # duplicate leg control signals
				next_state, reward, done, _ = env._step(control_action, force_pelvis[0])
				reward_add = reward + reward_decay * reward_add

				logging.add_run('target_ankle_joint',action_org)
				logging.add_run('interpolated_target_ankle_joint',action)

				ob = env.getObservation()
				on_filtered = env.getFilteredObservation()
				for i in range(len(ob)):
					logging.add_run('observation'+str(i),ob[i])
					logging.add_run('filtered_observation' + str(i), on_filtered[i])

				readings = env.getReadings()
				for key, value in readings.items():
					logging.add_run(key,value)

			reward = reward_add*reward_scale  # / sampling_skip
			total_reward += reward
			if done:
				break
	ave_reward = total_reward / TEST
	logging.save_run()
	print(' Evaluation Average Reward:' + str(ave_reward))
	#t_min=min(t_total)
	#t_max=max(t_total)
	#t_avg=sum(t_total)/float(len(t_total))
	#(t_min,t_max,t_avg)
	#env.monitor.close()

if __name__ == '__main__':
	main()
