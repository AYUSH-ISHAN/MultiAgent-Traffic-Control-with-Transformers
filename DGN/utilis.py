import numpy as np
import torch
from alg_parameters import EnvParameters, SetupParameters
from config import *
import tensorflow as tf
import torch
from alg_parameters import *

n_ant = EnvParameters.N_AGENTS

import numpy as np
import torch
from alg_parameters import EnvParameters, SetupParameters
from config import *
import tensorflow as tf
import torch
import wandb
from alg_parameters import *

n_ant = EnvParameters.N_AGENTS

def evaluate(test_env, model, n_ant):

	test_r=0
	eval_reward = []
	eval_collide = []
	eval_win_rate = []
	for _ in range(SetupParameters.EVALUE_EPISODES):
		test_obs = test_env.reset()
		# test_adj = test_env.get_visibility_matrix()[:,0:n_ant]*1 + np.eye(n_ant)
		t_adj = get_adjacency_matrix(test_obs)
		test_adj = t_adj+np.eye(n_ant)
		# test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
		terminated = False
		while not terminated:
			action=[]
			q = model(torch.Tensor(np.array([test_obs])).cuda(), torch.Tensor(np.array([test_adj])).cuda())[0]
			for i in range(n_ant):
				a = np.argmax(q[i].cpu().detach().numpy()) #- 9e15*(1 - test_mask[i]))
				action.append(a)
			test_obs, reward, terminated, info = test_env.step(action)
			test_r += reward
			if terminated:
				eval_reward.append(info['episode_reward'])
				eval_collide.append(info['number_collide'])
				eval_win_rate.append(info['success'])

			t_adj = get_adjacency_matrix(test_obs)
			test_adj=t_adj+np.eye(n_ant)
			
	return eval_reward,eval_collide, eval_win_rate

def get_adjacency_matrix(obs):
    adj = np.zeros((n_ant, n_ant))
    for agent in range(n_ant):
        for i in range(agent):  # already other half is marked below in index
                # print(agent, i)
                if(((obs[agent][2]-obs[i][2])**2 +(obs[agent][3]-obs[i][3])**2) < 0.1):
                    adj[agent][i] = 1
                    adj[i][agent]=1
    return adj

def write_to_tensorboard( global_summary, step, performance_r,num_collide,win_rate, mb_loss=None,evalue=True):
    if evalue:
        summary_eval = tf.Summary()

        summary_eval.value.add(tag='Perf_evaluate/Reward', simple_value=performance_r)
        summary_eval.value.add(tag='Perf_evaluate/Num_collide', simple_value=num_collide)
        summary_eval.value.add(tag='Perf_evaluate/Win_rate', simple_value= win_rate)
        global_summary.add_summary(summary_eval, step)
        global_summary.flush()

    else:
        summary_e = tf.Summary()

        summary_e.value.add(tag='Perf/Reward', simple_value=performance_r)
        summary_e.value.add(tag='Perf/Num_collide', simple_value=num_collide)
        summary_e.value.add(tag='Perf/Win_rate', simple_value=win_rate)
        global_summary.add_summary(summary_e, step)
        global_summary.flush()
