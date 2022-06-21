import os 
import numpy as np
from envs.traffic_junction_env_add_collide_reward import TrafficJunctionEnv
from alg_parameters import *
from model import DGN
from buffer import ReplayBuffer
from config import *
from utilis import *
import torch
import torch.optim as optim
import setproctitle
import os.path as osp
import tensorflow as tf

env = TrafficJunctionEnv()
n_ant = EnvParameters.N_AGENTS
n_actions = EnvParameters.N_ACTIONS
obs_space = EnvParameters.OBS_LEN 

buff = ReplayBuffer(TrainingParameters.capacity,obs_space,n_actions,n_ant)
model = DGN(n_ant,obs_space,TrainingParameters.hidden_dim,n_actions)
model_tar = DGN(n_ant,obs_space,TrainingParameters.hidden_dim,n_actions)
model = model.cuda()
model_tar = model_tar.cuda()
model_tar.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), TrainingParameters.learning_rates)

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self):
        return self.action_space.sample()

agent = RandomAgent(env.action_space)

# if RecordingParameters.WANDB:

#         wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
#                    name=RecordingParameters.EXPERIMENT_NAME,
#                    entity=RecordingParameters.ENTITY,
#                    notes=RecordingParameters.EXPERIMENT_NOTE,
#                    config=key_algArgs)
#         print('Launching wandb...\n')

if RecordingParameters.TENSORBORAD:

        if not os.path.exists(RecordingParameters.SUMMERAY_PATH):
            os.makedirs(RecordingParameters.SUMMERAY_PATH)
        global_summary = tf.summary.FileWriter(RecordingParameters.SUMMERAY_PATH)
        print('Launching tensorboard...\n')
        if RecordingParameters.TXT_WRITTER:
            txt_path = RecordingParameters.SUMMERAY_PATH + '/' + RecordingParameters.TXT_NAME
            with open(txt_path, "w") as f:
                f.write(str(algArgs))
            print('Logging txt...\n')

last_test_t = -SetupParameters.EVALUE_INTERVAL - 1
last_model = -RecordingParameters.SAVE_INTERVAL - 1

list_a = []
list_b = []
infos = []

setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)

while TrainingParameters.i_episode<TrainingParameters.n_episode:
	#infos = []
	if TrainingParameters.time_step > TrainingParameters.max_timestep:
		break

	if TrainingParameters.i_episode > 100:
		TrainingParameters.epsilon -= 0.001
		if TrainingParameters.epsilon < 0.02:
			TrainingParameters.epsilon = 0.02

	TrainingParameters.i_episode+=1
	obs = env.reset()
	terminated = False
	adj = get_adjacency_matrix(obs)
	adj=adj+np.eye(n_ant)
	while not terminated:

		# test_flag += 1
		TrainingParameters.time_step += 1

		action=[]
		q = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([adj])).cuda())[0]

		for i in range(n_ant):
			if np.random.rand() < TrainingParameters.epsilon:
				a = agent.act()  
			else:
				a = np.argmax(q[i].cpu().detach().numpy()) 

			action.append(a)		

		next_obs, reward, terminated, info = env.step(action)

		next_adj = get_adjacency_matrix(next_obs)
		next_adj=next_adj+np.eye(n_ant)

		buff.add(np.array(obs),action,reward,np.array(next_obs),adj,next_adj,terminated) # mask, 
		obs = next_obs
		adj = next_adj

	infos.append(info)

	if TrainingParameters.i_episode < 100:

		continue

	mb_loss = []

	for epoch in range(TrainingParameters.N_EPOCHS):

		# next mask before D was also there

		O,A,R,Next_O,Matrix,Next_Matrix,D = buff.getBatch(TrainingParameters.batch_size)

		q_values = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
		target_q_values = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda())
		# target_q_values = (target_q_values - 9e15*(1 - torch.Tensor(Next_Mask).cuda())).max(dim = 2)[0]
		target_q_values = np.array(target_q_values.cpu().data)

		expected_q = np.array(q_values.cpu().data)
		for j in range(TrainingParameters.batch_size):

			for i in range(n_ant):
				expected_q[j][i][A[j][i]] = (R[j][i]) + (1-D[j])*TrainingParameters.GAMMA*np.max(target_q_values[j][i])#[A[j][i]]

		loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
		mb_loss.append(loss.data)  
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# if(epoch % 10 == 0):
		# 	print(loss)

		with torch.no_grad():

			for p, p_targ in zip(model.parameters(), model_tar.parameters()):
				# p.data.mul_(TrainingParameters.tau)
				# p_targ.data.add_((1 - TrainingParameters.tau) * p_targ.data)
				p_targ.data.mul_(TrainingParameters.tau)
				p_targ.data.add_((1 - TrainingParameters.tau) * p.data)

		# TrainingParameters.time_step+=1#TrainingParameters.N_STEPS

        # Recording

	if(TrainingParameters.time_step % TrainingParameters.N_STEPS == 0):	

			performance_r=np.nanmean([item['episode_reward'] for item in infos])
			num_collide=np.nanmean([item['number_collide'] for item in infos])
			win_rate=np.nanmean([item['success'] for item in infos])

			if RecordingParameters.TENSORBORAD:
				write_to_tensorboard(global_summary, TrainingParameters.time_step,performance_r, num_collide,win_rate,mb_loss,evalue=False)
			
			infos = []
			h = 'eval reward : '+str(performance_r)+' Eval collide : '+str(num_collide)+' Eval win : '+str(win_rate)
			print(h)
			a = np.array([performance_r, num_collide, win_rate])
			list_a.append(a)
			a = np.array(list_a)
			np.save('TA_ER_NC_WR.npz',a, allow_pickle=True)

			if (TrainingParameters.time_step - last_test_t) / SetupParameters.EVALUE_INTERVAL >= 1.0:

					# Evaluate Model
					last_test_t = TrainingParameters.time_step
					eval_reward, eval_collide,eval_win_rate = evaluate(env, model_tar, EnvParameters.N_AGENTS)
					eval_reward = np.nanmean([item for item in eval_reward])
					eval_collide = np.nanmean([item for item in eval_collide])
					eval_win_rate = np.nanmean([item for item in eval_win_rate])

					if RecordingParameters.TENSORBORAD:
						write_to_tensorboard(global_summary, TrainingParameters.time_step, eval_reward, eval_collide, eval_win_rate, evalue=True)

					print('episodes: {}, step: {},episode reward: {}, number of collide: {},win rate: {}'.format(TrainingParameters.i_episode,
					 															TrainingParameters.time_step, eval_reward, eval_collide,
					 															eval_win_rate))

					b = np.array([eval_reward, eval_collide, eval_win_rate])
					list_b.append(b)
					b = np.array(list_b)
					np.save('EA_ER_NC_WR.npz',b, allow_pickle=True)

			if (TrainingParameters.time_step - last_model) / RecordingParameters.SAVE_INTERVAL >= 1.0:

					last_model = TrainingParameters.time_step
					print('Saving Model !', end='\n')
					model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % TrainingParameters.time_step)
					os.makedirs(model_path)
					tar_model_checkpoint = model_path + "/target_model_checkpoint.pkl"
					model_checkpoint = model_path + "/model_checkpoint.pkl"
					torch.save(model_tar.state_dict(), tar_model_checkpoint)
					torch.save(model.state_dict(), model_checkpoint)

print('Saving Final Model !', end='\n')
model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % TrainingParameters.time_step)
os.makedirs(model_path)
tar_model_checkpoint = model_path + "/target_model_checkpoint.pkl"
model_checkpoint = model_path + "/model_checkpoint.pkl"
torch.save(model_tar.state_dict(), tar_model_checkpoint)
torch.save(model.state_dict(), model_checkpoint)
