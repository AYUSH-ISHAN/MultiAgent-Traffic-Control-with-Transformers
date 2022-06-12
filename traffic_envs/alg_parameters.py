import datetime

""" Create an attention communication algorithm.

Parameters:
   actor_lr :actor network learning rate.
   critic_lr :critic network learning rate.
   GAMMA :discount factor for reward.
   LAM  : discount factor for advantage.
   CLIP_RANGE : clip range for ratio.
   MAX_GRAD_NORM : clip gradient.
   ENTROPY_COEF : entropy discount factor.
   VALUE_COEF : value loss discount factor.
   POLICY_COEF : policy loss discount factor.
   VALID_COEF :  valid action loss discount factor.
   N_STEPS : each environment act step.
   N_MINIBATCHES : number of minibatch in one train batch.
   N_EPOCHS : number of reuse experience.
   N_ENVS : Number of environment copies being run in parallel.
   N_MAX_STEPS : max number of total step .
   N_UPDATES: number of update .
   BATCH_SIZE : size of each running.
   MINIBATCH_SIZE: size of each minibatch.
   SEED : random seed.
   SAVE_INTERVAL :save Model interval.
   EVALUE_INTERVAL :evaluate Model interval.
   EVALUE_EPISODES: number of evaluation episode.
 """


class EnvParameters:
    N_AGENTS = 10
    N_ACTIONS= 2
    EPISODE_LEN = 50
    OBS_LEN = 22
    ENV_SEED=1
    DIM=14
    VISION=1
    ADD_RATE_MIN=1
    ADD_RATE_MAX = 1
    CURR_START=0
    CURR_END = 0
    REACH_REWARD= 20
    if CURR_START==CURR_END:
        CURRICULUM=False
    else:
        CURRICULUM=True
    DIFFICULTY='medium'  # easy|medium|hard
    VOCAB_TYPE='scalar'  # bool|scalar

class TrainingParameters:
    actor_lr = 5e-5  # lambda f: f * 6e-4
    critic_lr = 5e-5  # lambda f: f * 6e-4
    # GAMMA = 1.0
    LAM = 0.95
    CLIP_RANGE = 0.2
    ACTOR_MAX_GRAD_NORM = 20
    CRITIC_MAX_GRAD_NORM = 50
    ENTROPY_COEF = 0.01
    VALUE_COEF = 1
    POLICY_COEF = 1
    N_EPOCHS = 10
    N_ENVS = 16
    N_MAX_STEPS = 2e7
    N_STEPS = 2 ** 9  # 2 ** 9
    MINIBATCH_SIZE = int(2 ** 9)
    DECAY_STEP = int(N_MAX_STEPS * N_EPOCHS // MINIBATCH_SIZE // 2 ** 10)
    DECAY_GAMMA = 0.99

    hidden_dim = 64#128#64
    GAMMA = 0.96  # previously this was commented and above was working
    time_step = 0
    max_timestep = 2e7 #500
    n_episode = 1000000
    i_episode = 0
    capacity = 65000#100000
    batch_size = int(2 ** 9) #64
    # n_epoch = 10
    epsilon = 0.6
    tau = 0.96
    test_flag = 0
    comm_flag = 1
    range_=3
    learning_rates = 0.0001
    adjacency_thresh = 0.6

class NetParameters:
    ACTOR_LAYER1 = 2 ** 7
    ACTOR_LAYER2 = 2 ** 6
    ACTOR_LAYER3 = 2
    CRITIC_LAYER1 = 2 ** 7
    CRITIC_LAYER2 = 2 ** 6
    CRITIC_LAYER3 = 1
    ACTOR_INPUT_LEN = EnvParameters.OBS_LEN
    CRITIC_INPUT_LEN = EnvParameters.OBS_LEN


class SetupParameters:
    SEED = 1234
    EVALUE_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS *2
    EVALUE_EPISODES = 16
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    WANDB = False
    TENSORBORAD = True
    TXT_WRITTER = True
    ENTITY = 'ayush_ishan'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'DGN'
    EXPERIMENT_NAME = 'version_42'
    EXPERIMENT_NOTE = 'version_42'
    SAVE_INTERVAL = 1e6 * 2
    BEST_INTERVAL = 1e6
    RECORD_BEST=False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    SUMMERAY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    TXT_NAME = 'alg.txt'
    LOSS_NAME = ['batch_loss']

key_algArgs = {'GAMMA' : TrainingParameters.GAMMA, 'EPSILON' : TrainingParameters.epsilon,
            'TAU' : TrainingParameters.tau, 'N_STEPS': TrainingParameters.N_STEPS,
            'CAPACITY' : TrainingParameters.capacity, 
            'BATCH_SIZE': TrainingParameters.batch_size,
            'DIFFICULTY': EnvParameters.DIFFICULTY,
            'LEARNING_RATE': TrainingParameters.learning_rates,
            'CURR_START': EnvParameters.CURR_START, 'CURR_END': EnvParameters.CURR_END,
            'CURRICULUM': EnvParameters.CURRICULUM, 
            }

algArgs = {'GAMMA' : TrainingParameters.GAMMA, 'EPSILON' : TrainingParameters.epsilon,
            'TAU' : TrainingParameters.tau, 'N_STEPS': TrainingParameters.N_STEPS,
            'CAPACITY' : TrainingParameters.capacity, 
            'ADJACENCY_THRESH': TrainingParameters.adjacency_thresh,
            'BATCH_SIZE': TrainingParameters.batch_size,
            'DIFFICULTY': EnvParameters.DIFFICULTY,
            'LEARNING_RATE': TrainingParameters.learning_rates,
            'CURR_START': EnvParameters.CURR_START, 'CURR_END': EnvParameters.CURR_END,
            'CURRICULUM': EnvParameters.CURRICULUM,
            'EVALUE_EPISODES': SetupParameters.EVALUE_EPISODES,
           'SAVE_INTERVAL': RecordingParameters.SAVE_INTERVAL, "BEST_INTERVAL": RecordingParameters.BEST_INTERVAL,
           'EXPERIMENT_PROJECT': RecordingParameters.EXPERIMENT_PROJECT,
           'EXPERIMENT_NAME': RecordingParameters.EXPERIMENT_NAME,
           'EXPERIMENT_NOTE': RecordingParameters.EXPERIMENT_NOTE,'N_AGENTS':EnvParameters.N_AGENTS,
           'EPISODE_LEN':EnvParameters.EPISODE_LEN,'DIM':EnvParameters.DIM,'VISION':EnvParameters.VISION,
        }

# algArgs = {'actor_lr': TrainingParameters.actor_lr, 'critic_lr': TrainingParameters.critic_lr,
#            'GAMMA': TrainingParameters.GAMMA, 'LAM': TrainingParameters.LAM,
#            'CLIPRANGE': TrainingParameters.CLIP_RANGE,
#            'ACTOR_MAX_GRAD_NORM': TrainingParameters.ACTOR_MAX_GRAD_NORM,
#             'CRITIC_MAX_GRAD_NORM': TrainingParameters.CRITIC_MAX_GRAD_NORM,
#            'ENTROPY_COEF': TrainingParameters.ENTROPY_COEF,
#            'VALUE_COEF': TrainingParameters.VALUE_COEF,
#            'POLICY_COEF': TrainingParameters.POLICY_COEF,
#            'N_EPOCHS': TrainingParameters.N_EPOCHS, 'N_ENVS': TrainingParameters.N_ENVS,
#            'N_STEPS': TrainingParameters.N_STEPS,
#            'MINIBATCH_SIZE': TrainingParameters.MINIBATCH_SIZE,
#            'N_MAX_STEPS': TrainingParameters.N_MAX_STEPS, "ENV_SEED":EnvParameters.ENV_SEED,
#            'ACTOR_LAYER1': NetParameters.ACTOR_LAYER1, 'ACTOR_LAYER2': NetParameters.ACTOR_LAYER2,
#            'ACTOR_LAYER3': NetParameters.ACTOR_LAYER3, 'CRITIC_LAYER1': NetParameters.CRITIC_LAYER1,
#            'CRITIC_LAYER2': NetParameters.CRITIC_LAYER2,
#            'CRITIC_LAYER3': NetParameters.CRITIC_LAYER3, 'ACTOR_INPUT_LEN': NetParameters.ACTOR_INPUT_LEN,
#            'CRITIC_INPUT_LEN': NetParameters.CRITIC_INPUT_LEN,
#            'SEED': SetupParameters.SEED, 'EVALUE_INTERVAL': SetupParameters.EVALUE_INTERVAL,
#            'EVALUE_EPISODES': SetupParameters.EVALUE_EPISODES,
#            'SAVE_INTERVAL': RecordingParameters.SAVE_INTERVAL, "BEST_INTERVAL": RecordingParameters.BEST_INTERVAL,
#            'EXPERIMENT_PROJECT': RecordingParameters.EXPERIMENT_PROJECT,
#            'EXPERIMENT_NAME': RecordingParameters.EXPERIMENT_NAME,
#            'EXPERIMENT_NOTE': RecordingParameters.EXPERIMENT_NOTE,'N_AGENTS':EnvParameters.N_AGENTS,
#            'EPISODE_LEN':EnvParameters.EPISODE_LEN,'DIM':EnvParameters.DIM,'VISION':EnvParameters.VISION,
#            'ADD_RATE_MIN':EnvParameters.ADD_RATE_MIN,'ADD_RATE_MAX':EnvParameters.ADD_RATE_MAX,
#            'CURR_START':EnvParameters.CURR_START,'CURR_END':EnvParameters.CURR_END,
#            'CURRICULUM':EnvParameters.CURRICULUM,'DIFFICULTY':EnvParameters.DIFFICULTY,
#            'VOCAB_TYPE':EnvParameters.VOCAB_TYPE}


# key_algArgs = {'actor_lr': TrainingParameters.actor_lr, 'critic_lr': TrainingParameters.critic_lr,
#                'GAMMA': TrainingParameters.GAMMA, 'LAM': TrainingParameters.LAM,
#                'CLIPRANGE': TrainingParameters.CLIP_RANGE,
#                'ACTOR_MAX_GRAD_NORM': TrainingParameters.ACTOR_MAX_GRAD_NORM,
#                'CRITIC_MAX_GRAD_NORM': TrainingParameters.CRITIC_MAX_GRAD_NORM,
#                'ENTROPY_COEF': TrainingParameters.ENTROPY_COEF,
#                'VALUE_COEF': TrainingParameters.VALUE_COEF,
#                'POLICY_COEF': TrainingParameters.POLICY_COEF,
#                'N_EPOCHS': TrainingParameters.N_EPOCHS, 'N_ENVS': TrainingParameters.N_ENVS,
#                'N_STEPS': TrainingParameters.N_STEPS,
#                'MINIBATCH_SIZE': TrainingParameters.MINIBATCH_SIZE,
#                'ACTOR_LAYER1': NetParameters.ACTOR_LAYER1, 'ACTOR_LAYER2': NetParameters.ACTOR_LAYER2,
#                'ACTOR_LAYER3': NetParameters.ACTOR_LAYER3, 'CRITIC_LAYER1': NetParameters.CRITIC_LAYER1,
#                'CRITIC_LAYER2': NetParameters.CRITIC_LAYER2,
#                'CRITIC_LAYER3': NetParameters.CRITIC_LAYER3, 'ACTOR_INPUT_LEN': NetParameters.ACTOR_INPUT_LEN,
#                'CRITIC_INPUT_LEN': NetParameters.CRITIC_INPUT_LEN, "N_AGENTS": EnvParameters.N_AGENTS,
#                'EPISODE_LEN': EnvParameters.EPISODE_LEN, 'DIM': EnvParameters.DIM, 'VISION': EnvParameters.VISION,
#                'ADD_RATE_MIN': EnvParameters.ADD_RATE_MIN, 'ADD_RATE_MAX': EnvParameters.ADD_RATE_MAX,
#                'CURR_START': EnvParameters.CURR_START, 'CURR_END': EnvParameters.CURR_END,
#                'CURRICULUM': EnvParameters.CURRICULUM, 'DIFFICULTY': EnvParameters.DIFFICULTY,
#                'VOCAB_TYPE': EnvParameters.VOCAB_TYPE
#                }
