import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import numpy as np

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone().cpu().data
        # print(masked_q_values.get_device())
        # masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long().cpu().data
        # random_actions = Categorical(avail_actions.float()).sample().long()
        random_actions=[]
        for j in range(agent_inputs.shape[0]):
            for i in range(10):  # number of agents = 10
                a = np.random.randint(0,2)#agent.act()  # change it to simple random sampling
                random_actions.append(a)
        
        # print(pick_random.get_device())
        # print(th.tensor(random_actions).get_device())
        picked_actions = th.tensor(random_actions) + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        # previously it was     pick_random * random_actions
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
