import random
# import numpy as np
import numpy as np
from collections import deque
import os
from itertools import combinations, product
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class FourierBasis:
    def __init__(self, state_dim, action_dim, order, max_non_zero=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.order = order
        self.max_non_zero = min(max_non_zero, state_dim)
        self.coeff = self._build_coefficients()

    def get_learning_rates(self, alpha):
        # lrs = np.linalg.norm(np.vstack((self.coeff, self.coeff)), axis=1)  # aqui sin
        lrs = np.linalg.norm(self.coeff, axis=1)
        lrs[lrs == 0.] = 1.
        lrs = alpha / lrs
        return lrs

    def _build_coefficients(self):
        coeff = np.array(np.zeros(self.state_dim), dtype=np.float32)  # Bias
        for i in range(1, self.max_non_zero + 1):
            for indices in combinations(range(self.state_dim), i):
                for c in product(range(1, self.order + 1), repeat=i):
                    coef = np.zeros(self.state_dim, dtype=np.float32)
                    coef[list(indices)] = list(c)
                    coeff = np.vstack((coeff, coef))
        return coeff

    def get_features(self, state):
        state_ = state.flatten()
        x = np.cos(np.dot(np.pi * self.coeff, state_))
        return x

    def get_num_basis(self) -> int:
        # return len(self.coeff)*2  # aqui sin
        return len(self.coeff)


class TOSFB(object):
    def __init__(self,
                 intersection_id,
                 # intersection,
                 state_size=9,
                 action_size=8,
                 batch_size=32,
                 phase_list=[],
                 timing_list = [],
                 env=None
                 ):
        self.env = env
        self.iid = intersection_id
        # self.intersection = intersection

        # Phase Vars:
        self.n_phases = len(phase_list)
        self.start_phase = 0

        # Simulation Vars:
        self.action_time = 0
        self.real_time = 0  # Tempo real (10+action_time*2)
        self.times_skiped = 0
        self.obs = []
        self.reward = []
        self.phase = 0

        self.state_dim = state_size
        self.action_dim = action_size
        self.batch_size = batch_size
        self.learning_start = 0

        # self.learning_start = 4000
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.meta_test_start = 100
        self.meta_test_update_model_freq = 10
        self.meta_test_update_target_model_freq = 200

        self.gamma = 0.95  # discount rate
        self.alpha = 0.0005
        self.lr = self.alpha
        self.lamb = 0  # 0.9
        self.epsilon = 0.01
        self.epsilon_decay = 0.9995 #1
        self.min_epsilon = 0.01
        self.fourier_order = 9  # 7
        self.max_non_zero_fourier = 2
        basis = 'fourier'

        if basis == 'fourier':
            self.basis = FourierBasis(self.state_dim, self.action_dim, self.fourier_order,
                                      max_non_zero=self.max_non_zero_fourier)
            self.lr = self.basis.get_learning_rates(self.alpha)
        self.num_basis = self.basis.get_num_basis()

        self.et = {a: np.zeros(self.num_basis, dtype=np.float32) for a in range(self.action_dim)}
        self.theta = {a: np.zeros(self.num_basis, dtype=np.float32) for a in range(self.action_dim)}

        self.use_buffer = True
        self.buffer = deque(maxlen=1000)

        self.q_old = None
        self.action = None
        self.td_error = 0

        ### added
        self.step = 0

        self.phase_list = phase_list
        self.timing_list = timing_list

    def reset_traces(self):
        self.q_old = None
        for a in range(self.action_dim):
            self.et[a].fill(0.0)

    def choose_action(self, obs):
        features = self.get_features(obs)
        return self.act(features)

    def act(self, features):
        if np.random.rand() < self.epsilon - self.step * 0.0002:
            #self.reset_traces()
            return random.randrange(self.action_dim)
        else:
            q_values = [self.get_q_value(features, a) for a in range(self.action_dim)]
            return q_values.index(max(q_values))

    def sample(self):
        return random.randrange(self.action_dim)

    def get_q_value(self, features, action):
        return np.dot(self.theta[action], features)

    def get_features(self, state):
        return self.basis.get_features(state)

    def remember(self, state, action_phase, reward, next_state, done=False):
        action = self.phase_list.index(action_phase)
        ###
        self.step += 1
        if self.use_buffer:
            self.buffer.append((state, action, reward, next_state))
            minibatch = [(state, action, reward, next_state)] + random.sample(self.buffer, min((self.batch_size-1), len(self.buffer)))
            # minibatch = [(state, action, reward, next_state)] + random.sample(self.buffer, min(31, len(self.buffer)))
        else:
            minibatch = [(state, action, reward, next_state)]
        for sample in minibatch:
            state, action, reward, next_state = sample
            phi = self.get_features(state)
            next_phi = self.get_features(next_state)
            q = self.get_q_value(phi, action)
            if not done:
                """ q_values = [self.get_q_value(next_phi, a) for a in range(self.action_dim)]
                next_q = max(q_values) """
                next_q = self.get_q_value(next_phi, self.act(next_phi))
            else:
                next_q = 0.0
            td_error = reward + self.gamma * next_q - q
            self.td_error = td_error
            if self.q_old is None:
                self.q_old = q

            for a in range(self.action_dim):
                if a == action:
                    self.et[a] = self.lamb * self.gamma * self.et[a] + (
                                1 - self.lr * self.gamma * self.lamb * np.dot(self.et[a], phi)) * phi
                    self.theta[a] += self.lr * (td_error + q - self.q_old) * self.et[a] - self.lr * (
                                q - self.q_old) * phi
                else:
                    self.et[a] = self.lamb * self.gamma * self.et[a]
                    self.theta[a] += self.lr * (td_error + q - self.q_old) * self.et[a]

            self.q_old = next_q
            if done:
                self.reset_traces()

        self.epsilon = max(self.epsilon_decay * self.epsilon, self.min_epsilon)

    def load_model(self, dir="model/tosfb", e=0):
        name = "tosfb_agent_{}_{}.pickle".format(self.iid, e)
        model_name = os.path.join(dir, name)
        with open(model_name, 'rb') as f:
            self = pickle.load(f)

    def save_model(self, dir="model/tosfb", e=0):
        env = self.env
        self.env = None
        name = "tosfb_agent_{}_{}.pickle".format(self.iid, e)
        model_name = os.path.join(dir, name)
        with open(model_name, 'wb+') as f:
            pickle.dump(self, f)
        self.env = env