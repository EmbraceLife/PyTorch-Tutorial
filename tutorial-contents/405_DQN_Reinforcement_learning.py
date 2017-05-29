# python -m pdb tutorial-contents/405_DQN_Reinforcement_learning.py
# if previously trained 400 times for the first time training, then now
# set train_again to True, and only train 5 times

"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.1.11
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

###############################
## Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
train_again = True # False # first time: set false; train again, set true
################################
## get a game object and access its attributes
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

#################################
## build model class
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
		# random initialization on weights help performance
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

## build DQN class
class DQN(object):
    def __init__(self):

		# DQN has two Net objects
        self.eval_net, self.target_net = Net(), Net()
		# attr: count training steps
        self.learn_step_counter = 0      # for target updating
        self.memory_counter = 0          # for storing memory
		# initialize memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
		# set up optimizer box
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
		# set up loss box
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
		# prepare input to a variable adding 1 more dimension
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # use eval_net to predict which action to take
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
			# only 2 actions: 0 and 1
            action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
        else:   # randomly choose an action
            action = np.random.randint(0, N_ACTIONS)
        return action

	## store a transition of 4 states into memory
    def store_transition(self, s, a, r, s_):
		# concat horizontally into an array
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
		# self.memory is 2d (MEMORY_CAPACITY, len(s)*2+2)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # every 100 iterations, update target parameter with eval_net parameters
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # sample batch transitions
		# randomly take 32 indices from 2000 memories
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
		# take those 32 memories
        b_memory = self.memory[sample_index, :]
		# take only the first state s out of 4 states, make variable
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
		# only take the second state and make variable
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
		# only take the third state and make variable
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
		# take the last state and make Variable
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience
		# self.eval_net(b_s): return a variable (32,2)
		# gather(dim, index)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
		# get outcome from target_net(b_s_), detach from graph, don't backpropagate
        q_next = self.target_net(b_s_).detach()
		# from q_next to q_target: check torch.max
        q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

if train_again:
	dqn = torch.load("/Users/Natsume/Downloads/temp_folders/405/dqn.pkl")
	# dqn.memory_counter = 0
	log = torch.load("/Users/Natsume/Downloads/temp_folders/405/log.pkl")
	steps = log[0]
	previous_epochs = steps[-1]
	losses = log[1]
else:
	dqn = DQN()
	steps = []
	previous_epochs = 0
	losses = []
print('\nCollecting experience...')
# which func enable collecting experience?

# for the first time training, set it 400; when train again, set 5
for i_episode in range(previous_epochs, previous_epochs+5):
    s = env.reset()
    ep_r = 0
    loss = 0
    while True:
        env.render()
		# chosen an action 0 or 1
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward: less reward if pole lean on sides, more reward if pole remain in the middle range; less reward if cart on sides, more reward if cart stays in middle
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

		# store transitions and count
        dqn.store_transition(s, a, r, s_)


        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            loss = dqn.learn()

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2), '| loss: %.4f' % loss.data.numpy()[0])

                losses.append(loss)
                steps.append(i_episode)

        if done:
            break
        s = s_

    if i_episode == previous_epochs + 1:
        print(i_episode)

torch.save(dqn, "/Users/Natsume/Downloads/temp_folders/405/dqn.pkl")
torch.save((steps, losses), "/Users/Natsume/Downloads/temp_folders/405/log.pkl")
