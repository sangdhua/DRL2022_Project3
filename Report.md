Project 3: Collaboration and Competition Report
This report summarizes the learning algorithm, and model architecture used to train two RL agents control rackets to bounce a ball over a net.

Goal: The goal of training is to allow the agent to receive an average reward (over 100 episodes) of at least +0.5.

Learning algorithm
This project uses Multi-Agent Deep Deterministic Policy Gradient (MADDPG). 
The paper titled Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments is used to multi-agent version of DDPG to train the agents.
https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf
Below is the conclusion of the paper.

We have proposed a multi-agent policy gradient algorithm where agents learn a centralized critic
based on the observations and actions of all agents. Empirically, our method outperforms traditional
RL algorithms on a variety of cooperative and competitive multi-agent environments. We can further
improve the performance of our method by training agents with an ensemble of policies, an approach
we believe to be generally applicable to any multi-agent algorithm.
One downside to our approach is that the input space of Q grows linearly (depending on what
information is contained in x) with the number of agents N. This could be remedied in practice by,
for example, having a modular Q function that only considers agents in a certain neighborhood of a
given agent. We leave this investigation to future work.
izing an ensemble of policies for each agent that leads to more robust multi-agent policies. — Multi-Agent Actor-Critic paper

It took over 3850 episodes to achieve the score.  



Model architecture
The Actor and Critic network have the following architectures.

---Actor---
Actor(
  (fc1): Linear(in_features=48, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
---Critic---
Critic(
  (fcs1): Linear(in_features=48, out_features=256, bias=True)
  (fc2): Linear(in_features=260, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
---Actor---
Actor(
  (fc1): Linear(in_features=48, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
---Critic---
Critic(
  (fcs1): Linear(in_features=48, out_features=256, bias=True)
  (fc2): Linear(in_features=260, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)

Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
N_EPISODES = 4000       # number of episodes
MAX_T = 2000            # maximum number of timesteps per episode

Results
Episode 1 - 100	Average Score: 0.00	Max Score: 0.000
Episode 101 - 200	Average Score: 0.00	Max Score: 0.100
Episode 201 - 300	Average Score: 0.01	Max Score: 0.200
Episode 301 - 400	Average Score: 0.01	Max Score: 0.100
Episode 401 - 500	Average Score: 0.03	Max Score: 0.200
Episode 501 - 600	Average Score: 0.08	Max Score: 0.500
Episode 601 - 700	Average Score: 0.09	Max Score: 0.300
Episode 701 - 800	Average Score: 0.14	Max Score: 0.500
Episode 801 - 900	Average Score: 0.14	Max Score: 0.500
Episode 901 - 1000	Average Score: 0.17	Max Score: 0.900
Episode 1001 - 1100	Average Score: 0.19	Max Score: 0.700
Episode 1101 - 1200	Average Score: 0.20	Max Score: 1.100
Episode 1201 - 1300	Average Score: 0.18	Max Score: 0.800
Episode 1301 - 1400	Average Score: 0.17	Max Score: 0.800
Episode 1401 - 1500	Average Score: 0.24	Max Score: 1.900
Episode 1501 - 1600	Average Score: 0.27	Max Score: 1.400
Episode 1601 - 1700	Average Score: 0.27	Max Score: 1.200
Episode 1701 - 1800	Average Score: 0.25	Max Score: 1.000
Episode 1801 - 1900	Average Score: 0.23	Max Score: 1.100
Episode 1901 - 2000	Average Score: 0.26	Max Score: 2.200
Episode 2001 - 2100	Average Score: 0.29	Max Score: 3.500
Episode 2101 - 2200	Average Score: 0.24	Max Score: 2.300
Episode 2201 - 2300	Average Score: 0.23	Max Score: 1.100
Episode 2301 - 2400	Average Score: 0.25	Max Score: 2.200
Episode 2401 - 2500	Average Score: 0.28	Max Score: 3.000
Episode 2501 - 2600	Average Score: 0.24	Max Score: 2.900
Episode 2601 - 2700	Average Score: 0.35	Max Score: 2.900
Episode 2701 - 2800	Average Score: 0.26	Max Score: 2.400
Episode 2801 - 2900	Average Score: 0.31	Max Score: 3.200
Episode 2901 - 3000	Average Score: 0.29	Max Score: 2.100
Episode 3001 - 3100	Average Score: 0.30	Max Score: 1.400
Episode 3101 - 3200	Average Score: 0.33	Max Score: 2.000
Episode 3201 - 3300	Average Score: 0.38	Max Score: 3.500
Episode 3301 - 3400	Average Score: 0.46	Max Score: 2.800
Episode 3401 - 3500	Average Score: 0.36	Max Score: 2.900
Episode 3501 - 3600	Average Score: 0.35	Max Score: 2.000
Episode 3601 - 3700	Average Score: 0.41	Max Score: 2.700
Episode 3701 - 3800	Average Score: 0.38	Max Score: 3.000
Episode 3850	Average Score: 0.51	Score: 1.600
Environment succesfully solved in 3850 episodes!	Average Score: 0.51

Elapsed Time: 84.12 mins.





![image.png](attachment:image.png)

To improve this project,  I would try the following things.

1) modify the hyper-paramaters and add batch normalization to see if I can speed the learning.  
2) Multi-Agent PPO (MAPPO)

 


