import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple
import gym
from itertools import count

gamma = 0.99

env = gym.make('LunarLander-v2').unwrapped
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'state_'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.action_head = nn.Linear(128, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_pro = F.softmax(self.action_head(x), dim=1)  # dim=1对每一行进行softmax
        return action_pro


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO(object):
    clip = 0.1
    max_grad_norm = 0.5
    update_time = 10
    buffer_capacity = 2000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.buffer_counter = 0
        self.train_step = 0

        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=0.01)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)    # unsqueeze(0), 维度变化 [8] -> [1, 8] | torch.Size([1, 8])
        with torch.no_grad():
            action_prob = self.actor_net(state)                 # tensor([[0.2343, 0.2082, 0.2800, 0.2775]])
        dist = Categorical(action_prob)                         # Categorical(probs: torch.Size([1, 4]))
        action = dist.sample()                                  # tensor([2])
        return action.item(), action_prob[:, action.item()].item()  # 2, 0.2800

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)                      # tensor([-0.2023])
        return value.item()                                     # -0.2023

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.buffer_counter += 1

    def update(self, ):
        # for b in self.buffer:
        #     state = torch.tensor(b.state, dtype=torch.float)
        #     state_ = torch.tensor(b.state_, dtype=torch.float)
        #     action = torch.tensor(b.action, dtype=torch.long).view(-1, 1)
        #     reward = b.reward
        #     old_action_log_prob = torch.tensor(b.a_log_prob, dtype=torch.float).view(-1, 1)
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # state_ = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R       # discounted reward
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # updating
        for i in range(self.update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, drop_last=False):
                Gt_index = Gt[index].view(-1, 1)
                state_value = self.critic_net(state[index])
                delta = Gt_index - state_value
                advantage = delta.detach()
                action_prob = self.actor_net(state[index]).gather(1, action[index])     # new policy

                ratio = action_prob / old_action_log_prob[index]
                border1 = ratio*advantage
                border2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantage

                # update actor_net
                actor_loss = -torch.min(border1, border2).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)   # 梯度裁剪
                self.actor_optim.step()

                # update critic_net
                critic_loss = F.mse_loss(Gt_index, state_value)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                self.train_step += 1

        del self.buffer[:]

    def save_model(self):
        torch.save(self.actor_net.state_dict(),"actor.pth")
        torch.save(self.critic_net.state_dict(),"critic.pth")
        print("save model")

def main():
    ppo = PPO()
    total_num = 1e5
    for epoch in range(int(total_num)):
        score = 0
        state = env.reset()
        env.render()
        for t in count():
            action, action_prob = ppo.choose_action(state)
            state_, reward, done, info = env.step(action)
            transition = Transition(state, action, action_prob, reward, state_)
            env.render()
            ppo.store_transition(transition)
            state = state_
            score += reward

            if done or t>1000:
                print('epoch {}, score is:{:.2f}'.format(epoch, score))
                if len(ppo.buffer) >= ppo.batch_size:
                    ppo.update()
                    break
    PPO.save_model()

if __name__ == '__main__':
    main()
    env.close()
