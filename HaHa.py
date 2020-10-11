import os
import gym
import shutil
import argparse
import numpy as np
from tqdm import  trange
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

class PPOMemory:
    def __init__(self, gamma, tau):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.tdlamret = []
        self.advants = []
        self.gamma = gamma
        self.tau = tau
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, s, a, r, v, lp):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.logprobs.append(lp)
        self.ptr += 1

    def finish_path(self, v):
        """
        https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py line 64
        """
        path_slice = np.arange(self.path_start_idx, self.ptr)

        rewards_np = np.array(self.rewards)[path_slice]
        values_np = np.array(self.values)[path_slice]
        values_np_added = np.append(values_np, v)

        # GAE
        gae = 0
        advants = []
        for t in reversed(range(len(rewards_np))):
            delta = rewards_np[t] + self.gamma * values_np_added[t+1] - values_np_added[t]
            gae = delta + self.gamma * self.tau * gae
            advants.insert(0, gae)

        self.advants.extend(advants)

        advants_np = np.array(advants)
        tdlamret_np = advants_np + values_np
        self.tdlamret.extend(tdlamret_np.tolist())

        self.path_start_idx = self.ptr

    def reset_storage(self):
        self.ptr, self.path_start_idx = 0, 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.tdlamret = []
        self.advants = []

    def get(self):
        # reset marker
        data = dict(states=self.states, actions=self.actions, logpas=self.logprobs,
                    rewards=self.rewards, values=self.values,
                    tdlamret=self.tdlamret, advants=self.advants)
        self.reset_storage()
        return data

    def __len__(self):
        return len(self.rewards)

class PPO():
    def __init__(self):
        super(PPO, self).__init__()
        self.seed = 66
        self.average_interval = 100
        self.gae_tau = 0.95
        self.gamma = 0.99
        self.max_episodes = 5000
        self.max_steps_per_episode = 300
        self.batch_size = 32
        self.clip_range = 0.2
        self.coef_entpen = 0.001
        self.coef_vf = 0.5
        self.memory_size = 2048
        self.optim_epochs = 4
        self.terminal_score = 230
        self.evn_name = "LunarLander-v2"
        self.lr = 0.002
        self.betas = [0.9,0.999]
        self.game = gym.make(self.evn_name)
        self.input_dim = self.game.observation_space.shape[0]
        self.output_dim = self.game.action_space.n
        self.reward_clipping = False
        self.value_clipping = False
        self.clipping_gradient = False
        self.policy_noclip = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print("Using: {} Type: {}".format(self.device,torch.cuda.get_device_name(0)))
        self.actor = Actor(device=self.device,input_dim=self.input_dim, output_dim=self.output_dim,)

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.lr,betas=self.betas)

        self.critic = Critic(device=self.device , input_dim=self.input_dim)

        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.lr,betas=self.betas)
    @staticmethod
    def load_weight(model):
        weight = torch.load("ep_best.pth")
        model.load_state_dict(weight)
    @staticmethod
    def save_weight(model, weight_name):
        save_path = os.path.join("experiments", "checkpoints", "ep_{}.pth".format(weight_name))
        weight = model.state_dict()
        torch.save(weight, save_path)
    # train
    def train(self):
        """
        # initialize env, memory
        # foreach episode
        #   foreach timestep
        #     select action
        #     step action
        #     add exp to the memory
        #     if done or timeout or memory_full: update gae & tdlamret
        #     if memory is full
        #       bootstrap value
        #       optimize
        #       clear memory
        #     if done:
        #       wrapup episode
        #       break
        """
        writer_path = os.path.join('experiment')
        self.writer = SummaryWriter(writer_path)
        self.best_score = 0

        # prepare env, memory, stuff
        env = gym.make(self.evn_name)
        env.seed(self.seed)
        self.memory = PPOMemory(gamma=self.gamma , tau=self.gae_tau)
        score_queue = deque(maxlen=self.average_interval)
        length_queue = deque(maxlen=self.average_interval)

        for episode in trange(1, self.max_episodes+1):
            self.episode = episode
            episode_score = 0
            # reset env
            state = env.reset()
            for t in range(1, self.max_steps_per_episode+1):
                if self.episode % 100 == 0:
                    env.render()
                # select action & estimate value from the state
                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).float() # bsz = 1
                    action_tensor, logpa_tensor = self.actor.select_action(state_tensor)
                    value_tensor = self.critic(state_tensor).squeeze(1) # don't need bsz dim
                # step action
                action = action_tensor.numpy()[0] # single worker
                next_state, reward, done, _ = env.step(action)
                # update episode_score
                episode_score += reward
                # [EXPERIMENT] - reward clipping [-5, 5]
                if self.reward_clipping:
                    reward = np.clip(reward, -5, 5)
                # add experience to the memory
                self.memory.store(s=state, a=action, r=reward, v=value_tensor.item(), lp=logpa_tensor.item())
                # done or timeout or memory full
                # done => v = 0
                # timeout or memory full => v = critic(next_state)
                # update gae & return in the memory!!
                timeout = t == self.max_steps_per_episode
                time_to_optimize = len(self.memory) == self.memory_size
                if done or timeout or time_to_optimize:
                    if done:
                        # cuz the game is over, value of the next state is 0
                        v = 0
                    else:
                        # if not, estimate it with the critic
                        next_state_tensor = torch.tensor(next_state).unsqueeze(0).float() # bsz = 1
                        with torch.no_grad():
                            next_value_tensor = self.critic(next_state_tensor).squeeze(1)
                        v = next_value_tensor.item()

                    # update gae & tdlamret
                    self.memory.finish_path(v)

                # if memory is full, optimize PPO
                if time_to_optimize:
                    self.optimize()
                if done:
                    score_queue.append(episode_score)
                    length_queue.append(t)
                    break
                # update state
                state = next_state

            avg_score = np.mean(score_queue)
            std_score = np.std(score_queue)
            avg_duration = np.mean(length_queue)
            self.writer.add_scalar("info/score", avg_score, self.episode)
            self.writer.add_scalar("info/duration", avg_duration, self.episode)

            if self.episode % 100 == 0:
                print("{} - score: {:.1f} +-{:.1f} \t duration: {}".format(self.episode, avg_score, std_score, avg_duration))

            # game-solved condition
            # if avg_score >= self.config['train']['terminal_score']:
            #     print("game solved at ep {}".format(self.episode))
            #     self.save_weight(self.actor, self.config['exp_name'], "best")
            #     break
            if avg_score >= self.best_score and self.episode >= 200:
                print("found best model at episode: {}".format(self.episode))
                self.save_weight(self.actor, "best")
                self.best_score = avg_score

        self.save_weight(self.actor, "last")
        return self.best_score

    # optimize
    def optimize(self):
        data = self.prepare_data(self.memory.get())
        self.optimize_ppo(data)

    def prepare_data(self, data):
        states_tensor = torch.from_numpy(np.stack(data['states'])).float() # bsz, 8
        actions_tensor = torch.tensor(data['actions']).long() # bsz
        logpas_tensor = torch.tensor(data['logpas']).float() # bsz
        tdlamret_tensor = torch.tensor(data['tdlamret']).float() # bsz
        advants_tensor = torch.tensor(data['advants']).float() # bsz
        values_tensor = torch.tensor(data['values']).float() # bsz

        # normalize advant a.k.a atarg
        advants_tensor = (advants_tensor - advants_tensor.mean()) / (advants_tensor.std() + 1e-5)

        data_tensor = dict(states=states_tensor, actions=actions_tensor, logpas=logpas_tensor,
                    tdlamret=tdlamret_tensor, advants=advants_tensor, values=values_tensor)

        return data_tensor

    def ppo_iter(self, batch_size, ob, ac, oldpas, atarg, tdlamret, vpredbefore):
        total_size = ob.size(0)
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        n_batches = total_size // batch_size
        for nb in range(n_batches):
            ind = indices[batch_size * nb : batch_size * (nb+1)]
            yield ob[ind], ac[ind], oldpas[ind], atarg[ind], tdlamret[ind], vpredbefore[ind]

    def optimize_ppo(self, data):

        """
        https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py line 164

        # get data from the memory
        # prepare dataloader
        # foreach optim_epochs
        #   foreach batch
        #     calculate loss and gradient
        #     update nn
        """

        ob = data['states']
        ac = data['actions']
        oldpas = data['logpas']
        atarg = data['advants']
        tdlamret = data['tdlamret']
        vpredbefore = data['values']

        # can't be arsed..
        eps = self.clip_range

        policy_losses = []
        entropy_losses = []
        value_losses = []

        # foreach policy_update_epochs
        for i in range(self.optim_epochs):
            # foreach batch
            data_loader = self.ppo_iter(self.batch_size,
                                        ob, ac, oldpas, atarg, tdlamret, vpredbefore)
            for batch in data_loader:
                ob_b, ac_b, old_logpas_b, atarg_b, vtarg_b, old_vpred_b = batch

                # policy loss
                cur_logpas, cur_entropies = self.actor.get_predictions(ob_b, ac_b)
                ratio = torch.exp(cur_logpas - old_logpas_b)

                # clip ratio
                clipped_ratio = torch.clamp(ratio, 1.-eps, 1.+eps)

                # policy_loss
                surr1 = ratio * atarg_b

                if self.policy_noclip:
                    pol_surr = -surr1.mean()
                else:
                    surr2 = clipped_ratio * atarg_b
                    pol_surr = -torch.min(surr1, surr2).mean()

                # value_loss
                cur_vpred = self.critic(ob_b).squeeze(1)

                # [EXPERIMENT] - value clipping: clipped_value = old_values + (curr_values - old_values).clip(-eps, +eps)
                if self.value_clipping:
                    cur_vpred_clipped = old_vpred_b + (cur_vpred - old_vpred_b).clamp(-eps, eps)
                    vloss1 = (cur_vpred - vtarg_b).pow(2)
                    vloss2 = (cur_vpred_clipped - vtarg_b).pow(2)
                    vf_loss = torch.max(vloss1, vloss2).mean()
                else:
                    # original value_loss
                    vf_loss = (cur_vpred - vtarg_b).pow(2).mean()

                # entropy_loss
                pol_entpen = -cur_entropies.mean()

                # total loss
                c1 = self.coef_vf
                c2 = self.coef_entpen

                # actor - backward
                self.actor_optimizer.zero_grad()
                policy_loss = pol_surr + c2 * pol_entpen
                policy_loss.backward()

                # [EXPERIMENT] - clipping gradient with max_norm=0.5
                if self.clipping_gradient:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

                self.actor_optimizer.step()

                # critic - backward
                self.critic_optimizer.zero_grad()
                value_loss = c1 * vf_loss
                value_loss.backward()

                # [EXPERIMENT] - clipping gradient with max_norm=0.5
                if self.clipping_gradient:

                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.critic_optimizer.step()

                policy_losses.append(pol_surr.item())
                entropy_losses.append(pol_entpen.item())
                value_losses.append(vf_loss.item())

        avg_policy_loss = np.mean(policy_losses)
        avg_value_losses = np.mean(value_losses)
        avg_entropy_losses = np.mean(entropy_losses)

        self.writer.add_scalar("info/policy_loss", avg_policy_loss, self.episode)
        self.writer.add_scalar("info/value_loss", avg_value_losses, self.episode)
        self.writer.add_scalar("info/entropy_loss", avg_entropy_losses, self.episode)

    # play
    def play(self, num_episodes=1,seed=9999):
        # load policy
        self.load_weight(self.actor)
        print(self.config['exp_name'])
        print(self.config)
        env = gym.make(self.evn_name)
        env.seed(seed)
        scores = []
        for episode in range(num_episodes):
            episode_score = 0
            # initialize env
            state = env.reset()
            while True:
                env.render()
                # select greedy action
                with torch.no_grad():
                    action_tensor = self.actor.select_greedy_action(state)
                action = action_tensor.numpy()[0] # single env
                # run action
                next_state, reward, done, _ = env.step(action)
                # add reward
                episode_score += reward
                # update state
                state = next_state
                # game over condition
                if done:
                    scores.append(episode_score)
                    break
        avg_score = np.mean(scores)
        print("Average score {} on {} games".format(avg_score, num_episodes))
        env.close()

#初始化
def init_normal_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self, device, input_dim, output_dim,):
        super(Actor, self).__init__()

        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layers = nn.ModuleList()
        for idx in range(1):
            self.hidden_layers.append(nn.Linear(64, 64))
        self.output_layer = nn.Linear(64, output_dim)
        self.hfn = torch.tanh
        self.apply(init_normal_weights)
        self.device = device

    def select_action(self, states):
        # sample action
        probs = self.forward(states)
        dist = Categorical(probs=probs)
        actions = dist.sample()

        # log prob of that action
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def select_greedy_action(self, states):
        # select action with the highest prob
        probs = self.forward(states)
        _, actions = probs.max(1)
        return actions

    def get_predictions(self, states, old_actions):
        # get log_probs of old actions and current entropy of the distribution
        state, old_actions = self._format(states), self._format(old_actions)
        probs = self.forward(states)
        dist = Categorical(probs=probs)

        log_probs = dist.log_prob(old_actions)
        entropies = dist.entropy()
        return log_probs, entropies

    def forward(self, state):
        """return action probabilities given state"""
        state = self._format(state)

        x = self.input_layer(state)
        x = self.hfn(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)
        x = self.output_layer(x)
        x = torch.softmax(x, dim=1)
        return x

    def _format(self, state):
        """convert numpy state to tensor and add batch dim"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            state = state.unsqueeze(0) # add bsz dim if state is in numpy array
        return state

class Critic(nn.Module):
    def __init__(self, device, input_dim):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layers = nn.ModuleList()
        for idx in range(1):
            self.hidden_layers.append(nn.Linear(64, 64))
        self.output_layer = nn.Linear(64, 1)
        self.hfn = torch.tanh
        self.apply(init_normal_weights)
        self.device = device

    def forward(self, state):
        """return estimated value given state"""
        state = self._format(state)

        x = self.input_layer(state)
        x = self.hfn(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)
        x = self.output_layer(x)
        return x
    def _format(self, state):
        """convert numpy state to tensor and add batch dim"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.unsqueeze(0)  # add bsz dim if state is in numpy array
        return state

def prepare_dir(overwrite=False):

    exp_dir = os.path.join("experiments")
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    tb_dir = os.path.join(exp_dir, "runs")
    if overwrite:
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(tb_dir)

def main():
    '''
    store_true 是指带触发action时为真，不触发则为假
    '''
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    agent = PPO()

    if args.eval:
        """play mode"""
        agent.play(num_episodes=args.eval_episodes, seed=args.seed)
    else:
        print("Training PPO agent on game {}...".format(agent.evn_name))
        prepare_dir(overwrite=args.overwrite)
        agent.train()
        print("Done\n")

if __name__ == "__main__":
    main()