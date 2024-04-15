import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models.Attention import Attention

# The network of the actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.d_bus = 8
        self.d_stop = 4
        self.d_proj = 16
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)
        self.norm = nn.LayerNorm(hidden_width)

    def forward(self, s):
        s = self.l1(s.view(s.shape[0], -1))
        s = F.relu(s)
        s = self.norm(s)
        s = self.l2(s)
        a_prob = F.softmax(s, dim=1)
        return a_prob


# The network of the critic
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        s = F.relu(self.l1(s.view(s.shape[0], -1)))
        v_s = self.l2(s)
        return v_s


class A2C(object):
    def __init__(self, bus_state_dim, stop_state_dim, action_dim, n_buses, n_stops, hidden_dim = 64):
        self.bus_state_dim = bus_state_dim
        self.stop_state_dim = stop_state_dim
        self.action_dim = action_dim
        self.attention_out_dim = 16
        self.hidden_dim = 64  # The number of neurons in hidden layers of the neural network
        self.lr = 5e-4  # learning rate
        self.GAMMA = 0.99  # discount factor
        self.I = 1
        # output from attention (1, N_BUSES, attention_out_dim)
        self.shared_attention = Attention(bus_state_dim, stop_state_dim, self.attention_out_dim, 64) # dim_q, dim_k, dim_out, proj_d):

        self.actor = Actor((self.attention_out_dim + bus_state_dim) * n_buses, action_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic((self.attention_out_dim + bus_state_dim) * n_buses, hidden_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s_bus, s_stop = s
        s_bus = torch.unsqueeze(torch.tensor(s_bus, dtype=torch.float), 0)
        s_stop = torch.unsqueeze(torch.tensor(s_stop, dtype=torch.float), 0)
        with torch.no_grad():
            s = self.shared_attention(s_stop, s_stop, s_bus)
            # s = torch.cat((attention_s, s_bus), dim=2)

            prob_weights = self.actor(s).detach().numpy().flatten()  # probability distribution(numpy)
            prob_weights = np.nan_to_num(prob_weights, nan=0.0)
            prob_weights /= np.sum(prob_weights)
            if deterministic:  # We use the deterministic policy during the evaluating
                a = np.argmax(prob_weights)  # Select the action with the highest probability
                return a
            else:  # We use the stochastic policy during the training
                a = np.random.choice(range(self.action_dim), p=prob_weights)  # Sample the action according to the probability distribution
                return a

    def learn(self, s, a, r, s_, dw):
        s_bus, s_stop = s
        s_bus_, s_stop_ = s_
        s_bus = torch.unsqueeze(torch.tensor(s_bus, dtype=torch.float), 0)
        s_stop = torch.unsqueeze(torch.tensor(s_stop, dtype=torch.float), 0)
        s_bus_ = torch.unsqueeze(torch.tensor(s_bus_, dtype=torch.float), 0)
        s_stop_ = torch.unsqueeze(torch.tensor(s_stop_, dtype=torch.float), 0)

        s = self.shared_attention(s_stop, s_stop, s_bus)
        s_ = self.shared_attention(s_stop_, s_stop_, s_bus_)


        # s = torch.cat((attention_s, s_bus), dim=2)
        # s_ = torch.cat((attention_s_, s_bus_), dim=2)

        v_s = self.critic(s).flatten()  # v(s)
        v_s_ = self.critic(s_).flatten()  # v(s')

        with torch.no_grad():  # td_target has no gradient
            td_target = r + self.GAMMA * (1 - dw) * v_s_

        # Update actor
        log_pi = torch.log(self.actor(s).flatten()[a])  # log pi(a|s)
        actor_loss = -self.I * ((td_target - v_s).detach()) * log_pi  # Only calculate the derivative of log_pi
        if torch.isnan(actor_loss):
            print("here")
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        # print("actor_loss:",actor_loss )
        self.actor_optimizer.step()

        # Update critic
        critic_loss = (td_target - v_s) ** 2  # Only calculate the derivative of v(s)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.I *= self.GAMMA  # Represent the gamma^t in th policy gradient theorem


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1']
    env_index = 0
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    number = 9
    # Set random seed
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode

    agent = A2C(state_dim, action_dim)
    writer = SummaryWriter(log_dir='runs/A2C/A2C_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))  # Build a tensorboard

    max_train_steps = 3e5  # Maximum number of training steps
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        episode_steps = 0
        s = env.reset()
        done = False
        agent.I = 1
        while not done:
            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, _ = env.step(a)

            # When dead or win or reaching the max_epsiode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            agent.learn(s, a, r, s_, dw)
            s = s_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                # print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./data_train/A2C_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1
