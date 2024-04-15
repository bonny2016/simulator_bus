from simulator.environ import BusEnv
from simulator.globals import N_BUSES
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models.A2C import A2C
from datetime import datetime
from simulator.globals import N_STOPS,N_BUSES,N_STOP_FEATURES,N_BUS_FEATURES,N_ACTIONS

def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    env.reset()
    evaluate_reward = 0
    for _ in range(times):
        env.reset()
        s = None
        done = False
        episode_reward = 0
        while not done:
            a = [None] * N_BUSES
            if s is None:
                s = []
            for position, state in zip(env.current_bus_indices_needs_decision, s):
                # print(position)
                # print(state)
                a[position] = agent.choose_action(state, deterministic=False)
            s_, r, done, buses_need_decision = env.step(a)
            print("r:", r)
            episode_reward += r
            s = s_
            print()
        print("episode_reward:", episode_reward)
        evaluate_reward += episode_reward
    return float(evaluate_reward / times)


if __name__ == "__main__":
    env = BusEnv()
    env_evaluate = BusEnv()
    # Set random seed
    seed = 0
    # env.seed(seed)
    # env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = A2C(N_BUS_FEATURES, N_STOP_FEATURES, N_ACTIONS, N_BUSES, N_STOPS, 64) #bus_state_dim, stop_state_dim, action_dim, hidden_dim = 64):
    current_time = datetime.now()
    output_format = '%Y%m%d_%H%M%S'
    writer = SummaryWriter(
        log_dir='runs/A2C/{}/seed_{}'.format(current_time.strftime(output_format), seed))

    max_train_steps = 3e5  # Maximum number of training steps
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        episode_steps = 0
        (s, r, is_done, buses_need_decision) = env.reset()
        done = False
        agent.I = 1
        while not done:
            episode_steps += 1
            a = [None] * N_BUSES
            for position, state in zip(env.current_bus_indices_needs_decision, s):
                # print(position)
                # print(state)
                a[position] = agent.choose_action(state, deterministic=False)
            s_, r, done, buses_need_decision = env.step(a)
            rows = env.get_current_data()
            for row in rows:
                (state, action, reward, next_state) = row
                agent.learn(state, action, reward, next_state, dw=0)
            s = s_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                # print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_a2c', evaluate_reward,
                                  global_step=total_steps)
                # Save the rewards
                # if evaluate_num % 10 == 0:
                #     np.save('/log/A2C_number_{}_seed_{}.npy'.format(number, seed),
                #             np.array(evaluate_rewards))if evaluate_num % 10 == 0:
                #     np.save('/log/A2C_number_{}_seed_{}.npy'.format(number, seed),
                #             np.array(evaluate_rewards))if evaluate_num % 10 == 0:
                #     np.save('/log/A2C_number_{}_seed_{}.npy'.format(number, seed),
                #             np.array(evaluate_rewards))

            total_steps += 1
