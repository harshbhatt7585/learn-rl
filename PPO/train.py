import os
import time
from datetime import datetime

import torch
import numpy as np
import gymnasium as gym  # Updated gym import

from PPO import PPO  # Ensure PPO is implemented correctly


def train():
    env_name = "Walker2d-v4"  # Updated environment

    has_continuous_action_space = True

    max_ep_len = 1000
    max_training_timesteps = int(3e6)
    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 2
    save_model_freq = int(1e5)

    action_std = 0.6
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = int(2.5e5)

    update_timestep = max_ep_len * 4
    K_epochs = 80

    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001

    random_seed = 0

    env = gym.make(env_name)

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if has_continuous_action_space else env.action_space.n

    log_dir = "PPO_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_dir = os.path.join(log_dir, env_name)
    os.makedirs(log_dir, exist_ok=True)

    # Get number of log files
    run_num = len(next(os.walk(log_dir), (None, None, []))[2])

    log_f_name = os.path.join(log_dir, f'PPO_{env_name}_log_{run_num}.csv')

    print(f"Current logging run number for {env_name}: {run_num}")
    print(f"Logging at: {log_f_name}")

    run_num_pretrained = 0
    directory = os.path.join("PPO_preTrained", env_name)
    os.makedirs(directory, exist_ok=True)

    checkpoint_path = os.path.join(directory, f"PPO_{env_name}_{random_seed}_{run_num_pretrained}.pth")
    print(f"Save checkpoint path: {checkpoint_path}")

    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # Start time tracking
    start_time = datetime.now().replace(microsecond=0)
    print(f"Started training at (GMT): {start_time}")

    print("============================================================================================")

    # Open log file
    with open(log_f_name, "w") as log_f:
        log_f.write('episode,timestep,reward\n')

        print_running_reward = 0
        print_running_episodes = 0
        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        while time_step <= max_training_timesteps:
            state, _ = env.reset()  # Fixed reset method for gymnasium
            current_ep_reward = 0

            for t in range(1, max_ep_len + 1):
                action = ppo_agent.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)  # Fixed step output
                done = terminated or truncated

                # Store reward and terminal flag
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # Update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # Decay action std
                if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # Log rewards
                if time_step % log_freq == 0 and log_running_episodes > 0:
                    log_avg_reward = round(log_running_reward / log_running_episodes, 4)
                    log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
                    log_f.flush()
                    log_running_reward = 0
                    log_running_episodes = 0

                # Print average reward
                if time_step % print_freq == 0 and print_running_episodes > 0:
                    print_avg_reward = round(print_running_reward / print_running_episodes, 2)
                    print(f"Episode: {i_episode} \t Timestep: {time_step} \t Average Reward: {print_avg_reward}")

                    print_running_reward = 0
                    print_running_episodes = 0

                # Save model periodically
                if time_step % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print(f"Saving model at: {checkpoint_path}")
                    ppo_agent.save(checkpoint_path)
                    print(f"Model saved | Elapsed Time: {datetime.now().replace(microsecond=0) - start_time}")
                    print("--------------------------------------------------------------------------------------------")

                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1

    env.close()

    # Print total training time
    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print(f"Started training at (GMT): {start_time}")
    print(f"Finished training at (GMT): {end_time}")
    print(f"Total training time: {end_time - start_time}")
    print("============================================================================================")


if __name__ == "__main__":
    train()