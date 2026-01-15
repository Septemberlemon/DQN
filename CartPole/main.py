import random
import copy
from collections import deque

import numpy as np
import torch
import gymnasium as gym
import wandb

from model import DQN
from buffer import Buffer


test_name = "test44"
episodes = 2000
buffer_size = 100000
batch_size = 64
learning_rate = 0.0001
gamma = 0.99
epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.01

env = gym.make('CartPole-v1')
obs, info = env.reset(seed=0)

dqn = DQN(4, 2).to("cuda")
# dqn.load_state_dict(torch.load("checkpoints/test7.pth"))
target_dqn = copy.deepcopy(dqn)
optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

wandb.init(project="CartPole", name=test_name, mode="online")

buffer = Buffer(buffer_size)

all_steps = 0
perfect_trajectory = 0
steps_per_episode = deque(maxlen=100)
steps_per_episode.append(10)
for episode in range(episodes):
    obs, info = env.reset()
    steps = 0
    episode_loss = 0
    experiences = []
    while True:
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            q_values = dqn(torch.tensor(obs).to("cuda"))
            action = q_values.argmax().item()

        new_obs, reward, terminated, truncated, info = env.step(action)
        reward = -1.0 if terminated and not truncated else float(reward) * 0.01
        if sum(steps_per_episode) / len(steps_per_episode) < 450:
            buffer.add(obs, action, reward, new_obs, 1 - int(terminated))
        else:
            experiences.append((obs, action, reward, new_obs, 1 - int(terminated)))

        obs = new_obs

        if buffer.size > batch_size:
            samples = buffer.sample(batch_size)
            states, actions, rewards, new_states, not_dones = zip(*samples)

            states = torch.tensor(np.array(states)).to("cuda")
            actions = torch.tensor(np.array(actions)).to("cuda")
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to("cuda")
            new_states = torch.tensor(np.array(new_states)).to("cuda")
            not_dones = torch.tensor(np.array(not_dones)).to("cuda")

            q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_max_q_values, _ = target_dqn(new_states).max(1)
                td_targets = rewards + gamma * next_max_q_values * not_dones
            td_errors = torch.nn.functional.huber_loss(td_targets, q_values)

            optimizer.zero_grad()
            td_errors.backward()
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1)
            optimizer.step()

            episode_loss += td_errors.mean().item()

        steps += 1
        all_steps += 1

        if all_steps % 1000 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        if terminated or truncated:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"episode {episode}, done! steps: {steps}", flush=True)

    steps_per_episode.append(steps)

    if sum(steps_per_episode) / len(steps_per_episode) >= 450:
        if steps == 500:
            if random.random() < 0.1:
                for experience in experiences:
                    buffer.add(*experience)
            else:
                for experience in experiences:
                    buffer.add(*experience)

    wandb.log({
        "steps": steps,
        "td_errors_per_step": episode_loss / steps,
    })

    if steps == 500:
        perfect_trajectory += 1
        if perfect_trajectory == 100:
            break
    else:
        perfect_trajectory = 0

torch.save(dqn.state_dict(), f"checkpoints/{test_name}.pth")
env.close()
wandb.finish()
