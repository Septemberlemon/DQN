import random
import copy

import numpy as np
import torch
import gymnasium as gym
import wandb

from model import DQN
from buffer import Buffer


test_name = "test3"
episodes = 1000
buffer_size = 50000
batch_size = 100
learning_rate = 0.0001
gamma = 0.99
epsilon = 1
epsilon_decay = 0.99
epsilon_min = 0.01

env = gym.make('CartPole-v1')
obs, info = env.reset(seed=0)

dqn = DQN(4, 2).to("cuda")
# dqn.load_state_dict(torch.load("checkpoints/test1.pth"))
target_dqn = copy.deepcopy(dqn)
optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

wandb.init(project="CartPole", name=test_name, mode="online")

buffer = Buffer(buffer_size)

all_steps = 0
not_update_episodes = 0
for episode in range(episodes):
    obs, info = env.reset()
    steps = 0
    episode_loss = 0
    not_update_episodes += 1
    while True:
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            q_values = dqn(torch.tensor(obs).to("cuda"))
            action = q_values.argmax().item()

        new_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward) * 0.01 if reward else -1.0
        buffer.add(obs, action, float(reward) * 0.01, new_obs, 1 - int(terminated))

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

        if all_steps % 500 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
            not_update_episodes = 0

        if terminated or truncated:
            break

    if not_update_episodes == 5:
        target_dqn.load_state_dict(dqn.state_dict())
        not_update_episodes = 0

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"episode {episode}, done! steps: {steps}", flush=True)

    wandb.log({
        "steps": steps,
        "td_errors_per_step": episode_loss / steps,
    })

torch.save(dqn.state_dict(), f"checkpoints/{test_name}.pth")
env.close()
wandb.finish()
