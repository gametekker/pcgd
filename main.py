#modules
from PCGD import PCGD
from SimGD import SimGD
from PolicyPI import PolicyPi
from visualize import show_pretty_learning_graph

#RL playground
from pettingzoo.mpe import simple_adversary_v3

#training
import torch
import torch.nn as nn
from torch.nn import functional as F

#data
import numpy as np
import pandas as pd
from collections import defaultdict

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Define agents' policies
policy_pi = {"adversary_0": PolicyPi(8).to(device),
            "agent_0": PolicyPi(10).to(device),
             "agent_1": PolicyPi(10).to(device)}

# Save optimizers on policies
opts = {"adversary_0": SimGD(policy_pi["adversary_0"], lr=0.01),
            "agent_0": SimGD(policy_pi["agent_0"], lr=0.01),
            "agent_1": SimGD(policy_pi["agent_1"], lr=0.01)}
pcgd = PCGD(policy_pi, 0.6)

# Training setup
env = simple_adversary_v3.env(render_mode="rgb_array")
reward_records = defaultdict(lambda : [])
batch_size = 2 ** 11 # this size doesn't run on my GPU (don't think it ran on collab either) so using CPU
epochs = 1001
gamma = 0.99

# Training metadata
df = pd.DataFrame({"epoch": [],
                  "trajectories": [],
                  "adversary reward": [],
                  "agent reward": []})

# Choose action from policy network
def pick_sample(s, agent):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        logits = policy_pi[agent](s_batch)
        logits = logits.squeeze(dim=0)
        probs = F.softmax(logits, dim=-1)
        a = torch.multinomial(probs, num_samples=1)
        return a.tolist()[0]

# Run epochs
for i in range(epochs):

    # Gradient information for each sample in the epoch
    zetas = []
    loss_mats = []

    # Run batch
    for j in range(batch_size):

        # Sample from environment
        done = False
        states = defaultdict(lambda : [])
        actions = defaultdict(lambda : [])
        rewards = defaultdict(lambda : [])
        env.reset(seed=(j * epochs + i))
        ss = {}
        for agent in env.agents:
            env.agent_selection = agent
            ss[agent] = env.last()[0]
        while not done:
            t_actions = {}
            for agent in env.agents:
                states[agent].append(ss[agent].tolist())
                t_actions[agent] = pick_sample(ss[agent], agent)
            for agent in env.agents:
                env.agent_selection = agent
                env.step(t_actions[agent])
            for agent in env.agents:
                env.agent_selection = agent
                s, r, term, trunc, _ = env.last()
                ss[agent] = s
                done = term or trunc
                actions[agent].append(t_actions[agent])
                rewards[agent].append(r)

        # Save optimization information
        pcgd_log_probs = {}
        pcgd_cum_rewards = {}
        for agent in env.agents:
            cum_rewards = np.zeros_like(rewards[agent])
            reward_len = len(rewards[agent])
            for j in reversed(range(reward_len)):
                cum_rewards[j] = rewards[agent][j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)

            t_states = torch.tensor(states[agent], dtype=torch.float).to(device)
            t_actions = torch.tensor(actions[agent], dtype=torch.int64).to(device)
            cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
            logits = policy_pi[agent](t_states).to(device)
            log_probs = -F.cross_entropy(logits, t_actions, reduction="none")
            loss = -log_probs * cum_rewards

            pcgd_log_probs[agent] = log_probs
            pcgd_cum_rewards[agent] = cum_rewards

        # Record losses for PCGD
        loss_mat = pcgd.loss_matrix(pcgd_log_probs, pcgd_cum_rewards)
        zeta = pcgd.zeta(pcgd_log_probs, pcgd_cum_rewards)
        loss_mats.append(loss_mat)
        zetas.append(zeta)

    # Perform PCGD update
    batch_zeta = torch.stack(zetas, dim=0).mean(dim=0)
    batch_loss_mat = torch.stack(loss_mats, dim=0).mean(dim=0)
    update = pcgd.compute_loss_mat_update_iterative(batch_loss_mat, batch_zeta)
    pcgd.update_parameters(update)

    # Save details / models as appropriate
    for agent in env.agents:
        print("Run epoch{} with rewards {}".format(i, sum(rewards[agent])))
        if agent == "agent_0":
            ad = pcgd_cum_rewards["adversary_0"][0].detach().numpy()
            ag = pcgd_cum_rewards["agent_0"][0].detach().numpy()
            df.loc[len(df.index)] = [i, i * batch_size, ad, ag]
            df.to_csv(f"pcgd_training_metadata_{i}.csv", index=False)
        torch.save(policy_pi[agent], f"pcgd_{agent}_{i}.model")
        print("MODEL SAVED")
        reward_records[agent].append(sum(rewards[agent]))
        show_pretty_learning_graph(env,reward_records)

print("\nDone")
env.close()