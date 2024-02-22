import matplotlib.pyplot as plt
import numpy as np
# Create and display learning curve for all agents
def show_pretty_learning_graph(env,reward_records):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Simple Adversary Reward (PCGD)')
    plt.xlabel("training step")
    plt.ylabel("cumulative reward")
    i = 0
    for agent in env.agents:
        average_reward = []
        std = []
        for idx in range(len(reward_records[agent])):
            avg_list = np.empty(shape=(1,), dtype=int)
            if idx < 5:
                avg_list = reward_records[agent][:idx+1]
            else:
                avg_list = reward_records[agent][idx-4:idx+1]
            average_reward.append(np.average(avg_list))
            std.append(np.std(avg_list))
        axs[i].set_title(agent)
        axs[i].plot(average_reward, label="average reward (last 5 steps)")
        axs[i].fill_between(range(len(reward_records[agent])), np.array(average_reward) - np.array(std),
                        np.array(average_reward) + np.array(std), alpha=0.2)
        i += 1
    plt.legend(loc="lower right")
    fig.set_size_inches(22, 10)
    plt.show()