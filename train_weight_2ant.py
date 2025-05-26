'''
完整版本的mappo算法代码
created by: Guowei Zou
modified: 
'''
import os
import shutil
from time import sleep
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from gymnasium_robotics import mamujoco_v1


# 创建结构化的结果目录
result_base_dir = "results"
env_name = "mappo_2ant"
result_dir = os.path.join(result_base_dir, env_name)

# 清空整个 results 目录（可选）
# if os.path.exists(result_base_dir):
#     shutil.rmtree(result_base_dir)
# os.makedirs(result_base_dir, exist_ok=True)

# 创建子目录
logs_dir = os.path.join(result_dir, "logs")
plots_dir = os.path.join(result_dir, "plots")
weights_dir = os.path.join(result_dir, "weights")

# 确保目录存在
for directory in [logs_dir, plots_dir, weights_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# 设置日志文件路径
log_file = os.path.join(logs_dir, "training_log.txt")

if os.path.exists(log_file):
    open(log_file, "w").close()

# 日志记录函数
def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def plot_all_metrics(metrics_dict, episode):
    """
    将所有指标绘制到一个包含多个子图的图表中
    - 对曲线进行平滑处理
    - 添加误差带显示
    参数:
    metrics_dict: 包含所有指标数据的字典，格式为 {metric_name: values_list}
    episode: 当前的episode数
    """
    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics of {env_name} (Up to Episode {episode})', fontsize=16)
    
    # 压平axes数组以便迭代
    axes = axes.flatten()
    
    # 为每个指标获取x轴值
    any_metric = list(metrics_dict.values())[0]
    x_values = [50 * (i + 1) for i in range(len(any_metric))]
    
    # 平滑参数 - 窗口大小
    window_size = min(5, len(x_values)) if len(x_values) > 0 else 1
    
    # 在每个子图中绘制一个指标
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= 5:  # 我们只有5个指标
            break
            
        ax = axes[i]
        values_array = np.array(values)
        
        # 应用平滑处理
        if len(values) > window_size:
            # 创建平滑曲线
            smoothed = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
            
            # 计算滚动标准差用于误差带
            std_values = []
            for j in range(len(values) - window_size + 1):
                std_values.append(np.std(values_array[j:j+window_size]))
            std_values = np.array(std_values)
            
            # 调整x轴以匹配平滑后的数据长度
            smoothed_x = x_values[window_size-1:]
            
            # 绘制平滑曲线和原始散点
            ax.plot(smoothed_x, smoothed, '-', linewidth=2, label='Smoothed')
            ax.scatter(x_values, values, alpha=0.3, label='Original')
            
            # 添加误差带
            ax.fill_between(smoothed_x, smoothed-std_values, smoothed+std_values, 
                           alpha=0.2, label='±1 StdDev')
        else:
            # 如果数据点太少，只绘制原始数据
            ax.plot(x_values, values, 'o-', label='Data')
        
        ax.set_title(metric_name.replace('_', ' '))
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 删除未使用的子图
    if len(metrics_dict) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, f'mappo_training_metrics.png'))
    plt.close(fig)

def compute_entropy(mean, log_std):
    std = log_std.exp()
    entropy = 0.5 + 0.5 * np.log(2 * np.pi) + log_std
    return entropy.mean().item()

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


# 策略网络(Actor)
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mean = torch.nn.Linear(hidden_dim, action_dim)  # 输出动作均值
        self.log_std = torch.nn.Linear(hidden_dim, action_dim)  # 输出动作标准差的对数
        
    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        return mean, log_std

# 全局价值网络(CentralValueNet)
# 输入: 所有智能体的状态拼接 (team_size * state_dim)
# 输出: 对每个智能体的价值估计 (team_size维向量)
class CentralValueNet(torch.nn.Module):
    def __init__(self, total_state_dim, hidden_dim, team_size):
        super(CentralValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, team_size)  # 输出为每个智能体一个价值

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)  # [batch, team_size]


class MAPPO:
    def __init__(self, team_size, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, eps, gamma, device):
        self.team_size = team_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device

        # 为每个智能体一个独立的actor
        self.actors = [PolicyNet(state_dim, hidden_dim, action_dim).to(device)
                       for _ in range(team_size)]

        # 一个全局critic，输入为所有智能体状态拼接
        self.critic = CentralValueNet(team_size * state_dim, hidden_dim, team_size).to(device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), actor_lr) 
                                 for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

    def save_model(self, path=None):
        if path is None:
            path = weights_dir
        if not os.path.exists(path):
            os.makedirs(path)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path, f"actor_{i}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load_model(self, path=None):
        if path is None:
            path = weights_dir
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor_{i}.pth")
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path))
        critic_path = os.path.join(path, "critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))

    def take_action(self, state_per_agent):
        actions = []
        action_probs = []
        for i, actor in enumerate(self.actors):
            s = torch.tensor(np.array([state_per_agent[i]]), dtype=torch.float).to(self.device)
            mean, log_std = actor(s)
            std = log_std.exp()
            
            # 使用正态分布采样动作
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            action_prob = normal.log_prob(action).sum(dim=-1).exp()  # 动作的概率密度
            
            actions.append(action.cpu().detach().numpy().flatten())
            action_probs.append(action_prob.cpu().detach().numpy())
        return actions, action_probs

    def update(self, transition_dicts, state_dim):
        # 拼接所有智能体的数据，用于全局critic
        # 首先统一长度T，假设所有智能体长度相同（因为同步环境步）
        T = len(transition_dicts[0]['states'])
        # 将所有智能体在同一时间步的state拼接起来，得到 [T, team_size*state_dim]
        states_all = []
        next_states_all = []
        for t in range(T):
            concat_state = []
            concat_next_state = []
            for i in range(self.team_size):
                concat_state.append(transition_dicts[i]['states'][t])
                concat_next_state.append(transition_dicts[i]['next_states'][t])
            states_all.append(np.concatenate(concat_state))
            next_states_all.append(np.concatenate(concat_next_state))

        states_all = torch.tensor(states_all, dtype=torch.float).to(self.device)  # [T, team_size*state_dim]
        next_states_all = torch.tensor(next_states_all, dtype=torch.float).to(self.device) # [T, team_size*state_dim]

        rewards_all = torch.tensor([[transition_dicts[i]['rewards'][t] for i in range(self.team_size)] 
                                     for t in range(T)], dtype=torch.float).to(self.device) # [T, team_size]
        dones_all = torch.tensor([[transition_dicts[i]['dones'][t] for i in range(self.team_size)] 
                                   for t in range(T)], dtype=torch.float).to(self.device) # [T, team_size]

        # 从critic计算价值和TD-target
        values = self.critic(states_all) # [T, team_size]    
        next_values = self.critic(next_states_all) # [T, team_size]
        td_target = rewards_all + self.gamma * next_values * (1 - dones_all) # [T, team_size]
        td_delta = td_target - values # [T, team_size]

        # 为每个智能体计算其优势
        advantages = []
        for i in range(self.team_size):
            adv_i = compute_advantage(self.gamma, self.lmbda, td_delta[:, i])
            advantages.append(adv_i.to(self.device))  # [T]

        # 更新critic
        # critic的loss是所有智能体的均方误差平均
        critic_loss = F.mse_loss(values, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新每个智能体的actor
        action_losses = []
        entropies = []

        for i in range(self.team_size):
            states = torch.tensor(transition_dicts[i]['states'], dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dicts[i]['actions'], dtype=torch.float).to(self.device)  # 确保actions是浮点型
            old_probs = torch.tensor(transition_dicts[i]['action_probs'], dtype=torch.float).to(self.device)

            # 获取当前的均值和对数标准差，创建正态分布
            mean, log_std = self.actors[i](states)
            std = log_std.exp()
            normal_dist = torch.distributions.Normal(mean, std)
            
            # 计算当前动作的对数概率
            log_probs = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
            old_log_probs = torch.log(old_probs).detach()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages[i].unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages[i].unsqueeze(-1)

            action_loss = torch.mean(-torch.min(surr1, surr2))
            
            # 对于连续动作空间，熵的计算不同
            entropy_val = torch.mean(normal_dist.entropy()).item()

            self.actor_optimizers[i].zero_grad()
            action_loss.backward()
            self.actor_optimizers[i].step()

            action_losses.append(action_loss.item())
            entropies.append(entropy_val)

        return np.mean(action_losses), critic_loss.item(), np.mean(entropies)


# 参数设置
actor_lr = 3e-4
critic_lr = 1e-3
total_episodes = 100000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.3
team_size = 2  # 智能体数量
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env = mamujoco_v1.parallel_env("Ant", "2x4")
# 重置环境并获取初始智能体列表（用于保证动作、状态的顺序一致）
states = env.reset()
agent_order = list(env.agents)  # 保持agent的顺序
# 提取状态维度和动作维度（假设所有智能体相同）
state_dim = env.observation_space(agent_order[0]).shape[0]
action_dim = env.action_space(agent_order[0]).shape[0]

# 创建MAPPO智能体（共有team_size个actor, 一个共享critic）
mappo = MAPPO(team_size, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

# 用于统计指标的列表
total_rewards_per_episode = []
episode_lengths = []
policy_losses = []
value_losses = []
entropies = []

# 每50个episode的平均值列表
avg_total_rewards_per_50 = []
avg_episode_length_per_50 = []
avg_policy_loss_per_50 = []
avg_value_loss_per_50 = []
avg_entropy_per_50 = []

with tqdm(total=total_episodes, desc="Training") as pbar:
    for episode in range(1, total_episodes + 1):
        # 初始化Trajectory buffer
        buffers = [{
            'states': [], 
            'actions': [], 
            'next_states': [], 
            'rewards': [], 
            'dones': [], 
            'action_probs': []
        } for _ in range(team_size)]
        
        # 重置环境，注意 reset() 返回可能是元组 (state_dict, info)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            current_states, _ = reset_result  # 解包元组，获取状态字典
        else:
            current_states = reset_result  # 如果直接返回状态字典

        # 保持每个episode开始时的 agent 顺序（以当前 env.agents 为准）
        agent_order = list(env.agents)
        terminal = False
        episode_reward = 0.0
        steps = 0

        while not terminal:
            steps += 1
            # 将当前状态字典转换为列表（按照 agent_order 的顺序）
            state_list = [current_states[agent] for agent in agent_order]
            actions, prob_dists = mappo.take_action(state_list)
            # 将动作列表转换为字典
            action_dict = {agent_order[i]: actions[i] for i in range(team_size)}
            
            # 修改：适应新的 PettingZoo API 返回值格式
            next_states, rewards, terminations, truncations, infos = env.step(action_dict)
            # 合并 terminations 和 truncations 为 dones
            dones = {agent: terminations[agent] or truncations[agent] for agent in agent_order}
            
            # 累计总奖励
            step_reward = sum(rewards.values())
            episode_reward += step_reward

            # 存储transition
            for i, agent in enumerate(agent_order):
                buffers[i]['states'].append(np.array(current_states[agent]))
                buffers[i]['actions'].append(actions[i])
                buffers[i]['next_states'].append(np.array(next_states[agent]))
                buffers[i]['rewards'].append(rewards[agent])
                buffers[i]['dones'].append(float(dones[agent]))
                buffers[i]['action_probs'].append(prob_dists[i])

            current_states = next_states
            terminal = all(dones.values())

        # 使用MAPPO更新参数
        a_loss, c_loss, ent = mappo.update(buffers, state_dim)

        # 记录指标
        total_rewards_per_episode.append(episode_reward)
        episode_lengths.append(steps)
        policy_losses.append(a_loss)
        value_losses.append(c_loss)
        entropies.append(ent)
        # 保存模型的权重参数
        if episode % 500 == 0:
            mappo.save_model()
            log_message(f"Model saved at episode {episode}")

        # 每50个episode统计一次平均值并记录日志、绘图
        if episode % 50 == 0:
            avg_reward_50 = np.mean(total_rewards_per_episode[-50:])
            avg_length_50 = np.mean(episode_lengths[-50:])
            avg_policy_loss_50 = np.mean(policy_losses[-50:])
            avg_value_loss_50 = np.mean(value_losses[-50:])
            avg_entropy_50 = np.mean(entropies[-50:])

            avg_total_rewards_per_50.append(avg_reward_50)
            avg_episode_length_per_50.append(avg_length_50)
            avg_policy_loss_per_50.append(avg_policy_loss_50)
            avg_value_loss_per_50.append(avg_value_loss_50)
            avg_entropy_per_50.append(avg_entropy_50)

            log_message(f"Episode {episode}: "
                        f"AvgTotalReward(last50)={avg_reward_50:.3f}, "
                        f"AvgEpisodeLength(last50)={avg_length_50:.3f}, "
                        f"AvgPolicyLoss(last50)={avg_policy_loss_50:.3f}, "
                        f"AvgValueLoss(last50)={avg_value_loss_50:.3f}, "
                        f"AvgEntropy(last50)={avg_entropy_50:.3f}")
                
            # 创建指标字典
            metrics_dict = {
                "Average_Total_Reward": avg_total_rewards_per_50,
                "Average_Episode_Length": avg_episode_length_per_50,
                "Average_Policy_Loss": avg_policy_loss_per_50,
                "Average_Value_Loss": avg_value_loss_per_50, 
                "Average_Entropy": avg_entropy_per_50
            }
                
            # 调用新的绘图函数
            plot_all_metrics(metrics_dict, episode)

        pbar.update(1)
