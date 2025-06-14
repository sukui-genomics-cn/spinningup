import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gym

# 神经网络架构定义
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 策略头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # 价值头
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

# PPO核心实现
class PPO:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        self.gamma = config['gamma']          # 折扣因子
        self.epsilon = config['epsilon']      # clip参数
        self.epochs = config['epochs']        # 优化轮次
        self.batch_size = config['batch_size']
        self.gae_lambda = config['gae_lambda'] # GAE参数
        
    def compute_gae(self, rewards, values, dones):
        """计算GAE（广义优势估计）"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            next_value = values[t]
        return advantages
    
    def update(self, states, actions, old_probs, rewards, dones):
        """执行PPO更新"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 转换为张量
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze()
        
        # 计算GAE和回报
        advantages = self.compute_gae(rewards.cpu().numpy(), 
                                    values.cpu().numpy(), 
                                    dones.cpu().numpy())
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values #监督信号
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 分批次训练
        policies_loss = []
        values_loss = []
        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))
            for i in range(0, len(states), self.batch_size):
                idx = indices[i:i+self.batch_size]
                
                # 获取当前策略的概率和价值
                new_probs, current_values = self.model(states[idx])
                new_probs = new_probs.gather(1, actions[idx].unsqueeze(1)).squeeze() #在states的idx下, 模型实际动作对应(idx)的概率
                current_values = current_values.squeeze()
                
                # 计算概率比
                ratio = new_probs / old_probs[idx]
                
                # 策略损失（含clip）
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = nn.MSELoss()(current_values, returns[idx])
                # print(f"epoch: {_}, i:,{i}, ratio:{ratio.mean()}, policy_loss:{policy_loss:.4f}, value_loss:{value_loss:.4f}")
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss
                policies_loss.append(policy_loss)
                values_loss.append(value_loss)
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 梯度裁剪
                self.optimizer.step()
        print(f"Current mean loss: policy: {torch.tensor(policies_loss).mean():.4f}, value: {torch.tensor(values_loss).mean():.4f}")
# 训练循环
def train(env_name="CartPole-v1", config=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if config is None:
        config = {
            'gamma': 0.99,
            'epsilon': 0.2,
            'epochs': 1000,
            'batch_size': 64,
            'lr': 3e-4,
            'gae_lambda': 0.95,
            'max_steps': 10000,
            'update_interval': 2000
        }
    
    agent = PPO(state_dim, action_dim, config)
    state = env.reset()
    episode_rewards = []
    buffer = []
    
    for step in range(1, config['max_steps']+1):
        env.render() 
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action_probs, value = agent.model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
        
        next_state, reward, done, _ = env.step(action)
        buffer.append((state, action, action_probs[0, action].item(), reward, done))
        
        if len(buffer) >= config['update_interval'] or done:
            # 解压缓冲区
            states, actions, old_probs, rewards, dones = zip(*buffer)
            
            # 转换为numpy数组
            states = np.array(states)
            actions = np.array(actions)
            old_probs = np.array(old_probs)
            rewards = np.array(rewards)
            dones = np.array(dones)
            
            # 执行PPO更新
            agent.update(states, actions, old_probs, rewards, dones)
            buffer = []
        
        state = next_state if not done else env.reset()
        
        if done:
            episode_rewards.append(step)
            print(f"Episode {len(episode_rewards)}, reward:{len(rewards)}, Steps {step}")
            
    env.close()
    return episode_rewards

if __name__ == "__main__":
    train()