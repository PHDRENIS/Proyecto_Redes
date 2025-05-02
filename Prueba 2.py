import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import cv2
import os

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_SHAPE = (4, 84, 84)
BATCH_SIZE = 32
MEMORY_CAPACITY = 10000
GAMMA = 0.99
LR = 0.00025
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999
CHECKPOINT_PATH = 'pong_dqn_checkpoint.pth'

# Corregir el generador de aleatorios de gym
def fix_random_generator(env):
    env.unwrapped.np_random = np.random.RandomState()

# Entorno
env = gym.make('PongNoFrameskip-v4', render_mode='human')
fix_random_generator(env)
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False)
env = gym.wrappers.FrameStack(env, 4)

# Red Neuronal
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Buffer de Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# Agente modificado
class DQNAgent:
    def __init__(self, load_checkpoint=False):
        self.policy_net = DQN(env.action_space.n).to(DEVICE)
        self.target_net = DQN(env.action_space.n).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        
        if load_checkpoint and os.path.exists(CHECKPOINT_PATH):
            print("Cargando checkpoint...")
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=DEVICE,
                weights_only=False  # Necesario para cargar el optimizador
            )
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.start_episode = checkpoint['episode'] + 1
            self.frame_idx = checkpoint['frame_idx']
            print(f"Checkpoint cargado. Reanudando desde episodio {self.start_episode}")
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = EPSILON_START
            self.start_episode = 0
            self.frame_idx = 0
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                return self.policy_net(state).argmax().item()
        else:
            return env.action_space.sample()
    
    def update_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, episode):
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'frame_idx': self.frame_idx
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint guardado en episodio {episode}")

# Entrenamiento principal
try:
    agent = DQNAgent(load_checkpoint=True)
    episode_rewards = []
    
    for episode in range(agent.start_episode, 10):  # Entrenar hasta 1000 episodios
        state = env.reset()
        episode_reward = 0
        
        while True:
            agent.frame_idx += 1
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.update_model()
            
            state = next_state
            episode_reward += reward
            
            if agent.frame_idx % 1000 == 0:
                agent.update_target_net()
            
            if done:
                episode_rewards.append(episode_reward)
                print(f"Episodio: {episode}, Recompensa: {episode_reward}, Epsilon: {agent.epsilon:.2f}, Frames: {agent.frame_idx}")
                
                if episode % 10 == 0:
                    agent.save_checkpoint(episode)
                break

except KeyboardInterrupt:
    print("\nEntrenamiento interrumpido manualmente")

finally:
    if 'agent' in locals():
        agent.save_checkpoint(episode)
        print("Checkpoint final guardado")
    env.close()