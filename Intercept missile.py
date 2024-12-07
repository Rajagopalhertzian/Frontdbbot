import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from IPython.display import Video

# Set up video output
FRAME_RATE = 30
OUTPUT_DIR = "frames"
VIDEO_FILENAME = "space_game.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Screen settings
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Game settings
FPS = 30
SPACECRAFT_SPEED = 5
MISSILE_SPEED = 7
METEOR_SPEED = 3

# Initialize Pygame
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))  # Render to an off-screen surface

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Environment for DQN
class SpaceEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.spacecraft_y = HEIGHT // 2
        self.meteor_y = random.randint(50, HEIGHT - 50)
        self.meteor_x = WIDTH - 100
        self.done = False
        return np.array([self.spacecraft_y, self.meteor_x, self.meteor_y])

    def step(self, action):
        # Move spacecraft (up/down)
        if action == 0 and self.spacecraft_y > 0:  # Move up
            self.spacecraft_y -= SPACECRAFT_SPEED
        elif action == 1 and self.spacecraft_y < HEIGHT - 30:  # Move down
            self.spacecraft_y += SPACECRAFT_SPEED
        
        # Move meteor
        self.meteor_x -= METEOR_SPEED

        # Check if the game is over
        if self.meteor_x <= 0:
            self.done = True
            reward = -100  # Penalize for missing the meteor
        else:
            # Reward based on proximity of spacecraft to meteor
            distance = abs(self.spacecraft_y - self.meteor_y)
            reward = -distance  # Smaller distance = higher reward

        return np.array([self.spacecraft_y, self.meteor_x, self.meteor_y]), reward, self.done

# Initialize environment and DQN
env = SpaceEnv()
state_dim = 3
action_dim = 2
dqn = DQN(state_dim, action_dim)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Experience Replay
memory = deque(maxlen=10000)
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64

# Training loop
def train_dqn():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = dqn(states)
    next_q_values = dqn(next_states)
    target_q_values = rewards + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
    q_value = q_values[range(batch_size), actions]

    loss = criterion(q_value, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main game loop
clock = pygame.time.Clock()
state = env.reset()

# Create objects
spacecraft = pygame.Rect(100, HEIGHT // 2, 50, 30)
meteor = pygame.Rect(WIDTH - 100, random.randint(50, HEIGHT - 50), 50, 50)

frame_count = 0
for episode in range(5):  # Run for 5 episodes (adjust as needed)
    state = env.reset()
    for t in range(200):  # Limit frames per episode
        screen.fill(BLACK)

        # DQN Action
        if np.random.rand() < epsilon:
            action = random.randint(0, 1)
        else:
            q_values = dqn(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()

        # Environment step
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            break

        train_dqn()
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Draw spacecraft
        spacecraft.y = env.spacecraft_y
        pygame.draw.rect(screen, BLUE, spacecraft)

        # Draw meteor
        meteor.x = env.meteor_x
        meteor.y = env.meteor_y
        pygame.draw.ellipse(screen, RED, meteor)

        # Save frame to disk
        frame_filename = f"{OUTPUT_DIR}/frame_{frame_count:05d}.png"
        pygame.image.save(screen, frame_filename)
        frame_count += 1

        clock.tick(FPS)

# Use ffmpeg to create video from frames
os.system(f"ffmpeg -y -framerate {FRAME_RATE} -i {OUTPUT_DIR}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {VIDEO_FILENAME}")

# Display the video
Video(VIDEO_FILENAME)
