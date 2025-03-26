import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from heapq import heappush, heappop
from model import DQN
import os 

class Agent:
    def __init__(self, state_size, action_size, device):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.n_games = 0
        self.record = 0
        self.food_history = deque(maxlen=10)
        self.predicted_food = None

    def get_state(self, game):
        head = game.snake[0]
        point_l = (head[0] - game.cell_size, head[1])
        point_r = (head[0] + game.cell_size, head[1])
        point_u = (head[0], head[1] - game.cell_size)
        point_d = (head[0], head[1] + game.cell_size)

        dir_l = game.snake_dir == (-game.cell_size, 0)
        dir_r = game.snake_dir == (game.cell_size, 0)
        dir_u = game.snake_dir == (0, -game.cell_size)
        dir_d = game.snake_dir == (0, game.cell_size)

        dist_to_food = game.dist_to_food()
        dist_to_body_straight = game.dist_to_body_straight()
        dist_to_body_left = game.dist_to_body_left()
        dist_to_body_right = game.dist_to_body_right()

        dist_to_wall_left = head[0] / (game.grid_width * game.cell_size)
        dist_to_wall_right = (game.grid_width * game.cell_size - head[0]) / (game.grid_width * game.cell_size)
        dist_to_wall_up = head[1] / (game.grid_height * game.cell_size)
        dist_to_wall_down = (game.grid_height * game.cell_size - head[1]) / (game.grid_height * game.cell_size)

        state = [
            game.is_danger_straight(),
            game.is_danger_left(),
            game.is_danger_right(),
            dir_l, dir_r, dir_u, dir_d,
            game.food[0] < head[0],
            game.food[0] > head[0],
            game.food[1] < head[1],
            game.food[1] > head[1],
            dist_to_food / max(game.grid_width, game.grid_height),
            min(dist_to_body_straight, 10) / 10,
            min(dist_to_body_left, 10) / 10,
            min(dist_to_body_right, 10) / 10,
            dist_to_wall_left,
            dist_to_wall_right,
            dist_to_wall_up,
            dist_to_wall_down
        ]

        return np.array(state, dtype=float)

    def a_star_path(self, game, start, goal):
        start = (start[0] // game.cell_size, start[1] // game.cell_size)
        goal = (goal[0] // game.cell_size, goal[1] // game.cell_size)
        grid_width, grid_height = game.grid_width, game.grid_height
        snake_body = set((x // game.cell_size, y // game.cell_size) for x, y in game.snake[1:])

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append((current[0] * game.cell_size, current[1] * game.cell_size))
                    current = came_from[current]
                path.append((start[0] * game.cell_size, start[1] * game.cell_size))
                return path[::-1]

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height and
                    neighbor not in snake_body):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def get_action(self, state, game, training=True):
        head = game.snake[0]
        current_dir = game.snake_dir
        left_turn = (-current_dir[1], current_dir[0])
        right_turn = (current_dir[1], -current_dir[0])


        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]
            q_values_str = [f"{q:.4f}" for q in q_values]

        if training and random.random() >= self.epsilon:  # 90% A* path
            path = self.a_star_path(game, head, game.food)
            if path and len(path) > 1:
                next_pos = path[1]
                dx = next_pos[0] - head[0]
                dy = next_pos[1] - head[1]
                if (dx, dy) == current_dir:
                    action = 0
                elif (dx, dy) == right_turn:
                    action = 1
                elif (dx, dy) == left_turn:
                    action = 2
                else:
                    action = random.randrange(self.action_size)
                print(f"A* Path Action: {action}, Q-values: {q_values_str}")
                return action

        # 10% chance: either random or DQN
        if random.random() < 0.5:  # 5% random
            action = random.randrange(self.action_size)
            print(f"Exploration: Action {action} (Epsilon: {self.epsilon}), Q-values: {q_values_str}")
        else:  # 5% DQN
            action = np.argmax(q_values)
            print(f"Exploitation: Action {action}, Q-values: {q_values_str}")
        return action

    def predict_action_to_food(self, game, food_pos):
        """Predict the action to move toward a given food position using A*."""
        head = game.snake[0]
        current_dir = game.snake_dir
        left_turn = (-current_dir[1], current_dir[0])
        right_turn = (current_dir[1], -current_dir[0])

        path = self.a_star_path(game, head, food_pos)
        if path and len(path) > 1:
            next_pos = path[1]
            dx = next_pos[0] - head[0]
            dy = next_pos[1] - head[1]
            if (dx, dy) == current_dir:
                return 0  # Straight
            elif (dx, dy) == right_turn:
                return 1  # Right
            elif (dx, dy) == left_turn:
                return 2  # Left
        return None  # No path found

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 64:
            minibatch = random.sample(self.memory, 64)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, states, actions, rewards, next_states, dones):
        self.model.train()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)

        targets = rewards + self.gamma * torch.max(self.model(next_states), dim=1)[0] * (1 - dones)
        q_values = self.model(states).gather(1, torch.tensor(actions).unsqueeze(1).to(self.device)).squeeze(1)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def pretrain(self, manual_data):
        if not manual_data:
            print("No valid manual data for pretraining.")
            return
        states, actions = zip(*manual_data)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        valid_actions = []
        for action in actions:
            if sum(action) == 1:
                try:
                    valid_actions.append(action.index(1))
                except ValueError:
                    print(f"Warning: Invalid action {action} skipped.")
                    continue
            else:
                print(f"Warning: Invalid action {action} skipped.")
                continue
        if not valid_actions:
            print("No valid actions for pretraining.")
            return
        actions = torch.tensor(valid_actions, dtype=torch.long).to(self.device)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(10):
            outputs = self.model(states)
            targets = torch.zeros_like(outputs)
            for i, action in enumerate(actions):
                targets[i, action] = 1.0
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Pretraining Epoch {epoch + 1}/10, Loss: {loss.item():.4f}")

        self.model.eval()
        print("✅ Pretraining completed.")

    def save(self, file_name="snake_ai_model.pth"):
        state = {
            'state_dict': self.model.state_dict(),
            'state_size': self.state_size,
            'hidden_size': 256,
            'action_size': self.action_size,
            'food_history': list(self.food_history)
        }
        torch.save(state, file_name)
        print(f"✅ Model saved to {file_name}")

    def load(self, file_name="snake_ai_model.pth"):
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                if (checkpoint['state_size'] == self.state_size and
                    checkpoint['hidden_size'] == 256 and
                    checkpoint['action_size'] == self.action_size):
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.food_history = deque(checkpoint.get('food_history', []), maxlen=10)
                    self.model.eval()
                    print(f"✅ Loaded model from {file_name} with metadata")
                else:
                    print(f"⚠ Model architecture mismatch in {file_name}. Expected state_size={self.state_size}, hidden_size=256, got {checkpoint.get('state_size', 'N/A')}, {checkpoint.get('hidden_size', 'N/A')}")
            else:
                try:
                    self.model.load_state_dict(checkpoint)
                    self.model.eval()
                    print(f"✅ Loaded legacy model from {file_name} (no metadata)")
                except RuntimeError as e:
                    print(f"⚠ Failed to load legacy model from {file_name}: {e}")
        else:
            print(f"⚠ No model found at {file_name}")

    def predict_next_food(self, game):
        if not self.food_history:
            pred_x = (game.grid_width * game.cell_size) // 2
            pred_y = (game.grid_height * game.cell_size) // 2
        else:
            avg_x = int(sum(f[0] for f in self.food_history) / len(self.food_history))
            avg_y = int(sum(f[1] for f in self.food_history) / len(self.food_history))
            pred_x = max(0, min(avg_x, (game.grid_width - 1) * game.cell_size))
            pred_y = max(0, min(avg_y, (game.grid_height - 1) * game.cell_size))
            while (pred_x, pred_y) in game.snake:
                pred_x = (pred_x + game.cell_size) % (game.grid_width * game.cell_size)
                pred_y = (pred_y + game.cell_size) % (game.grid_height * game.cell_size)

        self.predicted_food = (pred_x, pred_y)
        print(f"Predicted next food at: {self.predicted_food}")
        return self.predicted_food