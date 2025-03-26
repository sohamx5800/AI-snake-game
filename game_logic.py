import pygame
import random
import math

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class SnakeGame:
    def __init__(self, grid_width, grid_height, cell_size):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.reset_game()
        self.speed = 10

    def reset_game(self):
        self.snake = [(5 * self.cell_size, 5 * self.cell_size)]
        self.snake_dir = (self.cell_size, 0)
        self.food = self.generate_food()
        self.score = 0
        self.speed = 10
        self.prev_dist_to_food = self.dist_to_food()

    def generate_food(self):
        while True:
            x = random.randint(0, self.grid_width - 1) * self.cell_size
            y = random.randint(0, self.grid_height - 1) * self.cell_size
            if (x, y) not in self.snake:
                return (x, y)

    def change_direction(self, new_dir):
        if (new_dir[0] != -self.snake_dir[0] or new_dir[1] != -self.snake_dir[1]):
            self.snake_dir = new_dir

    def dist_to_food(self):
        head = self.snake[0]
        return math.sqrt((head[0] - self.food[0]) ** 2 + (head[1] - self.food[1]) ** 2) / self.cell_size

    def move(self):
        new_head = (self.snake[0][0] + self.snake_dir[0], self.snake[0][1] + self.snake_dir[1])
        done = False
        curr_dist_to_food = self.dist_to_food()
        reward = 0.1  # Base survival reward

        # Reward for moving toward food
        if curr_dist_to_food < self.prev_dist_to_food:
            reward += 1
        elif curr_dist_to_food > self.prev_dist_to_food:
            reward -= 0.5

        # Proximity penalty for walls and body
        if (new_head[0] <= self.cell_size or new_head[0] >= (self.grid_width - 1) * self.cell_size or
            new_head[1] <= self.cell_size or new_head[1] >= (self.grid_height - 1) * self.cell_size):
            reward -= 2  # Penalty for being too close to walls
        if min(self.dist_to_body_straight(), self.dist_to_body_left(), self.dist_to_body_right()) < 2:
            reward -= 2  # Penalty for being too close to body

        # Collision checks
        if (new_head[0] < 0 or new_head[1] < 0 or
            new_head[0] >= self.grid_width * self.cell_size or
            new_head[1] >= self.grid_height * self.cell_size):
            done = True
            reward = -15
        elif new_head in self.snake:
            done = True
            reward = -20
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                self.speed = 10 + self.score * 2
                reward = 10
                self.food = self.generate_food()
                self.prev_dist_to_food = self.dist_to_food()
            else:
                self.snake.pop()
                self.prev_dist_to_food = curr_dist_to_food

        return reward, done, self.score

    def play_step(self, action):
        current_dir = self.snake_dir
        left_turn = (-current_dir[1], current_dir[0])
        right_turn = (current_dir[1], -current_dir[0])

        if action == [1, 0, 0]:
            new_dir = current_dir
        elif action == [0, 1, 0]:
            new_dir = right_turn
        elif action == [0, 0, 1]:
            new_dir = left_turn
        else:
            new_dir = current_dir

        if (new_dir[0] != -self.snake_dir[0] or new_dir[1] != -self.snake_dir[1]):
            self.snake_dir = new_dir

        return self.move()

    def is_collision(self, point):
        x, y = point
        if x < 0 or y < 0 or x >= self.grid_width * self.cell_size or y >= self.grid_height * self.cell_size:
            return True
        if (x, y) in self.snake:
            return True
        return False

    def is_danger_straight(self):
        new_head = (self.snake[0][0] + self.snake_dir[0], self.snake[0][1] + self.snake_dir[1])
        return self.is_collision(new_head)

    def is_danger_left(self):
        left_dir = (-self.snake_dir[1], self.snake_dir[0])
        new_head = (self.snake[0][0] + left_dir[0], self.snake[0][1] + left_dir[1])
        return self.is_collision(new_head)

    def is_danger_right(self):
        right_dir = (self.snake_dir[1], -self.snake_dir[0])
        new_head = (self.snake[0][0] + right_dir[0], self.snake[0][1] + right_dir[1])
        return self.is_collision(new_head)

    def dist_to_body_straight(self):
        head = self.snake[0]
        direction = self.snake_dir
        for i in range(1, max(self.grid_width, self.grid_height)):
            check_pos = (head[0] + i * direction[0], head[1] + i * direction[1])
            if check_pos in self.snake[1:]:
                return i
            if (check_pos[0] < 0 or check_pos[0] >= self.grid_width * self.cell_size or
                check_pos[1] < 0 or check_pos[1] >= self.grid_height * self.cell_size):
                return i
        return float('inf')

    def dist_to_body_left(self):
        head = self.snake[0]
        left_dir = (-self.snake_dir[1], self.snake_dir[0])
        for i in range(1, max(self.grid_width, self.grid_height)):
            check_pos = (head[0] + i * left_dir[0], head[1] + i * left_dir[1])
            if check_pos in self.snake[1:]:
                return i
            if (check_pos[0] < 0 or check_pos[0] >= self.grid_width * self.cell_size or
                check_pos[1] < 0 or check_pos[1] >= self.grid_height * self.cell_size):
                return i
        return float('inf')

    def dist_to_body_right(self):
        head = self.snake[0]
        right_dir = (self.snake_dir[1], -self.snake_dir[0])
        for i in range(1, max(self.grid_width, self.grid_height)):
            check_pos = (head[0] + i * right_dir[0], head[1] + i * right_dir[1])
            if check_pos in self.snake[1:]:
                return i
            if (check_pos[0] < 0 or check_pos[0] >= self.grid_width * self.cell_size or
                check_pos[1] < 0 or check_pos[1] >= self.grid_height * self.cell_size):
                return i
        return float('inf')