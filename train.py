import torch
import numpy as np
import matplotlib.pyplot as plt
from game_logic import SnakeGame
from ai_agent import Agent

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EPISODES = 1000

def train_ai():
    game = SnakeGame(30, 20, 20)
    agent = Agent(15, 3, device)  # Updated state size to 15
    scores = []
    total_score = 0

    print("Starting training...\n")

    for episode in range(EPISODES):
        print(f"Training Episode: {episode + 1}/{EPISODES}", end="\r")
        game.reset_game()
        state = agent.get_state(game)
        done = False
        episode_score = 0

        while not done:
            action = agent.get_action(state, training=True)
            reward, done, score = game.play_step([1 if i == action else 0 for i in range(3)])
            next_state = agent.get_state(game)
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_score = score

        agent.train_long_memory()
        agent.n_games += 1
        scores.append(episode_score)
        total_score += episode_score
        mean_score = total_score / agent.n_games
        print(f"\nEpisode {episode+1}/{EPISODES} | Score: {episode_score} | Mean Score: {mean_score:.2f} | Epsilon: {agent.epsilon:.4f}")

    agent.save()
    print("\nTraining complete. Model saved as 'snake_ai_model.pth'.")

    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("AI Learning Progress")
    plt.show()

if __name__ == "__main__":
    train_ai()