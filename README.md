# Snake Game AI 🐍

A classic Snake game with an AI-powered autonomous mode, built using reinforcement learning (Deep Q-Learning, DQN). Play manually using arrow keys or watch the AI navigate the game to achieve high scores!

This project demonstrates the application of reinforcement learning to train an AI agent to play the Snake game. The AI learns to avoid obstacles, collect food, and maximize its score through trial and error, using a Deep Q-Network (DQN) model implemented with PyTorch.

## Features
### Core Gameplay Features
- **Manual Mode**: Play the game using arrow keys (↑, ↓, ←, →) to control the snake’s direction.
- **Autonomous Mode**: Watch the AI play the game using a pre-trained DQN model, optimizing its path to collect food and avoid collisions.
- **Game Over Screen**: Displays your final score when the game ends, with an option to restart (if implemented).
- **Food Prediction**: The AI predicts the location of food to plan its movements efficiently.
- **Score Tracking**: Keeps track of the player’s score based on the number of food items collected.
- **Collision Detection**: Detects collisions with the snake’s body or the game boundaries, ending the game.

### AI-Specific Features
- **Reinforcement Learning (DQN)**: The AI is trained using Deep Q-Learning, a reinforcement learning algorithm that learns optimal actions through trial and error.
- **Pre-Trained Models**: Includes two pre-trained models:
  - `manual_snake_model.pth`: For manual mode (if applicable, e.g., for comparison).
  - `snake_ai_model.pth`: For autonomous mode, trained to maximize the score.
- **State Representation**: The game state is represented as a vector of features (e.g., snake position, food position, direction, and danger proximity) to feed into the DQN.
- **Reward System**: The AI is rewarded for collecting food and penalized for collisions, encouraging it to learn effective strategies.

### Technical Features
- **Standalone Executable**: A Windows executable (`SnakeGameAI.exe`) allows users to play the game without installing Python or dependencies.
- **Customizable Parameters**: Training parameters (e.g., learning rate, discount factor, exploration rate) can be adjusted in the code for retraining the AI.
- **Modular Code Structure**: The codebase is organized into separate components (game logic, AI model, training loop) for easy modification and extension.

## Concepts
This project leverages several key concepts from game development and reinforcement learning. Below is an in-depth explanation of the core concepts used:

### 1. Snake Game Mechanics
The Snake game is a classic arcade game with the following rules:
- The snake moves continuously in a grid-based environment.
- The player (or AI) controls the snake’s direction (up, down, left, right).
- The snake grows longer each time it eats food, increasing the score.
- The game ends if the snake collides with itself (its body) or the game boundaries.
- **Implementation**: The game is built using Pygame, a Python library for 2D game development. Pygame handles rendering the snake, food, and game grid, as well as processing user input (arrow keys) and updating the game state.

### 2. Reinforcement Learning (RL)
Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and aims to maximize the cumulative reward over time.

- **Agent**: The snake (controlled by the AI).
- **Environment**: The game grid, including the snake’s position, food position, and boundaries.
- **Actions**: The snake can move in four directions (up, down, left, right).
- **State**: A representation of the game at a given time, including:
  - The snake’s head position.
  - The food’s position.
  - The snake’s direction.
  - Proximity to dangers (walls, body).
- **Reward Function**:
  - Positive reward (+10) for eating food.
  - Negative reward (-10) for colliding with walls or itself.
  - Small negative reward (-0.1) for each step (to encourage efficiency, if implemented).
- **Goal**: Maximize the total score by collecting food while avoiding collisions.

### 3. Deep Q-Learning (DQN)
Deep Q-Learning is a reinforcement learning algorithm that combines Q-Learning with deep neural networks to handle complex environments with large state spaces.

- **Q-Learning**: A value-based RL algorithm that learns a Q-function, which estimates the expected future reward for taking a given action in a given state. The Q-function is updated using the Bellman equation:
- Q(s, a) = Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]
- - `s`: Current state.
- `a`: Action taken.
- `r`: Reward received.
- `s'`: Next state.
- `α`: Learning rate.
- `γ`: Discount factor (weights future rewards).
- **Deep Neural Network**: In DQN, the Q-function is approximated using a neural network (instead of a Q-table), which takes the state as input and outputs Q-values for each possible action.
- **Experience Replay**: The AI stores past experiences (state, action, reward, next state) in a replay buffer and samples them randomly during training to break correlation between consecutive experiences.
- **Epsilon-Greedy Policy**: The AI balances exploration (trying random actions) and exploitation (choosing the best action based on current Q-values) using an epsilon-greedy strategy. Epsilon (exploration rate) decreases over time.
- **Implementation**: The DQN model is implemented using PyTorch, with a neural network architecture consisting of fully connected layers (e.g., input layer, hidden layers, output layer).

### 4. State Representation
To make the game state understandable to the DQN, the state is converted into a numerical vector. Features might include:
- **Danger Proximity**: Binary indicators for whether a collision is imminent in each direction (e.g., [danger_straight, danger_right, danger_left]).
- **Direction**: The snake’s current direction (e.g., [is_moving_up, is_moving_down, is_moving_left, is_moving_right]).
- **Food Location**: Relative position of the food (e.g., [food_is_up, food_is_down, food_is_left, food_is_right]).
This state vector is fed into the DQN to predict the best action.

### 5. Training Process
The AI is trained by playing the game repeatedly and learning from its experiences:
- The agent starts with random actions (high exploration rate).
- As it plays, it collects experiences and stores them in the replay buffer.
- The DQN is updated by sampling batches of experiences and minimizing the difference between predicted and target Q-values (using a loss function like Mean Squared Error).
- Over time, the exploration rate decreases, and the AI relies more on its learned policy.

## Workflow
The development of this project followed a structured workflow, from game implementation to AI training and deployment. Below is a detailed breakdown of the process:

### 1. Game Development
- **Objective**: Create a functional Snake game with manual control.
- **Steps**:
1. Set up the game environment using Pygame.
2. Implement the snake’s movement, food spawning, and collision detection.
3. Add a scoring system and game over screen.
4. Test the game in manual mode using arrow keys.
- **Tools**: Pygame for rendering and game logic, Python for scripting.

### 2. State and Reward Design
- **Objective**: Design a state representation and reward system for the AI.
- **Steps**:
1. Define the game state as a vector of features (e.g., danger proximity, direction, food location).
2. Design a reward function:
   - +10 for eating food.
   - -10 for collisions.
   - Optional: -0.1 per step to encourage efficiency.
3. Test the state and reward system by printing values during gameplay.

### 3. DQN Model Implementation
- **Objective**: Build and integrate a DQN model to control the snake.
- **Steps**:
1. Create a neural network using PyTorch with an input layer (state size), hidden layers, and output layer (number of actions: 3 or 4, depending on implementation).
2. Implement the DQN algorithm:
   - Experience replay buffer to store experiences.
   - Epsilon-greedy policy for action selection.
   - Loss function and optimizer (e.g., Adam optimizer).
3. Integrate the DQN model into the game loop to predict actions in autonomous mode.

### 4. Training the AI
- **Objective**: Train the DQN model to play the Snake game effectively.
- **Steps**:
1. Run the game in autonomous mode, allowing the AI to play and collect experiences.
2. Update the DQN model using batches of experiences from the replay buffer.
3. Adjust hyperparameters (e.g., learning rate, discount factor, exploration rate) to improve training.
4. Save the trained model weights (`snake_ai_model.pth`).
- **Duration**: Training may take several hours or days, depending on the number of episodes and hardware (CPU/GPU).

### 5. Testing and Evaluation
- **Objective**: Evaluate the AI’s performance and ensure the game works in both modes.
- **Steps**:
1. Test the AI in autonomous mode to verify it can collect food and avoid collisions.
2. Test manual mode to ensure user controls work as expected.
3. Compare scores between manual and autonomous modes.
4. Debug issues (e.g., AI getting stuck in loops, game crashes).

### 6. Building the Executable
- **Objective**: Create a standalone Windows executable for easy distribution.
- **Steps**:
1. Use PyInstaller to bundle the game into a single `.exe` file: pyinstaller --name SnakeGameAI --noconsole --onefile --hidden-import pygame --hidden-import torch --hidden-import numpy --add-data "manual_snake_model.pth;." --add-data "snake_ai_model.pth;." --icon snake.ico --distpath ./output main.py
2. 2. Resolve dependency issues (e.g., PyQt5/PyQt6 conflicts) by using a virtual environment.
3. Test the executable to ensure it runs without requiring Python or dependencies.
4. Move the executable to the `releases/` folder.

### 7. Deployment to GitHub
- **Objective**: Share the project on GitHub for others to use.
- **Steps**:
1. Initialize a Git repository and commit the project files.
2. Create a new GitHub repository and push the project.
3. Create a GitHub Release (`v1.0.0`) and upload `SnakeGameAI.exe`.
4. Update the `README.md` with a download link to the release.

## Demo
![Snake Game AI Demo](https://via.placeholder.com/600x400.png?text=Snake+Game+AI+Demo)  
*Replace the above placeholder with a screenshot or GIF of your game in action, if available.*

## Download the Game (Windows)
For Windows users, download the standalone executable and play the game directly:
- [Download SnakeGameAI.exe](https://github.com/sohamx5800/AI_RL_Snake_Game/releases/download/v1.0.0/SnakeGameAI.exe)
- Double-click the `.exe` file to launch the game.

**Note**: Some antivirus programs might flag the `.exe` as suspicious because it’s a newly created executable. You may need to add an exception in your antivirus settings to run it.

## Running the Source Code
If you prefer to run the game from the source code (e.g., to modify it or train the AI further), follow these steps:

### Prerequisites
- **Python 3.8**: Ensure you have Python 3.8 installed. [Download Python 3.8](https://www.python.org/downloads/release/python-380/).
- **Git**: To clone the repository. [Download Git](https://git-scm.com/downloads).

### Installation
1. **Clone the Repository**:
```bash
git clone https://github.com/sohamx5800/AI_RL_Snake_Game.git
cd AI-Snake-Game
