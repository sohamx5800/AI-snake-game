import pygame
import sys
import torch
import numpy as np
import os
from game_logic import SnakeGame
from ui import (initialize_display, update_display, draw_game_over, draw_homepage,
                handle_homepage_input, handle_game_over_input, draw_popup_message)
from ai_agent import Agent

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pygame Initialization
pygame.init()
font = pygame.font.Font(None, 36)

# Screen Settings
info = pygame.display.Info()
SCREEN_WIDTH = int(info.current_w * 0.8)
SCREEN_HEIGHT = int(info.current_h * 0.8)
SIDE_PANEL_WIDTH = 300
CELL_SIZE = 20
FPS = 10

GRID_WIDTH = (SCREEN_WIDTH - SIDE_PANEL_WIDTH) // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

screen = initialize_display(SCREEN_WIDTH, SCREEN_HEIGHT, SIDE_PANEL_WIDTH)
clock = pygame.time.Clock()
fullscreen = False

def main():
    global SCREEN_WIDTH, SCREEN_HEIGHT, GRID_WIDTH, GRID_HEIGHT, screen, fullscreen
    total_score, record = 0, 0
    agent = Agent(19, 3, device)
    game = SnakeGame(GRID_WIDTH, GRID_HEIGHT, CELL_SIZE)
    manual_data = []
    manual_movements = 0
    successful_moves = 0
    show_popup = False
    game_over = False

    # Start with homepage
    play_manual_button, play_ai_button = draw_homepage(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)
    pygame.display.flip()
    mode = None
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
                GRID_WIDTH = (SCREEN_WIDTH - SIDE_PANEL_WIDTH) // CELL_SIZE
                GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
                game.grid_width, game.grid_height = GRID_WIDTH, GRID_HEIGHT
                play_manual_button, play_ai_button = draw_homepage(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)
                pygame.display.flip()
            elif event.type == pygame.VIDEORESIZE:
                SCREEN_WIDTH, SCREEN_HEIGHT = event.w, event.h
                GRID_WIDTH = (SCREEN_WIDTH - SIDE_PANEL_WIDTH) // CELL_SIZE
                GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
                game.grid_width, game.grid_height = GRID_WIDTH, GRID_HEIGHT
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                play_manual_button, play_ai_button = draw_homepage(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)
                pygame.display.flip()
        mode = handle_homepage_input(screen, font, play_manual_button, play_ai_button, events)
        if mode:
            break
        pygame.display.flip()
        clock.tick(60)

    autonomous_mode = (mode == "ai")
    training_mode = True if autonomous_mode else False
    used_manual_model = False

    if autonomous_mode:
        if os.path.exists("manual_snake_model.pth"):
            agent.load("manual_snake_model.pth")
            if agent.model.fc1.in_features == 19:
                agent.epsilon = 0.1
                used_manual_model = True
                print("✅ Loaded human-trained model for autonomous play with epsilon =", agent.epsilon)
            else:
                print("⚠ manual_snake_model.pth has incompatible state size, starting fresh")
        elif os.path.exists("snake_ai_model.pth"):
            agent.load("snake_ai_model.pth")
            if agent.model.fc1.in_features == 19:
                print("✅ Loaded AI-trained model for autonomous play with epsilon =", agent.epsilon)
            else:
                print("⚠ snake_ai_model.pth has incompatible state size, starting fresh")
        else:
            print("⚠ No pre-trained model found, starting with fresh model")
        agent.predict_next_food(game)

    while True:
        if show_popup:
            message = "Model trained successfully!\nReason: Player completed 1000 moves."
            continue_button, exit_button = draw_popup_message(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT, message)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if continue_button.collidepoint(mouse_pos):
                        # Reset game state and continue manual mode
                        game.reset_game()
                        manual_data = []
                        manual_movements = 0
                        successful_moves = 0
                        show_popup = False
                        game_over = False
                        agent.predicted_food = None  # Clear predicted food
                    elif exit_button.collidepoint(mouse_pos):
                        # Return to homepage
                        play_manual_button, play_ai_button = draw_homepage(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)
                        pygame.display.flip()
                        while True:
                            events = pygame.event.get()
                            for event in events:
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()
                                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                                    pygame.quit()
                                    sys.exit()
                            mode = handle_homepage_input(screen, font, play_manual_button, play_ai_button, events)
                            if mode:
                                break
                            pygame.display.flip()
                            clock.tick(60)
                        autonomous_mode = (mode == "ai")
                        training_mode = True if autonomous_mode else False
                        game.reset_game()
                        manual_data = []
                        manual_movements = 0
                        successful_moves = 0
                        show_popup = False
                        game_over = False
                        used_manual_model = False
                        agent = Agent(19, 3, device)
                        if autonomous_mode:
                            if os.path.exists("manual_snake_model.pth"):
                                agent.load("manual_snake_model.pth")
                                if agent.model.fc1.in_features == 19:
                                    agent.epsilon = 0.1
                                    used_manual_model = True
                                    print("✅ Loaded human-trained model for autonomous play with epsilon =", agent.epsilon)
                                else:
                                    print("⚠ manual_snake_model.pth has incompatible state size, starting fresh")
                            elif os.path.exists("snake_ai_model.pth"):
                                agent.load("snake_ai_model.pth")
                                if agent.model.fc1.in_features == 19:
                                    print("✅ Loaded AI-trained model for autonomous play with epsilon =", agent.epsilon)
                                else:
                                    print("⚠ snake_ai_model.pth has incompatible state size, starting fresh")
                            else:
                                print("⚠ No pre-trained model found, starting with fresh model")
                            agent.predict_next_food(game)
            continue

        if not game_over:
            done = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    # Return to homepage for both modes
                    play_manual_button, play_ai_button = draw_homepage(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)
                    pygame.display.flip()
                    while True:
                        events = pygame.event.get()
                        for event in events:
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                sys.exit()
                        mode = handle_homepage_input(screen, font, play_manual_button, play_ai_button, events)
                        if mode:
                            break
                        pygame.display.flip()
                        clock.tick(60)
                    autonomous_mode = (mode == "ai")
                    training_mode = True if autonomous_mode else False
                    game.reset_game()
                    manual_data = []
                    manual_movements = 0
                    successful_moves = 0
                    show_popup = False
                    game_over = False
                    used_manual_model = False
                    agent = Agent(19, 3, device)
                    if autonomous_mode:
                        if os.path.exists("manual_snake_model.pth"):
                            agent.load("manual_snake_model.pth")
                            if agent.model.fc1.in_features == 19:
                                agent.epsilon = 0.1
                                used_manual_model = True
                                print("✅ Loaded human-trained model for autonomous play with epsilon =", agent.epsilon)
                            else:
                                print("⚠ manual_snake_model.pth has incompatible state size, starting fresh")
                        elif os.path.exists("snake_ai_model.pth"):
                            agent.load("snake_ai_model.pth")
                            if agent.model.fc1.in_features == 19:
                                print("✅ Loaded AI-trained model for autonomous play with epsilon =", agent.epsilon)
                            else:
                                print("⚠ snake_ai_model.pth has incompatible state size, starting fresh")
                        else:
                            print("⚠ No pre-trained model found, starting with fresh model")
                        agent.predict_next_food(game)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
                    GRID_WIDTH = (SCREEN_WIDTH - SIDE_PANEL_WIDTH) // CELL_SIZE
                    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
                    game.grid_width, game.grid_height = GRID_WIDTH, GRID_HEIGHT
                    pygame.display.flip()
                elif event.type == pygame.VIDEORESIZE:
                    SCREEN_WIDTH, SCREEN_HEIGHT = event.w, event.h
                    GRID_WIDTH = (SCREEN_WIDTH - SIDE_PANEL_WIDTH) // CELL_SIZE
                    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
                    game.grid_width, game.grid_height = GRID_WIDTH, GRID_HEIGHT
                    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                    pygame.display.flip()
                elif not autonomous_mode and event.type == pygame.KEYDOWN:
                    new_direction = game.snake_dir
                    if event.key == pygame.K_UP:
                        new_direction = (0, -CELL_SIZE)
                    elif event.key == pygame.K_DOWN:
                        new_direction = (0, CELL_SIZE)
                    elif event.key == pygame.K_LEFT:
                        new_direction = (-CELL_SIZE, 0)
                    elif event.key == pygame.K_RIGHT:
                        new_direction = (CELL_SIZE, 0)
                    game.change_direction(new_direction)
                    manual_movements += 1

            if not autonomous_mode and not done:
                state = agent.get_state(game)
                action = [0, 0, 0]
                current_dir = game.snake_dir
                left_turn = (-current_dir[1], current_dir[0])
                right_turn = (current_dir[1], -current_dir[0])
                if game.snake_dir == current_dir:
                    action = [1, 0, 0]
                elif game.snake_dir == right_turn:
                    action = [0, 1, 0]
                elif game.snake_dir == left_turn:
                    action = [0, 0, 1]

                reward, done, score = game.move()
                if sum(action) == 1:
                    manual_data.append((state, action))
                    next_state = agent.get_state(game)
                    action_idx = action.index(1)
                    agent.train_short_memory(state, action_idx, reward, next_state, done)
                    agent.remember(state, action_idx, reward, next_state, done)
                if not game.is_danger_straight():
                    successful_moves += 1

                if manual_movements == 1000:
                    try:
                        agent.pretrain(manual_data)
                        agent.save("manual_snake_model.pth")
                        show_popup = True
                    except Exception as e:
                        print(f"Error during pretraining: {e}. Continuing game...")
                        show_popup = True
                    manual_data = []
            elif autonomous_mode:
                state = agent.get_state(game)
                if training_mode:
                    action = agent.get_action(state, game, training=True)
                    reward, done, score = game.play_step([1 if i == action else 0 for i in range(3)])
                    agent.food_history.append(game.food)
                    agent.train_short_memory(state, action, reward, agent.get_state(game), done)
                    agent.remember(state, action, reward, agent.get_state(game), done)
                    if reward > 0:  # Food eaten
                        if used_manual_model:
                            used_manual_model = False
                            agent.save("snake_ai_model.pth")
                            print("✅ Switched to autonomous training after finding food, epsilon =", agent.epsilon)
                        agent.predict_next_food(game)
                    if done:
                        agent.train_long_memory()
                        agent.n_games += 1
                        total_score += score
                        mean_score = total_score / agent.n_games
                        print(f"Game {agent.n_games}, Score: {score}, Record: {agent.record}, Mean Score: {mean_score}, Epsilon: {agent.epsilon}")
                        if score > agent.record:
                            agent.record = score
                            agent.save("snake_ai_model.pth")
                        game.reset_game()
                        agent.predict_next_food(game)
                else:
                    action = agent.get_action(state, game, training=False)
                    reward, done, score = game.play_step([1 if i == action else 0 for i in range(3)])
                    if done:
                        game.reset_game()
                        agent.predict_next_food(game)

            if done and not autonomous_mode:
                game_over = True
                total_score += score
                print(f"Game Over - Score: {score}")

            if not show_popup:
                accuracy = (successful_moves / manual_movements * 100) if manual_movements > 0 else 0
                learning_progress = (agent.record / max(1, score) * 100) if agent.n_games > 0 else 0
                # Compute predicted action toward predicted food (only for autonomous mode)
                predicted_action = agent.predict_action_to_food(game, agent.predicted_food) if autonomous_mode and agent.predicted_food else None
                # Pass autonomous_mode to update_display
                update_display(game, screen, font, SIDE_PANEL_WIDTH, manual_movements, accuracy, learning_progress, 
                              agent.predicted_food if autonomous_mode else None, 
                              predicted_action, autonomous_mode)
                clock.tick(game.speed if hasattr(game, 'speed') else FPS)

        else:
            restart_button, exit_button = draw_game_over(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)
            pygame.display.flip()
            events = pygame.event.get()  # Get events once
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            action = handle_game_over_input(restart_button, exit_button, events)  # Use the same events
            if action == "restart":
                game.reset_game()
                game_over = False
                if autonomous_mode:
                    agent.predict_next_food(game)
                else:
                    agent.predicted_food = None  # Clear predicted food in manual mode
                if autonomous_mode and training_mode:
                    agent.n_games += 1
                    manual_movements = 0
                    successful_moves = 0
            elif action == "exit":
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()