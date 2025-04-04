import pygame

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

def draw_wire_boundary(surface, grid_width, grid_height, cell_size):
    """Draws a wire-like boundary around the playable area."""
    pygame.draw.rect(surface, WHITE, (0, 0, grid_width * cell_size, grid_height * cell_size), 5)

def draw_snake(surface, snake):
    """Draw the snake."""
    for segment in snake:
        pygame.draw.rect(surface, GREEN, (segment[0], segment[1], 20, 20))

def draw_food(surface, food):
    """Draw the food."""
    pygame.draw.rect(surface, RED, (food[0], food[1], 20, 20))

def draw_side_panel(surface, font, score, screen_width, panel_width, movements, accuracy, learning_progress, predicted_food=None, predicted_action=None, game=None, autonomous_mode=False):
    """Draw the UI panel showing the score and additional metrics."""
    panel_x = screen_width - panel_width + 10
    pygame.draw.rect(surface, BLACK, (screen_width - panel_width, 0, panel_width, surface.get_height()))
    score_text = font.render(f"Score: {score}", True, WHITE)
    movements_text = font.render(f"Movements: {movements}", True, WHITE)
    accuracy_text = font.render(f"Accuracy: {accuracy:.1f}%", True, WHITE)
    progress_text = font.render(f"Learning: {learning_progress:.1f}%", True, WHITE)
    surface.blit(score_text, (panel_x, 20))
    surface.blit(movements_text, (panel_x, 60))
    surface.blit(accuracy_text, (panel_x, 100))
    surface.blit(progress_text, (panel_x, 140))

    # Display predicted food info and action only in autonomous mode
    if autonomous_mode and predicted_food and game:
        row = predicted_food[1] // game.cell_size
        col = predicted_food[0] // game.cell_size
        pred_pos_text = font.render(f"Pred Food: ({row}, {col})", True, WHITE)
        surface.blit(pred_pos_text, (panel_x, 180))

        if predicted_action is not None:
            action_str = {0: "Straight", 1: "Right", 2: "Left"}.get(predicted_action, "None")
            pred_action_text = font.render(f"Pred Action: {action_str}", True, WHITE)
            surface.blit(pred_action_text, (panel_x, 220))

    # Add instruction text at the bottom of the side panel
    instruction_text = font.render(" press ESC to exit ", True, WHITE)
    surface.blit(instruction_text, (panel_x, surface.get_height() - 40))

def draw_popup_message(surface, font, screen_width, screen_height, message):
    """Draw a pop-up message with Continue and Exit buttons."""
    overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    popup_width = 500
    popup_height = 250
    popup_x = (screen_width - popup_width) // 2
    popup_y = (screen_height - popup_height) // 2
    pygame.draw.rect(surface, WHITE, (popup_x, popup_y, popup_width, popup_height), border_radius=10)

    message_lines = message.split('\n')
    line_height = font.get_linesize()
    total_text_height = len(message_lines) * line_height
    text_start_y = popup_y + (popup_height - total_text_height - 60) // 2

    for i, line in enumerate(message_lines):
        message_text = font.render(line, True, BLACK)
        message_rect = message_text.get_rect(center=(screen_width // 2, text_start_y + i * line_height))
        surface.blit(message_text, message_rect.topleft)

    button_width = 100
    button_height = 40
    button_y = popup_y + popup_height - 80
    continue_button = pygame.Rect((screen_width // 2 - 120, button_y, button_width, button_height))
    exit_button = pygame.Rect((screen_width // 2 + 20, button_y, button_width, button_height))

    pygame.draw.rect(surface, GREEN, continue_button, border_radius=5)
    pygame.draw.rect(surface, RED, exit_button, border_radius=5)

    continue_text = font.render("Continue", True, BLACK)
    exit_text = font.render("Exit", True, BLACK)
    surface.blit(continue_text, continue_text.get_rect(center=continue_button.center).topleft)
    surface.blit(exit_text, exit_text.get_rect(center=exit_button.center).topleft)

    return continue_button, exit_button

def draw_game_over(surface, font, screen_width, screen_height):
    """Draws the Game Over screen with Restart and Exit buttons, centered properly."""
    surface.fill(BLACK)

    game_over_text = font.render("GAME OVER", True, RED)
    game_over_rect = game_over_text.get_rect(center=(screen_width // 2, screen_height // 2 - 100))
    surface.blit(game_over_text, game_over_rect.topleft)

    button_width = 200
    button_height = 60
    button_x = (screen_width // 2) - (button_width // 2)
    button_y = screen_height // 2

    restart_button = pygame.Rect(button_x, button_y, button_width, button_height)
    exit_button = pygame.Rect(button_x, button_y + 80, button_width, button_height)

    pygame.draw.rect(surface, WHITE, restart_button)
    pygame.draw.rect(surface, WHITE, exit_button)

    restart_text = font.render("Restart", True, BLACK)
    exit_text = font.render("Exit", True, BLACK)
    
    surface.blit(restart_text, restart_text.get_rect(center=restart_button.center).topleft)
    surface.blit(exit_text, exit_text.get_rect(center=exit_button.center).topleft)

    return restart_button, exit_button

def draw_homepage(surface, font, screen_width, screen_height):
    """Draw the homepage with Play options centered dynamically."""
    surface.fill(BLACK)

    title_text = font.render("Classic Snake Game", True, WHITE)
    title_rect = title_text.get_rect(center=(screen_width // 2, screen_height // 3 - 50))
    surface.blit(title_text, title_rect.topleft)

    button_width = 250
    button_height = 60
    button_x = (screen_width // 2) - (button_width // 2)
    button_y = screen_height // 2 - 50

    play_manual_button = pygame.Rect(button_x, button_y, button_width, button_height)
    play_ai_button = pygame.Rect(button_x, button_y + 100, button_width, button_height)

    pygame.draw.rect(surface, WHITE, play_manual_button)
    pygame.draw.rect(surface, WHITE, play_ai_button)

    manual_text = font.render("Play Manually", True, BLACK)
    ai_text = font.render("Play Autonomously", True, BLACK)

    surface.blit(manual_text, manual_text.get_rect(center=play_manual_button.center).topleft)
    surface.blit(ai_text, ai_text.get_rect(center=play_ai_button.center).topleft)

    # Add instruction text in the bottom-right corner
    instruction_text = font.render("ESC to exit", True, WHITE)
    instruction_rect = instruction_text.get_rect(bottomright=(screen_width - 10, screen_height - 10))
    surface.blit(instruction_text, instruction_rect.topleft)

    return play_manual_button, play_ai_button

def initialize_display(width, height, panel_width):
    """Initialize and return the Pygame display with a resizable window."""
    screen = pygame.display.set_mode((width + panel_width, height), pygame.RESIZABLE)
    pygame.display.set_caption('Snake Game')
    return screen

def update_display(game, surface, font, panel_width, movements, accuracy, learning_progress, predicted_food=None, predicted_action=None, autonomous_mode=False):
    """Update the entire display based on game state."""
    surface.fill(BLACK)
    draw_wire_boundary(surface, game.grid_width, game.grid_height, game.cell_size)
    draw_snake(surface, game.snake)
    draw_food(surface, game.food)

    # Draw predicted food location with row and column lines only in autonomous mode
    if autonomous_mode and predicted_food:
        pred_x, pred_y = predicted_food
        cell_size = game.cell_size
        grid_width = game.grid_width * cell_size
        grid_height = game.grid_height * cell_size
        num_dots = 10  # Same as previous dotted line

        # Horizontal line (row)
        for i in range(grid_width // (cell_size // 2)):
            x = i * (cell_size // 2)
            if i % 2 == 0:  # Create dotted effect
                pygame.draw.circle(surface, BLUE, (x, pred_y + cell_size // 2), 2)

        # Vertical line (column)
        for i in range(grid_height // (cell_size // 2)):
            y = i * (cell_size // 2)
            if i % 2 == 0:  # Create dotted effect
                pygame.draw.circle(surface, BLUE, (pred_x + cell_size // 2, y), 2)

    # Draw side panel
    draw_side_panel(surface, font, game.score, game.grid_width * game.cell_size + game.cell_size + panel_width,
                    panel_width, movements, accuracy, learning_progress, predicted_food, predicted_action, game, autonomous_mode)

    pygame.display.flip()

def handle_homepage_input(surface, font, play_manual_button, play_ai_button, events):
    """Handle homepage button clicks and return mode."""
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if play_manual_button.collidepoint(mouse_pos):
                return "manual"
            elif play_ai_button.collidepoint(mouse_pos):
                return "ai"
    return None

def handle_game_over_input(restart_button, exit_button, events):
    """Handle game over screen button clicks and return action."""
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if restart_button.collidepoint(mouse_pos):
                return "restart"
            elif exit_button.collidepoint(mouse_pos):
                return "exit"
    return None