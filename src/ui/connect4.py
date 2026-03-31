"""
Author: Priyansh Nayak
Description: Pygame UI to play Connect 4
"""

from src.agents.connect4.default_opponent import choose_default_move
from src.games.connect4.game import Connect4


MODES = [
    "Two Player",
    "Default",
    "Minimax",
    "Alpha Beta",
    "Q-learning",
    "DQN",
]


def reset_game(state):
    state["game"] = Connect4()
    state["winner_text"] = ""
    state["status"] = "Game reset."
    state["hover_col"] = None


def handle_game_end(state):
    winner = state["game"].winner

    if winner == "Draw":
        state["winner_text"] = "Draw"
        return

    state["winner_text"] = f"{winner} wins"


def get_board_layout(board_rect):
    cell_size = min(board_rect.width // Connect4.COLS, board_rect.height // (Connect4.ROWS + 1))
    total_w = cell_size * Connect4.COLS
    total_h = cell_size * (Connect4.ROWS + 1)
    x0 = board_rect.left + (board_rect.width - total_w) // 2
    y0 = board_rect.top + (board_rect.height - total_h) // 2
    return x0, y0, cell_size


def get_hover_column(mouse_pos, board_rect):
    x0, y0, cell_size = get_board_layout(board_rect)
    mx, my = mouse_pos

    if not (x0 <= mx < x0 + Connect4.COLS * cell_size and y0 <= my < y0 + cell_size):
        return None

    return int((mx - x0) // cell_size)


def draw_board(screen, pygame, board_rect, game, hover_col):
    bg = (245, 245, 245)
    board_blue = (40, 90, 185)
    hole = (230, 235, 245)
    x_color = (200, 80, 40)
    o_color = (240, 210, 70)
    grid = (30, 30, 30)

    pygame.draw.rect(screen, bg, board_rect)

    x0, y0, cell_size = get_board_layout(board_rect)
    total_w = cell_size * Connect4.COLS

    # top click area
    top_rect = pygame.Rect(x0, y0, total_w, cell_size)
    pygame.draw.rect(screen, (235, 235, 235), top_rect)
    pygame.draw.rect(screen, grid, top_rect, 3)

    if hover_col is not None and hover_col in game.available_moves():
        cx = x0 + hover_col * cell_size + cell_size // 2
        cy = y0 + cell_size // 2
        color = x_color if game.current_player == "X" else o_color
        pygame.draw.circle(screen, color, (cx, cy), int(cell_size * 0.34))

    # main blue board
    board_surface = pygame.Rect(x0, y0 + cell_size, total_w, cell_size * Connect4.ROWS)
    pygame.draw.rect(screen, board_blue, board_surface, border_radius=12)
    pygame.draw.rect(screen, grid, board_surface, 4, border_radius=12)

    # board slots
    for row in range(Connect4.ROWS):
        for col in range(Connect4.COLS):
            cx = x0 + col * cell_size + cell_size // 2
            cy = y0 + (row + 1) * cell_size + cell_size // 2
            radius = int(cell_size * 0.36)

            marker = game.board[row][col]
            if marker == "X":
                color = x_color
            elif marker == "O":
                color = o_color
            else:
                color = hole

            pygame.draw.circle(screen, color, (cx, cy), radius)
            pygame.draw.circle(screen, grid, (cx, cy), radius, 3)


def draw_sidebar(screen, pygame, sidebar_rect, state, fonts, button_rects):
    pygame.draw.rect(screen, (35, 35, 35), sidebar_rect)

    y = 24

    title = fonts["title"].render("Connect 4", True, (245, 245, 245))
    screen.blit(title, (sidebar_rect.left + 20, y))
    y += 52

    mode_text = fonts["body"].render(f"Mode: {state['mode']}", True, (230, 230, 230))
    screen.blit(mode_text, (sidebar_rect.left + 20, y))
    y += 34

    if state["game"].winner is None:
        turn_text = fonts["body"].render(f"Turn: {state['game'].current_player}", True, (230, 230, 230))
    else:
        turn_text = fonts["body"].render(f"Result: {state['winner_text']}", True, (230, 230, 230))
    screen.blit(turn_text, (sidebar_rect.left + 20, y))
    y += 34

    status_text = fonts["small"].render(state["status"], True, (180, 180, 180))
    screen.blit(status_text, (sidebar_rect.left + 20, y))
    y += 64

    hint1 = fonts["small"].render("Click top row to drop a token.", True, (190, 190, 190))
    hint2 = fonts["small"].render("Press R to reset.", True, (190, 190, 190))
    screen.blit(hint1, (sidebar_rect.left + 20, y))
    y += 40
    screen.blit(hint2, (sidebar_rect.left + 20, y))

    for key, rect in button_rects.items():
        if key == state["mode"]:
            color = (70, 120, 210)
        elif key == "Reset":
            color = (90, 90, 90)
        else:
            color = (70, 70, 70)

        pygame.draw.rect(screen, color, rect, border_radius=8)
        label = fonts["small"].render(key, True, (245, 245, 245))
        label_rect = label.get_rect(center=rect.center)
        screen.blit(label, label_rect)


def run_game():
    import pygame

    pygame.init()

    state = {
        "mode": "Two Player",
        "game": Connect4(),
        "winner_text": "",
        "status": "Ready.",
        "hover_col": None,
    }

    screen_w = 1100
    screen_h = 780
    sidebar_w = 280

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Connect 4")

    fonts = {
        "title": pygame.font.SysFont("arial", 28, bold=True),
        "body": pygame.font.SysFont("arial", 22),
        "small": pygame.font.SysFont("arial", 18),
    }

    sidebar_rect = pygame.Rect(0, 0, sidebar_w, screen_h)
    board_rect = pygame.Rect(sidebar_w, 0, screen_w - sidebar_w, screen_h)

    button_rects = {}
    y = 300
    for mode in MODES:
        button_rects[mode] = pygame.Rect(20, y, sidebar_w - 40, 42)
        y += 54

    button_rects["Reset"] = pygame.Rect(20, y + 10, sidebar_w - 40, 42)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_game(state)

            elif event.type == pygame.MOUSEMOTION:
                state["hover_col"] = get_hover_column(event.pos, board_rect)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handled = False
                for key, rect in button_rects.items():
                    if rect.collidepoint(event.pos):
                        handled = True
                        if key == "Reset":
                            reset_game(state)
                            break

                        if key not in {"Two Player", "Default"}:
                            state["status"] = f"{key} not added yet."
                            break

                        state["mode"] = key
                        reset_game(state)
                        state["status"] = f"{key} selected."
                        break

                if handled:
                    continue

                if state["game"].winner is not None:
                    continue

                if state["mode"] != "Two Player" and state["game"].current_player != "X":
                    continue

                col = get_hover_column(event.pos, board_rect)
                if col is None:
                    continue

                if not state["game"].make_move(col):
                    state["status"] = "That column is full."
                    continue

                state["status"] = f"Played column {col + 1}."
                if state["game"].winner is not None:
                    handle_game_end(state)

        if state["mode"] == "Default" and state["game"].winner is None and state["game"].current_player == "O":
            move = choose_default_move(state["game"])
            if state["game"].make_move(move):
                state["status"] = f"Default played column {move + 1}."
                if state["game"].winner is not None:
                    handle_game_end(state)

        screen.fill((20, 20, 20))
        draw_sidebar(screen, pygame, sidebar_rect, state, fonts, button_rects)
        draw_board(screen, pygame, board_rect, state["game"], state["hover_col"])
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
