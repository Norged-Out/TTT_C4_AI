"""
Author: Priyansh Nayak
Description: Pygame UI to play Connect 4
"""

import time

from src.games.connect4.game import Connect4


MODES = [
    "Human",
    "Default",
    "Minimax",
    "Alpha Beta",
    "Q-learning",
    "DQN",
]

SEARCH_DEPTH = 5
DROP_SPEED = 900


def reset_game(state):
    state["game"] = Connect4()
    state["winner_text"] = ""
    state["status"] = "Game reset."
    state["hover_col"] = None
    state["pending_move"] = None
    state["drop_token"] = None
    state["train_progress"] = None
    state["game_started"] = False
    state["expanded_dropdown"] = None
    state["last_ai_move_time"] = 0.0


def handle_game_end(state):
    winner = state["game"].winner

    if winner == "Draw":
        state["winner_text"] = "Draw"
        return

    label = "Player 1" if winner == "X" else "Player 2"
    state["winner_text"] = f"{label} wins"


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


def get_current_mode(state):
    if state["game"].current_player == "X":
        return state["player1_mode"]

    return state["player2_mode"]


def get_ai_move(state, mode):
    if mode == "Human":
        raise ValueError("Human mode does not choose moves.")

    if mode == "Default":
        from src.agents.connect4.default_opponent import choose_default_move
        return choose_default_move(state["game"]), None

    if mode == "Minimax":
        from src.agents.connect4.minimax import choose_minimax_move_limited
        return choose_minimax_move_limited(state["game"], depth_limit=SEARCH_DEPTH)

    if mode == "Alpha Beta":
        from src.agents.connect4.alphabeta import choose_alphabeta_move_limited
        return choose_alphabeta_move_limited(state["game"], depth_limit=SEARCH_DEPTH)

    if mode == "Q-learning":
        from src.agents.connect4.q_learning import choose_q_move
        return choose_q_move(state["game"], state["q_table"]), None

    if mode == "DQN":
        from src.agents.connect4.dqn import choose_dqn_move
        return choose_dqn_move(state["game"], state["dqn_model"]), None

    raise ValueError(f"Unknown mode: {mode}")


def ensure_agent_ready_for_mode(state, mode, render_callback, pygame):
    if mode == "Q-learning" and state["q_table"] is None:
        from src.agents.connect4.q_learning import train_q_learning

        state["status"] = "Loading Q-learning..."
        state["train_progress"] = None
        render_callback()

        def progress(done, total):
            state["train_progress"] = int((done / total) * 100)
            state["status"] = f"Loading Q-learning... {state['train_progress']}%"
            pygame.event.pump()
            render_callback()

        state["q_table"] = train_q_learning(progress_callback=progress)
        state["status"] = "Q-learning loaded."
        state["train_progress"] = None
        render_callback()
        return

    if mode == "DQN" and state["dqn_model"] is None:
        from src.agents.connect4.dqn import train_dqn

        state["status"] = "Loading DQN..."
        state["train_progress"] = None
        render_callback()

        def progress(done, total):
            state["train_progress"] = int((done / total) * 100)
            state["status"] = f"Loading DQN... {state['train_progress']}%"
            pygame.event.pump()
            render_callback()

        state["dqn_model"] = train_dqn(progress_callback=progress)
        state["status"] = "DQN loaded."
        state["train_progress"] = None
        render_callback()


def draw_board(screen, pygame, board_rect, game, hover_col, show_hover):
    bg = (245, 245, 245)
    board_blue = (40, 90, 185)
    hole = (230, 235, 245)
    x_color = (200, 80, 40)
    o_color = (240, 210, 70)
    grid = (30, 30, 30)

    pygame.draw.rect(screen, bg, board_rect)

    x0, y0, cell_size = get_board_layout(board_rect)
    total_w = cell_size * Connect4.COLS

    top_rect = pygame.Rect(x0, y0, total_w, cell_size)
    pygame.draw.rect(screen, (235, 235, 235), top_rect)
    pygame.draw.rect(screen, grid, top_rect, 3)

    if show_hover and hover_col is not None and hover_col in game.available_moves():
        cx = x0 + hover_col * cell_size + cell_size // 2
        cy = y0 + cell_size // 2
        color = x_color if game.current_player == "X" else o_color
        pygame.draw.circle(screen, color, (cx, cy), int(cell_size * 0.34))

    board_surface = pygame.Rect(x0, y0 + cell_size, total_w, cell_size * Connect4.ROWS)
    pygame.draw.rect(screen, board_blue, board_surface, border_radius=12)
    pygame.draw.rect(screen, grid, board_surface, 4, border_radius=12)

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


def draw_drop_token(screen, pygame, board_rect, drop_token):
    if drop_token is None:
        return

    x0, y0, cell_size = get_board_layout(board_rect)
    cx = x0 + drop_token["col"] * cell_size + cell_size // 2
    cy = int(drop_token["y"])
    radius = int(cell_size * 0.36)
    color = (200, 80, 40) if drop_token["player"] == "X" else (240, 210, 70)
    grid = (30, 30, 30)

    pygame.draw.circle(screen, color, (cx, cy), radius)
    pygame.draw.circle(screen, grid, (cx, cy), radius, 3)


def start_drop_animation(state, col):
    row = state["game"].get_drop_row(col)
    if row is None:
        state["status"] = "That column is full."
        return False

    x0, y0, cell_size = get_board_layout(state["board_rect"])
    start_y = y0 + cell_size // 2
    target_y = y0 + (row + 1) * cell_size + cell_size // 2

    state["pending_move"] = col
    state["drop_token"] = {
        "player": state["game"].current_player,
        "col": col,
        "y": float(start_y),
        "target_y": float(target_y),
    }
    return True


def finish_drop_animation(state):
    if state["pending_move"] is None:
        return

    col = state["pending_move"]
    player = state["game"].current_player
    state["game"].make_move(col)
    state["pending_move"] = None
    state["drop_token"] = None

    if state["game"].winner is not None:
        handle_game_end(state)
        return

    label = "Player 1" if player == "X" else "Player 2"
    state["status"] = f"{label} played column {col + 1}."


def draw_dropdown(screen, pygame, rect, value, options, expanded, fonts):
    pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=8)
    pygame.draw.rect(screen, (110, 110, 110), rect, 2, border_radius=8)

    label = fonts["small"].render(value, True, (245, 245, 245))
    screen.blit(label, (rect.left + 12, rect.top + 8))

    arrow = fonts["small"].render("v", True, (220, 220, 220))
    arrow_rect = arrow.get_rect(center=(rect.right - 18, rect.centery))
    screen.blit(arrow, arrow_rect)

    if not expanded:
        return

    option_y = rect.bottom + 6
    for option in options:
        option_rect = pygame.Rect(rect.left, option_y, rect.width, rect.height)
        pygame.draw.rect(screen, (60, 60, 60), option_rect, border_radius=8)
        pygame.draw.rect(screen, (95, 95, 95), option_rect, 1, border_radius=8)
        option_label = fonts["small"].render(option, True, (245, 245, 245))
        screen.blit(option_label, (option_rect.left + 12, option_rect.top + 8))
        option_y += rect.height


def draw_sidebar(screen, pygame, sidebar_rect, state, fonts, ui_rects):
    pygame.draw.rect(screen, (35, 35, 35), sidebar_rect)

    title = fonts["title"].render("Connect 4", True, (245, 245, 245))
    screen.blit(title, (sidebar_rect.left + 20, 24))

    p1_text = fonts["small"].render("Player 1", True, (220, 220, 220))
    p2_text = fonts["small"].render("Player 2", True, (220, 220, 220))
    screen.blit(p1_text, (sidebar_rect.left + 20, 96))
    screen.blit(p2_text, (sidebar_rect.left + 20, 220))

    draw_dropdown(screen, pygame, ui_rects["player1_dropdown"], state["player1_mode"], MODES, False, fonts)
    draw_dropdown(screen, pygame, ui_rects["player2_dropdown"], state["player2_mode"], MODES, False, fonts)

    for key in ["Start", "Reset"]:
        rect = ui_rects[key]
        color = (70, 120, 210) if key == "Start" else (90, 90, 90)
        pygame.draw.rect(screen, color, rect, border_radius=8)
        label = fonts["small"].render(key, True, (245, 245, 245))
        label_rect = label.get_rect(center=rect.center)
        screen.blit(label, label_rect)

    y = 500
    if state["game_started"]:
        if state["game"].winner is None:
            current_label = "Player 1" if state["game"].current_player == "X" else "Player 2"
            turn_text = fonts["body"].render(f"Turn: {current_label}", True, (230, 230, 230))
        else:
            turn_text = fonts["body"].render(f"Result: {state['winner_text']}", True, (230, 230, 230))
    else:
        turn_text = fonts["body"].render("Press Start", True, (230, 230, 230))
    screen.blit(turn_text, (sidebar_rect.left + 20, y))
    y += 40

    status_text = fonts["small"].render(state["status"], True, (180, 180, 180))
    screen.blit(status_text, (sidebar_rect.left + 20, y))
    y += 28

    if state["train_progress"] is not None:
        progress_text = fonts["small"].render(
            f"Loading agent: {state['train_progress']}%",
            True,
            (200, 200, 120),
        )
        screen.blit(progress_text, (sidebar_rect.left + 20, y))
        y += 28

    hint1 = fonts["small"].render("Click top row to drop a token.", True, (190, 190, 190))
    hint2 = fonts["small"].render("Press R to reset.", True, (190, 190, 190))
    screen.blit(hint1, (sidebar_rect.left + 20, y))
    y += 28
    screen.blit(hint2, (sidebar_rect.left + 20, y))

    if state["expanded_dropdown"] == "player1":
        draw_dropdown(screen, pygame, ui_rects["player1_dropdown"], state["player1_mode"], MODES, True, fonts)
    elif state["expanded_dropdown"] == "player2":
        draw_dropdown(screen, pygame, ui_rects["player2_dropdown"], state["player2_mode"], MODES, True, fonts)


def option_rects_from_dropdown(dropdown_rect):
    rects = []
    option_y = dropdown_rect.bottom + 6
    rect_type = type(dropdown_rect)
    for _ in MODES:
        rects.append(rect_type(dropdown_rect.left, option_y, dropdown_rect.width, dropdown_rect.height))
        option_y += dropdown_rect.height
    return rects


def run_game():
    import pygame

    pygame.init()

    state = {
        "player1_mode": "Human",
        "player2_mode": "Human",
        "expanded_dropdown": None,
        "game_started": False,
        "last_ai_move_time": 0.0,
        "game": Connect4(),
        "winner_text": "",
        "status": "Ready.",
        "hover_col": None,
        "pending_move": None,
        "drop_token": None,
        "train_progress": None,
        "q_table": None,
        "dqn_model": None,
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
    state["board_rect"] = board_rect

    ui_rects = {
        "player1_dropdown": pygame.Rect(20, 124, sidebar_w - 40, 38),
        "player2_dropdown": pygame.Rect(20, 248, sidebar_w - 40, 38),
        "Start": pygame.Rect(20, 360, sidebar_w - 40, 40),
        "Reset": pygame.Rect(20, 412, sidebar_w - 40, 40),
    }

    clock = pygame.time.Clock()
    running = True

    def render_frame():
        screen.fill((20, 20, 20))
        draw_sidebar(screen, pygame, sidebar_rect, state, fonts, ui_rects)
        show_hover = (
            state["game_started"]
            and state["game"].winner is None
            and state["drop_token"] is None
            and get_current_mode(state) == "Human"
        )
        draw_board(screen, pygame, board_rect, state["game"], state["hover_col"], show_hover)
        draw_drop_token(screen, pygame, board_rect, state["drop_token"])
        pygame.display.flip()

    while running:
        now = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_game(state)

            elif event.type == pygame.MOUSEMOTION:
                if state["drop_token"] is None:
                    state["hover_col"] = get_hover_column(event.pos, board_rect)
                else:
                    state["hover_col"] = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos

                if state["drop_token"] is not None:
                    continue

                handled_option = False
                for player_key in ["player1", "player2"]:
                    if state["expanded_dropdown"] != player_key:
                        continue

                    dropdown_rect = ui_rects[f"{player_key}_dropdown"]
                    option_rects = option_rects_from_dropdown(dropdown_rect)

                    for mode, rect in zip(MODES, option_rects):
                        if not rect.collidepoint(mouse_pos):
                            continue

                        state[f"{player_key}_mode"] = mode
                        state["status"] = f"{player_key.title()} set to {mode}."
                        state["expanded_dropdown"] = None
                        handled_option = True
                        break

                if handled_option:
                    continue

                if ui_rects["Start"].collidepoint(mouse_pos):
                    reset_game(state)
                    ensure_agent_ready_for_mode(state, state["player1_mode"], render_frame, pygame)
                    ensure_agent_ready_for_mode(state, state["player2_mode"], render_frame, pygame)
                    state["game_started"] = True
                    state["status"] = "Match started."
                    state["last_ai_move_time"] = now
                    continue

                if ui_rects["Reset"].collidepoint(mouse_pos):
                    reset_game(state)
                    continue

                if ui_rects["player1_dropdown"].collidepoint(mouse_pos):
                    state["expanded_dropdown"] = None if state["expanded_dropdown"] == "player1" else "player1"
                    continue

                if ui_rects["player2_dropdown"].collidepoint(mouse_pos):
                    state["expanded_dropdown"] = None if state["expanded_dropdown"] == "player2" else "player2"
                    continue

                state["expanded_dropdown"] = None

                if not state["game_started"] or state["game"].winner is not None:
                    continue

                if get_current_mode(state) != "Human":
                    continue

                col = get_hover_column(mouse_pos, board_rect)
                if col is None:
                    continue

                if not start_drop_animation(state, col):
                    continue
                state["last_ai_move_time"] = now

        if state["drop_token"] is not None:
            state["drop_token"]["y"] += DROP_SPEED * clock.get_time() / 1000.0
            if state["drop_token"]["y"] >= state["drop_token"]["target_y"]:
                finish_drop_animation(state)
                state["last_ai_move_time"] = time.perf_counter()
            render_frame()
            clock.tick(60)
            continue

        if not state["game_started"] or state["game"].winner is not None:
            render_frame()
            clock.tick(60)
            continue

        current_mode = get_current_mode(state)
        if current_mode == "Human":
            render_frame()
            clock.tick(60)
            continue

        move, stats = get_ai_move(state, current_mode)
        if start_drop_animation(state, move):
            if stats is not None:
                state["status"] = f"{current_mode} picked column {move + 1} ({stats['elapsed_seconds']:.2f}s)"
            else:
                state["status"] = f"{current_mode} picked column {move + 1}."
            state["last_ai_move_time"] = time.perf_counter()

        render_frame()
        clock.tick(60)

    pygame.quit()
