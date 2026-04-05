"""
Author: Priyansh Nayak
Description: Pygame UI to play Tic Tac Toe
"""

import time

from src.games.tictactoe.game import TicTacToe


MODES = [
    "Human",
    "Default",
    "Minimax",
    "Alpha Beta",
    "Q-learning",
    "DQN",
]

AI_DELAY_SECONDS = 0.4


def get_ai_move(game, mode, state):
    # choose move from selected agent
    if mode == "Human":
        raise ValueError("Human mode does not choose moves.")

    if mode == "Default":
        from src.agents.tictactoe.default_opponent import choose_default_move
        return choose_default_move(game)

    if mode == "Minimax":
        from src.agents.tictactoe.minimax import choose_minimax_move
        return choose_minimax_move(game)

    if mode == "Alpha Beta":
        from src.agents.tictactoe.alphabeta import choose_alphabeta_move
        return choose_alphabeta_move(game)

    if mode == "Q-learning":
        from src.agents.tictactoe.q_learning import choose_q_move
        return choose_q_move(game, state["q_table"])

    if mode == "DQN":
        from src.agents.tictactoe.dqn import choose_dqn_move
        return choose_dqn_move(game, state["dqn_model"])

    raise ValueError(f"Unknown mode: {mode}")


def reset_game(state):
    # reset match state
    state["game"] = TicTacToe()
    state["winner_text"] = ""
    state["status"] = "Game reset."
    state["train_progress"] = None
    state["game_started"] = False
    state["last_ai_move_time"] = 0.0


def handle_game_end(state):
    # build end text
    winner = state["game"].winner

    if winner == "Draw":
        state["winner_text"] = "Draw"
        return

    label = "Player 1" if winner == "X" else "Player 2"
    state["winner_text"] = f"{label} wins"


def get_board_layout(board_rect):
    # fit board inside area
    cell_size = min(board_rect.width, board_rect.height) // 3
    total_size = cell_size * 3
    x0 = board_rect.left + (board_rect.width - total_size) // 2
    y0 = board_rect.top + (board_rect.height - total_size) // 2
    return x0, y0, cell_size


def get_clicked_cell(mouse_pos, board_rect):
    # turn click into cell index
    x0, y0, cell_size = get_board_layout(board_rect)
    mx, my = mouse_pos

    if not (x0 <= mx < x0 + 3 * cell_size and y0 <= my < y0 + 3 * cell_size):
        return None

    col = (mx - x0) // cell_size
    row = (my - y0) // cell_size
    return int(row * 3 + col)


def get_current_mode(state):
    # active side for current turn
    if state["game"].current_player == "X":
        return state["player1_mode"]

    return state["player2_mode"]


def draw_board(screen, pygame, board_rect, game, fonts):
    # board colors
    bg = (245, 245, 245)
    grid = (30, 30, 30)
    x_color = (30, 80, 180)
    o_color = (200, 80, 40)

    # board background
    pygame.draw.rect(screen, bg, board_rect)

    x0, y0, cell_size = get_board_layout(board_rect)
    total_size = cell_size * 3

    # outer border
    pygame.draw.rect(screen, grid, (x0, y0, total_size, total_size), 4)

    # grid lines
    for i in range(1, 3):
        pygame.draw.line(
            screen,
            grid,
            (x0 + i * cell_size, y0),
            (x0 + i * cell_size, y0 + 3 * cell_size),
            4,
        )
        pygame.draw.line(
            screen,
            grid,
            (x0, y0 + i * cell_size),
            (x0 + 3 * cell_size, y0 + i * cell_size),
            4,
        )

    # marks
    for index, value in enumerate(game.board):
        row = index // 3
        col = index % 3
        cx = x0 + col * cell_size + cell_size // 2
        cy = y0 + row * cell_size + cell_size // 2
        half = cell_size // 3

        if value == "X":
            pygame.draw.line(screen, x_color, (cx - half, cy - half), (cx + half, cy + half), 12)
            pygame.draw.line(screen, x_color, (cx - half, cy + half), (cx + half, cy - half), 12)
        elif value == "O":
            pygame.draw.circle(screen, o_color, (cx, cy), half, 12)
        else:
            label = fonts["small"].render(str(index + 1), True, (160, 160, 160))
            label_rect = label.get_rect(center=(cx, cy))
            screen.blit(label, label_rect)


def draw_dropdown(screen, pygame, rect, value, options, expanded, fonts):
    # closed box
    pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=8)
    pygame.draw.rect(screen, (110, 110, 110), rect, 2, border_radius=8)

    label = fonts["small"].render(value, True, (245, 245, 245))
    screen.blit(label, (rect.left + 12, rect.top + 8))

    arrow = fonts["small"].render("v", True, (220, 220, 220))
    arrow_rect = arrow.get_rect(center=(rect.right - 18, rect.centery))
    screen.blit(arrow, arrow_rect)

    if not expanded:
        return

    # options list
    option_y = rect.bottom + 6
    for option in options:
        option_rect = pygame.Rect(rect.left, option_y, rect.width, rect.height)
        pygame.draw.rect(screen, (60, 60, 60), option_rect, border_radius=8)
        pygame.draw.rect(screen, (95, 95, 95), option_rect, 1, border_radius=8)
        option_label = fonts["small"].render(option, True, (245, 245, 245))
        screen.blit(option_label, (option_rect.left + 12, option_rect.top + 8))
        option_y += rect.height


def draw_sidebar(screen, pygame, sidebar_rect, state, fonts, ui_rects):
    # sidebar background
    pygame.draw.rect(screen, (35, 35, 35), sidebar_rect)

    # title
    title = fonts["title"].render("Tic Tac Toe", True, (245, 245, 245))
    screen.blit(title, (sidebar_rect.left + 20, 24))

    # player labels
    p1_text = fonts["small"].render("Player 1", True, (220, 220, 220))
    p2_text = fonts["small"].render("Player 2", True, (220, 220, 220))
    screen.blit(p1_text, (sidebar_rect.left + 20, 96))
    screen.blit(p2_text, (sidebar_rect.left + 20, 220))

    # closed dropdowns
    draw_dropdown(
        screen,
        pygame,
        ui_rects["player1_dropdown"],
        state["player1_mode"],
        MODES,
        False,
        fonts,
    )
    draw_dropdown(
        screen,
        pygame,
        ui_rects["player2_dropdown"],
        state["player2_mode"],
        MODES,
        False,
        fonts,
    )

    # buttons
    for key in ["Start", "Reset"]:
        rect = ui_rects[key]
        color = (70, 120, 210) if key == "Start" else (90, 90, 90)
        pygame.draw.rect(screen, color, rect, border_radius=8)
        label = fonts["small"].render(key, True, (245, 245, 245))
        label_rect = label.get_rect(center=rect.center)
        screen.blit(label, label_rect)

    # status text
    y = 420
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

    # load progress
    if state["train_progress"] is not None:
        progress_text = fonts["small"].render(
            f"Loading agent: {state['train_progress']}%",
            True,
            (200, 200, 120),
        )
        screen.blit(progress_text, (sidebar_rect.left + 20, y))
        y += 28

    # reset hint
    hint = fonts["small"].render("Press R to reset.", True, (190, 190, 190))
    screen.blit(hint, (sidebar_rect.left + 20, y))

    # expanded list on top
    if state["expanded_dropdown"] == "player1":
        draw_dropdown(
            screen,
            pygame,
            ui_rects["player1_dropdown"],
            state["player1_mode"],
            MODES,
            True,
            fonts,
        )
    elif state["expanded_dropdown"] == "player2":
        draw_dropdown(
            screen,
            pygame,
            ui_rects["player2_dropdown"],
            state["player2_mode"],
            MODES,
            True,
            fonts,
        )


def ensure_agent_ready_for_mode(state, mode, render_callback, pygame):
    if mode == "Q-learning" and state["q_table"] is None:
        from src.agents.tictactoe.q_learning import train_q_learning

        # load q table once
        state["status"] = "Loading Q-learning..."
        state["train_progress"] = None
        render_callback()

        def progress(done, total):
            # training progress
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
        from src.agents.tictactoe.dqn import train_dqn

        # load dqn once
        state["status"] = "Loading DQN..."
        state["train_progress"] = None
        render_callback()

        def progress(done, total):
            # training progress
            state["train_progress"] = int((done / total) * 100)
            state["status"] = f"Loading DQN... {state['train_progress']}%"
            pygame.event.pump()
            render_callback()

        state["dqn_model"] = train_dqn(progress_callback=progress)
        state["status"] = "DQN loaded."
        state["train_progress"] = None
        render_callback()


def option_rects_from_dropdown(dropdown_rect):
    # clickable option boxes
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

    # app state
    state = {
        "player1_mode": "Human",
        "player2_mode": "Human",
        "expanded_dropdown": None,
        "game_started": False,
        "last_ai_move_time": 0.0,
        "game": TicTacToe(),
        "winner_text": "",
        "status": "Ready.",
        "train_progress": None,
        "q_table": None,
        "dqn_model": None,
    }

    # window setup
    screen_w = 1000
    screen_h = 700
    sidebar_w = 280

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Tic Tac Toe")

    # fonts
    fonts = {
        "title": pygame.font.SysFont("arial", 28, bold=True),
        "body": pygame.font.SysFont("arial", 22),
        "small": pygame.font.SysFont("arial", 18),
    }

    # layout
    sidebar_rect = pygame.Rect(0, 0, sidebar_w, screen_h)
    board_rect = pygame.Rect(sidebar_w, 0, screen_w - sidebar_w, screen_h)

    # ui boxes
    ui_rects = {
        "player1_dropdown": pygame.Rect(20, 124, sidebar_w - 40, 38),
        "player2_dropdown": pygame.Rect(20, 248, sidebar_w - 40, 38),
        "Start": pygame.Rect(20, 540, sidebar_w - 40, 40),
        "Reset": pygame.Rect(20, 592, sidebar_w - 40, 40),
    }

    clock = pygame.time.Clock()
    running = True

    def render_frame():
        # draw one frame
        screen.fill((20, 20, 20))
        draw_sidebar(screen, pygame, sidebar_rect, state, fonts, ui_rects)
        draw_board(screen, pygame, board_rect, state["game"], fonts)
        pygame.display.flip()

    while running:
        now = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # keyboard shortcuts
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_game(state)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # mouse click
                mouse_pos = event.pos

                # dropdown choices
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

                # start button
                if ui_rects["Start"].collidepoint(mouse_pos):
                    reset_game(state)
                    ensure_agent_ready_for_mode(state, state["player1_mode"], render_frame, pygame)
                    ensure_agent_ready_for_mode(state, state["player2_mode"], render_frame, pygame)
                    state["game_started"] = True
                    state["status"] = "Match started."
                    state["expanded_dropdown"] = None
                    state["last_ai_move_time"] = now
                    continue

                # reset button
                if ui_rects["Reset"].collidepoint(mouse_pos):
                    reset_game(state)
                    state["expanded_dropdown"] = None
                    continue

                # player 1 menu
                if ui_rects["player1_dropdown"].collidepoint(mouse_pos):
                    state["expanded_dropdown"] = None if state["expanded_dropdown"] == "player1" else "player1"
                    continue

                # player 2 menu
                if ui_rects["player2_dropdown"].collidepoint(mouse_pos):
                    state["expanded_dropdown"] = None if state["expanded_dropdown"] == "player2" else "player2"
                    continue

                # close open menu
                state["expanded_dropdown"] = None

                # only allow human clicks during live game
                if not state["game_started"] or state["game"].winner is not None:
                    continue

                if get_current_mode(state) != "Human":
                    continue

                # board click
                cell = get_clicked_cell(mouse_pos, board_rect)
                if cell is None:
                    continue

                # play move
                if not state["game"].make_move(cell):
                    continue

                state["status"] = "Move played."
                state["last_ai_move_time"] = now
                if state["game"].winner is not None:
                    handle_game_end(state)

        if not state["game_started"] or state["game"].winner is not None:
            render_frame()
            clock.tick(60)
            continue

        current_mode = get_current_mode(state)
        if current_mode == "Human":
            render_frame()
            clock.tick(60)
            continue

        # ai delay
        if now - state["last_ai_move_time"] < AI_DELAY_SECONDS:
            render_frame()
            clock.tick(60)
            continue

        # ai move
        ensure_agent_ready_for_mode(state, current_mode, render_frame, pygame)
        move = get_ai_move(state["game"], current_mode, state)
        state["game"].make_move(move)
        state["status"] = f"{current_mode} played square {move + 1}."
        state["last_ai_move_time"] = time.perf_counter()

        if state["game"].winner is not None:
            handle_game_end(state)

        render_frame()
        clock.tick(60)

    pygame.quit()
