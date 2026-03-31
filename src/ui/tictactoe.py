"""
Author: Priyansh Nayak
Description: Pygame UI to play Tic Tac Toe
"""

from src.games.tictactoe.game import TicTacToe


MODES = [
    "Two Player",
    "Default",
    "Minimax",
    "Alpha Beta",
    "Q-learning",
    "DQN",
]


def get_ai_move(game, mode, state):
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
    state["game"] = TicTacToe()
    state["winner_text"] = ""
    state["status"] = "Game reset."
    state["train_progress"] = None


def handle_game_end(state):
    winner = state["game"].winner

    if winner == "Draw":
        state["winner_text"] = "Draw"
        return

    state["winner_text"] = f"{winner} wins"


def get_board_layout(board_rect):
    cell_size = min(board_rect.width, board_rect.height) // 3
    total_size = cell_size * 3
    x0 = board_rect.left + (board_rect.width - total_size) // 2
    y0 = board_rect.top + (board_rect.height - total_size) // 2
    return x0, y0, cell_size


def get_clicked_cell(mouse_pos, board_rect):
    x0, y0, cell_size = get_board_layout(board_rect)
    mx, my = mouse_pos

    if not (x0 <= mx < x0 + 3 * cell_size and y0 <= my < y0 + 3 * cell_size):
        return None

    col = (mx - x0) // cell_size
    row = (my - y0) // cell_size
    return int(row * 3 + col)


def draw_board(screen, pygame, board_rect, game, fonts):
    bg = (245, 245, 245)
    grid = (30, 30, 30)
    x_color = (30, 80, 180)
    o_color = (200, 80, 40)

    pygame.draw.rect(screen, bg, board_rect)

    x0, y0, cell_size = get_board_layout(board_rect)
    total_size = cell_size * 3

    # outer board boundary
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


def draw_sidebar(screen, pygame, sidebar_rect, state, fonts, button_rects):
    pygame.draw.rect(screen, (35, 35, 35), sidebar_rect)

    y = 24

    title = fonts["title"].render("Tic Tac Toe", True, (245, 245, 245))
    screen.blit(title, (sidebar_rect.left + 20, y))
    y += 50

    mode_text = fonts["body"].render(f"Mode: {state['mode']}", True, (230, 230, 230))
    screen.blit(mode_text, (sidebar_rect.left + 20, y))
    y += 32

    if state["game"].winner is None:
        turn_text = fonts["body"].render(f"Turn: {state['game'].current_player}", True, (230, 230, 230))
    else:
        turn_text = fonts["body"].render(f"Result: {state['winner_text']}", True, (230, 230, 230))
    screen.blit(turn_text, (sidebar_rect.left + 20, y))
    y += 32

    status_text = fonts["small"].render(state["status"], True, (180, 180, 180))
    screen.blit(status_text, (sidebar_rect.left + 20, y))
    y += 28

    if state["train_progress"] is not None:
        progress_text = fonts["small"].render(
            f"Preparing agent: {state['train_progress']}%",
            True,
            (200, 200, 120),
        )
        screen.blit(progress_text, (sidebar_rect.left + 20, y))
        y += 28

    hint = fonts["small"].render("Press R to reset.", True, (190, 190, 190))
    screen.blit(hint, (sidebar_rect.left + 20, y))

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


def ensure_agent_ready(state, render_callback, pygame):
    if state["mode"] == "Q-learning" and state["q_table"] is None:
        from src.agents.tictactoe.q_learning import train_q_learning

        state["status"] = "Preparing Q-learning..."
        state["train_progress"] = None
        render_callback()

        def progress(done, total):
            state["train_progress"] = int((done / total) * 100)
            state["status"] = f"Training Q-learning... {state['train_progress']}%"
            pygame.event.pump()
            render_callback()

        state["q_table"] = train_q_learning(progress_callback=progress)
        state["status"] = "Q-learning ready."
        state["train_progress"] = None
        render_callback()
        return

    if state["mode"] == "DQN" and state["dqn_model"] is None:
        from src.agents.tictactoe.dqn import train_dqn

        state["status"] = "Preparing DQN..."
        state["train_progress"] = None
        render_callback()

        def progress(done, total):
            state["train_progress"] = int((done / total) * 100)
            state["status"] = f"Training DQN... {state['train_progress']}%"
            pygame.event.pump()
            render_callback()

        state["dqn_model"] = train_dqn(progress_callback=progress)
        state["status"] = "DQN ready."
        state["train_progress"] = None
        render_callback()


def run_game():
    import pygame

    pygame.init()

    state = {
        "mode": "Two Player",
        "game": TicTacToe(),
        "winner_text": "",
        "status": "Ready.",
        "train_progress": None,
        "q_table": None,
        "dqn_model": None,
    }

    screen_w = 1000
    screen_h = 700
    sidebar_w = 280

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Tic Tac Toe")

    fonts = {
        "title": pygame.font.SysFont("arial", 28, bold=True),
        "body": pygame.font.SysFont("arial", 22),
        "small": pygame.font.SysFont("arial", 18),
    }

    sidebar_rect = pygame.Rect(0, 0, sidebar_w, screen_h)
    board_rect = pygame.Rect(sidebar_w, 0, screen_w - sidebar_w, screen_h)

    button_rects = {}
    y = 160
    for mode in MODES:
        button_rects[mode] = pygame.Rect(20, y, sidebar_w - 40, 42)
        y += 54

    button_rects["Reset"] = pygame.Rect(20, y + 10, sidebar_w - 40, 42)

    clock = pygame.time.Clock()
    running = True

    def render_frame():
        screen.fill((20, 20, 20))
        draw_sidebar(screen, pygame, sidebar_rect, state, fonts, button_rects)
        draw_board(screen, pygame, board_rect, state["game"], fonts)
        pygame.display.flip()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_game(state)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos

                # sidebar buttons
                for key, rect in button_rects.items():
                    if rect.collidepoint(mouse_pos):
                        if key == "Reset":
                            reset_game(state)
                            break

                        state["mode"] = key
                        reset_game(state)
                        state["status"] = f"{key} selected."
                        ensure_agent_ready(state, render_frame, pygame)
                        break
                else:
                    if state["game"].winner is not None:
                        continue

                    if state["mode"] != "Two Player" and state["game"].current_player != "X":
                        continue

                    cell = get_clicked_cell(mouse_pos, board_rect)
                    if cell is None:
                        continue

                    if not state["game"].make_move(cell):
                        continue

                    state["status"] = "Move played."
                    if state["game"].winner is not None:
                        handle_game_end(state)

        # AI turn in one-player modes
        if state["mode"] == "Two Player" or state["game"].winner is not None:
            render_frame()
            clock.tick(60)
            continue

        if state["game"].current_player != "O":
            render_frame()
            clock.tick(60)
            continue

        ensure_agent_ready(state, render_frame, pygame)
        move = get_ai_move(state["game"], state["mode"], state)
        state["game"].make_move(move)
        state["status"] = f"{state['mode']} played square {move + 1}."

        if state["game"].winner is not None:
            handle_game_end(state)

        render_frame()
        clock.tick(60)

    pygame.quit()
