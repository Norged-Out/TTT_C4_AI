"""
Author: Priyansh Nayak
Description: Text based UI to play Tic Tac Toe in the terminal
"""

from src.agents.tictactoe.alphabeta import choose_alphabeta_move
from src.agents.tictactoe.default_opponent import choose_default_move
from src.agents.tictactoe.minimax import choose_minimax_move
from src.agents.tictactoe.q_learning import choose_q_move, train_q_learning
from src.games.tictactoe.game import TicTacToe


def prompt_for_move(game: TicTacToe) -> int:
    while True:
        raw_value = input(f"Player {game.current_player}, choose a square (1-9): ").strip()

        if raw_value.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt

        if not raw_value.isdigit():
            print("Please enter a number from 1 to 9.")
            continue

        position = int(raw_value) - 1

        if position not in game.available_moves():
            print("That square is not available. Try again.")
            continue

        return position


def play_tictactoe() -> None:
    game = TicTacToe()

    print("Tic Tac Toe")
    print("Enter a number from 1 to 9 to place your mark.")
    print("Type q to quit.\n")

    try:
        while not game.is_game_over():
            print(game.render())
            print()
            move = prompt_for_move(game)
            game.make_move(move)
            print()

        print(game.render())
        print()
        if game.winner == "Draw":
            print("It's a draw.")
        else:
            print(f"Player {game.winner} wins.")

    except KeyboardInterrupt:
        print("\nGame ended.")


def play_tictactoe_vs_default() -> None:
    game = TicTacToe()
    human = "X"

    print("Tic Tac Toe")
    print("You are X.")
    print("Enter a number from 1 to 9 to place your mark.")
    print("Type q to quit.\n")

    try:
        while not game.is_game_over():
            print(game.render())
            print()

            if game.current_player == human:
                move = prompt_for_move(game)
            else:
                move = choose_default_move(game)
                print(f"Computer chooses square {move + 1}.")

            game.make_move(move)
            print()

        print(game.render())
        print()
        if game.winner == "Draw":
            print("It's a draw.")
        elif game.winner == human:
            print("You win.")
        else:
            print("Computer wins.")

    except KeyboardInterrupt:
        print("\nGame ended.")


def play_tictactoe_vs_minimax() -> None:
    game = TicTacToe()
    human = "X"

    print("Tic Tac Toe")
    print("You are X.")
    print("Computer uses Minimax.")
    print("Enter a number from 1 to 9 to place your mark.")
    print("Type q to quit.\n")

    try:
        while not game.is_game_over():
            print(game.render())
            print()

            if game.current_player == human:
                move = prompt_for_move(game)
            else:
                move = choose_minimax_move(game)
                print(f"Computer chooses square {move + 1}.")

            game.make_move(move)
            print()

        print(game.render())
        print()
        if game.winner == "Draw":
            print("It's a draw.")
        elif game.winner == human:
            print("You win.")
        else:
            print("Computer wins.")

    except KeyboardInterrupt:
        print("\nGame ended.")


def play_tictactoe_vs_alphabeta() -> None:
    game = TicTacToe()
    human = "X"

    print("Tic Tac Toe")
    print("You are X.")
    print("Computer uses Alpha Beta.")
    print("Enter a number from 1 to 9 to place your mark.")
    print("Type q to quit.\n")

    try:
        while not game.is_game_over():
            print(game.render())
            print()

            if game.current_player == human:
                move = prompt_for_move(game)
            else:
                move = choose_alphabeta_move(game)
                print(f"Computer chooses square {move + 1}.")

            game.make_move(move)
            print()

        print(game.render())
        print()
        if game.winner == "Draw":
            print("It's a draw.")
        elif game.winner == human:
            print("You win.")
        else:
            print("Computer wins.")

    except KeyboardInterrupt:
        print("\nGame ended.")


def play_tictactoe_vs_q_learning() -> None:
    game = TicTacToe()
    human = "X"

    print("Training Q-learning agent...")
    q_table = train_q_learning(episodes=5000)

    print("Tic Tac Toe")
    print("You are X.")
    print("Computer uses Q-learning.")
    print("Enter a number from 1 to 9 to place your mark.")
    print("Type q to quit.\n")

    try:
        while not game.is_game_over():
            print(game.render())
            print()

            if game.current_player == human:
                move = prompt_for_move(game)
            else:
                move = choose_q_move(game, q_table)
                print(f"Computer chooses square {move + 1}.")

            game.make_move(move)
            print()

        print(game.render())
        print()
        if game.winner == "Draw":
            print("It's a draw.")
        elif game.winner == human:
            print("You win.")
        else:
            print("Computer wins.")

    except KeyboardInterrupt:
        print("\nGame ended.")


def play_tictactoe_vs_dqn() -> None:
    from src.agents.tictactoe.dqn import choose_dqn_move, train_dqn

    game = TicTacToe()
    human = "X"

    print("Training DQN agent...")
    model = train_dqn(episodes=5000)

    print("Tic Tac Toe")
    print("You are X.")
    print("Computer uses DQN.")
    print("Enter a number from 1 to 9 to place your mark.")
    print("Type q to quit.\n")

    try:
        while not game.is_game_over():
            print(game.render())
            print()

            if game.current_player == human:
                move = prompt_for_move(game)
            else:
                move = choose_dqn_move(game, model)
                print(f"Computer chooses square {move + 1}.")

            game.make_move(move)
            print()

        print(game.render())
        print()
        if game.winner == "Draw":
            print("It's a draw.")
        elif game.winner == human:
            print("You win.")
        else:
            print("Computer wins.")

    except KeyboardInterrupt:
        print("\nGame ended.")
