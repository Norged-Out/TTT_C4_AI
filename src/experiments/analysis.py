"""
Author: Priyansh Nayak
Description: Loads experiment CSVs, builds summary tables, and saves plots
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = "figures"
TABLE_DIR = os.path.join("results", "analysis")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name))
    plt.close()


def load_csv(path):
    filepath = Path(path)
    if not filepath.exists():
        return None

    return pd.read_csv(filepath)


def normalize_matchup_df(df):
    if df is None or df.empty:
        return df

    rename_map = {
        "x_agent": "player1_agent",
        "o_agent": "player2_agent",
        "x_starts": "player1_starts",
        "o_starts": "player2_starts",
        "x_wins": "player1_wins",
        "o_wins": "player2_wins",
        "x_win_rate": "player1_win_rate",
        "o_win_rate": "player2_win_rate",
        "avg_x_time": "avg_player1_time",
        "avg_o_time": "avg_player2_time",
        "avg_x_nodes": "avg_player1_nodes",
        "avg_o_nodes": "avg_player2_nodes",
    }

    df = df.rename(columns=rename_map)

    if "player1_starts" not in df.columns:
        df["player1_starts"] = pd.NA
    if "player2_starts" not in df.columns:
        df["player2_starts"] = pd.NA
    if "avg_player1_nodes" not in df.columns:
        df["avg_player1_nodes"] = 0.0
    if "avg_player2_nodes" not in df.columns:
        df["avg_player2_nodes"] = 0.0

    return df


def matchup_to_long(df):
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "game": row["game"],
            "agent": row["player1_agent"],
            "opponent": row["player2_agent"],
            "starts": row["player1_starts"],
            "wins": row["player1_wins"],
            "win_rate": row["player1_win_rate"],
            "avg_time": row["avg_player1_time"],
            "avg_nodes": row["avg_player1_nodes"],
            "avg_moves": row["avg_moves"],
            "draw_rate": row["draw_rate"],
        })
        rows.append({
            "game": row["game"],
            "agent": row["player2_agent"],
            "opponent": row["player1_agent"],
            "starts": row["player2_starts"],
            "wins": row["player2_wins"],
            "win_rate": row["player2_win_rate"],
            "avg_time": row["avg_player2_time"],
            "avg_nodes": row["avg_player2_nodes"],
            "avg_moves": row["avg_moves"],
            "draw_rate": row["draw_rate"],
        })

    return pd.DataFrame(rows)


def save_table(df, name):
    if df is None or df.empty:
        return

    df.to_csv(os.path.join(TABLE_DIR, name), index=False)


def pairwise_summary(df, game_name):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    summary = long_df[
        ["agent", "opponent", "win_rate", "draw_rate", "avg_time", "avg_nodes", "avg_moves"]
    ].copy()
    summary = summary.sort_values(["agent", "opponent"]).reset_index(drop=True)
    save_table(summary, f"{game_name.lower()}_pairwise_summary.csv")


def agent_summary(df, game_name):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    summary = (
        long_df.groupby("agent", as_index=False)
        .agg(
            mean_win_rate=("win_rate", "mean"),
            mean_time=("avg_time", "mean"),
            mean_nodes=("avg_nodes", "mean"),
            mean_moves=("avg_moves", "mean"),
        )
        .sort_values("mean_win_rate", ascending=False)
    )

    save_table(summary, f"{game_name.lower()}_agent_summary.csv")


def rl_summary(df, game_name):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    rl_df = long_df[long_df["agent"].isin(["QLearning", "DQN"])].copy()
    if rl_df.empty:
        return

    rows = []
    for agent_name in ["QLearning", "DQN"]:
        subset = rl_df[rl_df["agent"] == agent_name]
        if subset.empty:
            continue

        default_subset = subset[subset["opponent"] == "Default"]
        rows.append({
            "agent": agent_name,
            "mean_win_rate": subset["win_rate"].mean(),
            "vs_default_win_rate": default_subset["win_rate"].mean() if not default_subset.empty else pd.NA,
            "mean_time": subset["avg_time"].mean(),
        })

    summary = pd.DataFrame(rows).sort_values("mean_win_rate", ascending=False)
    save_table(summary, f"{game_name.lower()}_rl_summary.csv")


def default_summary(df, game_name):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    summary = long_df[long_df["opponent"] == "Default"].copy()
    if summary.empty:
        return

    summary = summary.sort_values("win_rate", ascending=False)
    save_table(summary, f"{game_name.lower()}_vs_default.csv")


def overall_summary(ttt_df, c4_df):
    rows = []

    for game_name, df in [("TicTacToe", ttt_df), ("Connect4", c4_df)]:
        long_df = matchup_to_long(df)
        if long_df.empty:
            continue

        for agent_name in sorted(long_df["agent"].unique()):
            subset = long_df[long_df["agent"] == agent_name]
            default_subset = subset[subset["opponent"] == "Default"]

            rows.append({
                "game": game_name,
                "agent": agent_name,
                "mean_win_rate": subset["win_rate"].mean(),
                "vs_default_win_rate": default_subset["win_rate"].mean() if not default_subset.empty else pd.NA,
                "mean_time": subset["avg_time"].mean(),
                "mean_nodes": subset["avg_nodes"].mean(),
            })

    if not rows:
        return

    summary = pd.DataFrame(rows).sort_values(["game", "mean_win_rate"], ascending=[True, False])
    save_table(summary, "overall_agent_summary.csv")


def plot_win_rate_heatmap(df, game_name, filename):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    pivot = long_df.pivot_table(
        index="agent",
        columns="opponent",
        values="win_rate",
        aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(pivot.values, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{game_name}: Win Rate by Matchup")

    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            value = pivot.iloc[row, col]
            if pd.isna(value):
                continue
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Win Rate")
    save_fig(filename)


def plot_default_bars(df, game_name, filename):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    subset = long_df[long_df["opponent"] == "Default"].copy()
    subset = subset[subset["agent"] != "Default"]
    if subset.empty:
        return

    subset = subset.groupby("agent", as_index=False)["win_rate"].mean()
    subset = subset.sort_values("win_rate", ascending=False)

    plt.figure(figsize=(7, 4))
    plt.bar(subset["agent"], subset["win_rate"], color="#4c78a8")
    plt.ylim(0, 1)
    plt.ylabel("Win Rate")
    plt.title(f"{game_name}: Performance vs Default")
    save_fig(filename)


def plot_rl_training(log_path, title, filename):
    df = load_csv(log_path)
    if df is None or df.empty:
        return

    plt.figure(figsize=(8, 5))
    if "agent_win_rate" in df.columns:
        plt.plot(df["episode"], df["agent_win_rate"], label="Agent Win Rate")
    elif "x_win_rate" in df.columns:
        plt.plot(df["episode"], df["x_win_rate"], label="X Win Rate")

    if "opponent_win_rate" in df.columns:
        plt.plot(df["episode"], df["opponent_win_rate"], label="Opponent Win Rate")
    elif "o_win_rate" in df.columns:
        plt.plot(df["episode"], df["o_win_rate"], label="O Win Rate")

    if "draw_rate" in df.columns:
        plt.plot(df["episode"], df["draw_rate"], label="Draw Rate")

    if "opponent" in df.columns and "default" in set(df["opponent"]):
        default_rows = df[df["opponent"] == "default"]
        if not default_rows.empty:
            switch_episode = float(default_rows.iloc[0]["episode"])
            plt.axvline(switch_episode, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    save_fig(filename)


def plot_rl_reward(log_path, title, filename):
    df = load_csv(log_path)
    if df is None or df.empty or "avg_reward" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["avg_reward"], color="#f58518")

    if "opponent" in df.columns and "default" in set(df["opponent"]):
        default_rows = df[df["opponent"] == "default"]
        if not default_rows.empty:
            switch_episode = float(default_rows.iloc[0]["episode"])
            plt.axvline(switch_episode, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    save_fig(filename)


def plot_c4_search_cost(df):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    search = long_df[long_df["agent"].isin(["Minimax", "AlphaBeta"])].copy()
    if search.empty:
        return

    summary = (
        search.groupby("agent", as_index=False)
        .agg(mean_time=("avg_time", "mean"), mean_nodes=("avg_nodes", "mean"))
    )

    save_table(summary, "connect4_search_cost_summary.csv")

    plt.figure(figsize=(6, 4))
    plt.bar(summary["agent"], summary["mean_time"], color="#72b7b2")
    plt.ylabel("Average Time per Game (s)")
    plt.title("Connect 4 Search Runtime")
    save_fig("connect4_search_runtime.png")

    plt.figure(figsize=(6, 4))
    plt.bar(summary["agent"], summary["mean_nodes"], color="#e45756")
    plt.ylabel("Average Nodes per Game")
    plt.title("Connect 4 Search Work")
    save_fig("connect4_search_nodes.png")


def plot_c4_runtime_vs_nodes(df):
    long_df = matchup_to_long(df)
    if long_df.empty:
        return

    search = long_df[long_df["agent"].isin(["Minimax", "AlphaBeta"])].copy()
    if search.empty:
        return

    plt.figure(figsize=(6, 4))
    for agent_name in ["Minimax", "AlphaBeta"]:
        subset = search[search["agent"] == agent_name]
        if subset.empty:
            continue
        plt.scatter(subset["avg_nodes"], subset["avg_time"], label=agent_name)

    plt.xlabel("Average Nodes per Game")
    plt.ylabel("Average Time per Game (s)")
    plt.title("Connect 4 Search: Runtime vs Nodes")
    plt.legend()
    save_fig("connect4_runtime_vs_nodes.png")


def parse_connect4_search_benchmark(path):
    filepath = Path(path)
    if not filepath.exists():
        return pd.DataFrame()

    rows = []
    current = {}

    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Search: "):
            if current:
                rows.append(current)
            current = {"search": line.split(": ", 1)[1]}
            continue

        if line.startswith("Depth limit: "):
            depth_value = line.split(": ", 1)[1]
            for row in rows:
                if "depth_limit" not in row:
                    row["depth_limit"] = depth_value
            continue

        if ":" in line and current:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()

    if current:
        rows.append(current)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "depth_limit" not in df.columns:
        df["depth_limit"] = "full"

    return df


def benchmark_tables_and_plot():
    df = parse_connect4_search_benchmark("results/connect4_search_benchmark.txt")
    if df.empty:
        return

    save_table(df, "connect4_search_benchmark_summary.csv")

    limited = df[df["depth_limit"] != "full"].copy()
    if limited.empty:
        return

    limited["depth_limit"] = pd.to_numeric(limited["depth_limit"], errors="coerce")
    limited["elapsed_seconds"] = pd.to_numeric(limited["elapsed_seconds"], errors="coerce")
    limited["nodes_visited"] = pd.to_numeric(limited["nodes_visited"], errors="coerce")

    for metric, filename, ylabel, color in [
        ("elapsed_seconds", "connect4_benchmark_runtime.png", "Elapsed Time (s)", "#54a24b"),
        ("nodes_visited", "connect4_benchmark_nodes.png", "Nodes Visited", "#b279a2"),
    ]:
        plt.figure(figsize=(7, 4))
        for search_name in limited["search"].unique():
            subset = limited[limited["search"] == search_name].sort_values("depth_limit")
            plt.plot(subset["depth_limit"], subset[metric], marker="o", label=search_name)

        plt.xlabel("Depth Limit")
        plt.ylabel(ylabel)
        plt.title(f"Connect 4 Benchmark: {ylabel}")
        plt.legend()
        save_fig(filename)


def write_summary_notes(ttt_df, c4_df):
    lines = []

    if ttt_df is not None and not ttt_df.empty:
        lines.append("Tic Tac Toe")
        ttt_long = matchup_to_long(ttt_df)
        default_rows = ttt_long[ttt_long["opponent"] == "Default"]
        if not default_rows.empty:
            best = default_rows.sort_values("win_rate", ascending=False).iloc[0]
            lines.append(
                f"Best vs default: {best['agent']} win rate {best['win_rate']:.2f}"
            )

        overall = (
            ttt_long.groupby("agent", as_index=False)["win_rate"]
            .mean()
            .sort_values("win_rate", ascending=False)
        )
        if not overall.empty:
            best = overall.iloc[0]
            lines.append(
                f"Strongest overall: {best['agent']} mean win rate {best['win_rate']:.2f}"
            )

    if c4_df is not None and not c4_df.empty:
        lines.append("")
        lines.append("Connect 4")
        c4_long = matchup_to_long(c4_df)
        default_rows = c4_long[c4_long["opponent"] == "Default"]
        if not default_rows.empty:
            best = default_rows.sort_values("win_rate", ascending=False).iloc[0]
            lines.append(
                f"Best vs default: {best['agent']} win rate {best['win_rate']:.2f}"
            )

        overall = (
            c4_long.groupby("agent", as_index=False)["win_rate"]
            .mean()
            .sort_values("win_rate", ascending=False)
        )
        if not overall.empty:
            best = overall.iloc[0]
            lines.append(
                f"Strongest overall: {best['agent']} mean win rate {best['win_rate']:.2f}"
            )

        search_rows = c4_long[c4_long["agent"].isin(["Minimax", "AlphaBeta"])]
        if not search_rows.empty:
            fastest = search_rows.sort_values("avg_time").iloc[0]
            lines.append(
                f"Fastest search agent: {fastest['agent']} avg time {fastest['avg_time']:.3f}s"
            )

        rl_rows = c4_long[c4_long["agent"].isin(["QLearning", "DQN"])]
        if not rl_rows.empty:
            best_rl = (
                rl_rows.groupby("agent", as_index=False)["win_rate"]
                .mean()
                .sort_values("win_rate", ascending=False)
                .iloc[0]
            )
            lines.append(
                f"Best RL agent: {best_rl['agent']} mean win rate {best_rl['win_rate']:.2f}"
            )

    if not lines:
        return

    with open(os.path.join(TABLE_DIR, "summary_notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_analysis():
    ttt_df = normalize_matchup_df(load_csv("results/tictactoe_results.csv"))
    c4_df = normalize_matchup_df(load_csv("results/connect4_results.csv"))

    pairwise_summary(ttt_df, "tictactoe")
    pairwise_summary(c4_df, "connect4")
    agent_summary(ttt_df, "tictactoe")
    agent_summary(c4_df, "connect4")
    rl_summary(ttt_df, "tictactoe")
    rl_summary(c4_df, "connect4")
    default_summary(ttt_df, "tictactoe")
    default_summary(c4_df, "connect4")
    overall_summary(ttt_df, c4_df)

    plot_win_rate_heatmap(ttt_df, "Tic Tac Toe", "tictactoe_win_rates.png")
    plot_win_rate_heatmap(c4_df, "Connect 4", "connect4_win_rates.png")
    plot_default_bars(ttt_df, "Tic Tac Toe", "tictactoe_vs_default.png")
    plot_default_bars(c4_df, "Connect 4", "connect4_vs_default.png")

    plot_c4_search_cost(c4_df)
    plot_c4_runtime_vs_nodes(c4_df)
    benchmark_tables_and_plot()

    plot_rl_training(
        "results/training/tictactoe_q_learning.csv",
        "Tic Tac Toe Q-learning Training",
        "tictactoe_q_training_rates.png",
    )
    plot_rl_training(
        "results/training/tictactoe_dqn.csv",
        "Tic Tac Toe DQN Training",
        "tictactoe_dqn_training_rates.png",
    )
    plot_rl_training(
        "results/training/connect4_q_learning.csv",
        "Connect 4 Q-learning Training",
        "connect4_q_training_rates.png",
    )
    plot_rl_training(
        "results/training/connect4_dqn.csv",
        "Connect 4 DQN Training",
        "connect4_dqn_training_rates.png",
    )

    plot_rl_reward(
        "results/training/connect4_q_learning.csv",
        "Connect 4 Q-learning Reward",
        "connect4_q_training_reward.png",
    )
    plot_rl_reward(
        "results/training/connect4_dqn.csv",
        "Connect 4 DQN Reward",
        "connect4_dqn_training_reward.png",
    )

    write_summary_notes(ttt_df, c4_df)
