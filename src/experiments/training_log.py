"""
Author: Priyansh Nayak
Description: Helper functions for writing RL training logs
"""

import csv
import os


def write_training_log(log_path, rows):
    # write one csv file per training run
    if not rows:
        return

    # make sure the folder exists first
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
