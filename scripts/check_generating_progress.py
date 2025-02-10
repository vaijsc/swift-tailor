import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == '__main__':
    from src.utils.mics import get_all_gcd_files

    parser = argparse.ArgumentParser(description="Process a date argument.")
    parser.add_argument("--after-day", type=str,
                        help="Date in YYYY-MM-DD format", default="2025-01-01")

    args = parser.parse_args()
    after_day = time.mktime(
        datetime.datetime.strptime(args.after_day, "%Y-%m-%d").timetuple())

    all_samples = get_all_gcd_files()

    path_format = "{}/{}_gim.npz"

    st_day = np.inf
    end_day = -np.inf
    cnt = 0
    total_gb = 0
    for sample in all_samples:
        path = Path(sample)
        sample_id = path.name

        gim_path = path_format.format(sample, sample_id)
        if os.path.exists(gim_path):
            # get last editted day of file
            last_editted_day = os.path.getmtime(gim_path)
            if last_editted_day < after_day:
                continue
            st_day = min(st_day, last_editted_day)
            end_day = max(end_day, last_editted_day)
            cnt += 1

            total_gb += os.path.getsize(gim_path) / (1024 ** 3)

    progress_percent = cnt / len(all_samples) * 100
    exp_time_to_finish = (end_day - st_day) * \
        (len(all_samples) - cnt) / (cnt * 60 * 60)

    print(f"Total size: {total_gb:.2f} GB")
    print(f"Progress: {progress_percent:.2f}% ({cnt}/{len(all_samples)})")
    print(f"ETA: {exp_time_to_finish:.2f} hours")
