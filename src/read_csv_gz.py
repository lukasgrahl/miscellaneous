import gzip
import csv
import json
import pandas as pd

import os

from settings import DATA_DIR


def read_csvgz(filename: str,
               data_dir: str):
    _ = pd.read_csv(os.path.join(data_dir, f"{filename}.csv.gz"),
                    compression='gzip')

    return _


if __name__ == "__main__":
    from settings import DATA_DIR, OUTPUT_DIR

    df = read_csvgz(filename="GarciaNYTimesIndex",
                    data_dir=DATA_DIR)
    df.to_excel(os.path.join(OUTPUT_DIR, "garcia_nytimes_fears.xlsx"))
