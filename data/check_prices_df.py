from datetime import timedelta
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print('Install numpy and pandas. Exiting early.')
    exit(1)


prices_path = Path(__file__).parent.parent / '.data' / 'prices_pl.csv'


def _get_prices_df() -> pd.DataFrame:
    prices_df: pd.DataFrame = pd.read_csv(prices_path, header=0)
    prices_df.fillna(method='bfill', inplace=True)
    prices_df['index'] = pd.to_datetime(prices_df['index'])
    return prices_df


def check_df(fix: bool = False):
    print(f'Checking prices df with flag fix={fix}.')
    col = 'Fixing I Price [PLN/MWh]'
    df = _get_prices_df()
    cur = df['index'][0]
    end = df['index'][len(df) - 1]
    i = 0
    while cur < end:
        idx = np.where(df['index'] == cur)[0]
        while not len(idx):
            print(f'Found missing data for {cur}.')
            price_before = df.at[i - 1, col]
            price_after = df.at[i + 1, col]
            price_avg = round((price_before + price_after) / 2, ndigits=2)
            if fix:
                print(f'Filling in data for {cur}.')
                columns = df.columns
                df = pd.DataFrame(np.insert(df.values, i, values=[cur, price_avg, 0, 0, 0, 0, 0], axis=0))
                i += 1
                df.columns = columns
            cur += timedelta(hours=1)
            idx = np.where(df['index'] == cur)[0]
        cur += timedelta(hours=1)
        i += 1
    if fix:
        df.to_csv(prices_path, index=False)
    print('Done.')


if __name__ == '__main__':
    check_df(fix=True)
