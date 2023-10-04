from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import get_prices_df


def avg_price(df, name=''):
    cols = 'Fixing I Price [PLN/MWh]'
    if 'level_0' in df.columns:
        df = df.drop('level_0', axis=1)
    mean = df.groupby(df['index'].map(lambda t: t.hour)).mean()
    median = df.groupby(df['index'].map(lambda t: t.hour)).median()

    plt.plot(list(range(0, 24)), mean[cols], label='mean')
    plt.plot(list(range(0, 24)), median[cols], label='median')

    plt.xticks(range(0, 24))

    # plt.legend()
    plt.show()
    # plt.savefig(name)


def daily_prices_ratios():
    # df_new = df[:int(len(df) * 0.8)]
    # df_new = df[24:]
    df_new = df
    max_to_min_ratios = df.groupby(
        [df_new['index'].dt.year.rename('year'), df_new['index'].dt.month.rename('month'),
         df_new['index'].dt.day.rename('day')]
    )['Fixing I Price [PLN/MWh]'].apply(lambda day_prices: max(day_prices) / min(day_prices)).reset_index()
    # )['Fixing I Price [PLN/MWh]'].apply(lambda day_prices: np.average(day_prices)).reset_index()
    max_to_min_ratios.rename(columns={'Fixing I Price [PLN/MWh]': 'ratio'}, inplace=True)
    max_to_min_ratios['date'] = pd.to_datetime(max_to_min_ratios[['year', 'month', 'day']])


    # f = plt.figure()
    f, ax = plt.subplots()
    # start = 0
    # end = 4*365-1-2*90
    # print(end - start)
    # plt.plot(max_to_min_ratios['date'][start:end], max_to_min_ratios['ratio'][start:end], '.', color='green')
    # start = end
    # end = 4*365-1 - 90
    # print(end - start)
    # plt.plot(max_to_min_ratios['date'][start:end], max_to_min_ratios['ratio'][start:end], '.', color='yellow')
    # start = end
    # end = 4 * 365 - 1
    # print(end - start)
    # plt.plot(max_to_min_ratios['date'][start:end], max_to_min_ratios['ratio'][start:end], '.', color='red')
    # start = end
    # end = None
    # print(len(max_to_min_ratios['date'][start:end]))
    # plt.plot(max_to_min_ratios['date'][start:end], max_to_min_ratios['ratio'][start:end], '.', color='grey')
    # plt.ylabel('Zł / MWh')
    max_to_min_ratios = max_to_min_ratios[max_to_min_ratios['year'] < 2020]

    # plt.plot(max_to_min_ratios['date'], max_to_min_ratios['ratio'], '.', color='red')
    # plt.axhline(y=1 / 0.5, color='blue', linestyle='--')
    # plt.axhline(y=1 / 0.85, color='green', linestyle='--')
    # plt.yticks([1.17, 2, 4, 10])
    # plt.axvline(x=max_to_min_ratios['date'][int(len(max_to_min_ratios) * 0.8)], color='grey', linestyle='--')

    plt.hist(max_to_min_ratios['ratio'], bins=100)
    plt.xticks([1.17, 2, 4, 10])
    ax.set_xticklabels([0.85, 0.5, 0.25, 0.1])
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # plt.set_ticklabels([0.85, 0.5, 0.25, 0.1])
    plt.show()
    # f.savefig("prices-data-split.pdf", bbox_inches='tight')


def all_prices():
    cols = 'Fixing I Price [PLN/MWh]'

    # plt.plot(df['index'], df[cols])
    end_date = datetime(2022, 1, 1)
    cur_date = datetime(2016, 2, 1)

    idx = np.where(df['index'] == cur_date)[0][0]
    idx2 = np.where(df['index'] == end_date)[0][0]

    while cur_date < end_date:
        next_date = cur_date + timedelta(days=1)
        print(cur_date, next_date)
        idx = np.where(df['index'] == cur_date)[0][0]
        idx2 = np.where(df['index'] == next_date)[0][0]
        name = f'{str(cur_date)[:-9]}-{str(next_date)[:-9]}'
        plt.title(name)
        avg_price(df[idx:idx2], f'avg_prices/{name}.png')
        cur_date = next_date
        plt.cla()
        plt.clf()
    plt.axvline(x=df['index'][idx], color='grey', linestyle='--')
    plt.axvline(x=df['index'][idx2], color='grey', linestyle='--')
    # avg_price(df[idx2:idx])

    plt.show()


def get_prices_optimums(df: pd.DataFrame, test_start_datetime: datetime = None):
    # dt = datetime(year=2022, month=7, day=25) - datetime(year=2016, month=1, day=1)
    # print(dt)
    # print(len(df), len(df) / 24)

    time_col = 'index'
    col = 'Fixing I Price [PLN/MWh]'
    prices = df[[time_col, col]]
    if test_start_datetime is not None:
        prices = prices[prices[time_col] < test_start_datetime]

    prices_optimums: pd.DataFrame = df.groupby(
        [prices['index'].dt.year.rename('year'), prices['index'].dt.month.rename('month'),
         prices['index'].dt.day.rename('day')]
    )[col].apply(get_day_hilos).reset_index()
    prices_optimums.rename(columns={col: 'optimums'}, inplace=True)
    prices_optimums['date'] = pd.to_datetime(prices_optimums[['year', 'month', 'day']])
    prices_optimums['cls'] = prices_optimums.apply(lambda row: row['optimums'][-1], axis=1)
    prices_optimums['weekday'] = prices_optimums.apply(lambda row: row['date'].isoweekday(), axis=1)
    return prices_optimums


def daily_prices_diff_analysis(df: pd.DataFrame, test_start_datetime: datetime = None, group_by=None):
    if group_by is None:
        group_by = ['month', 'weekday']
    if not isinstance(group_by, list):
        group_by = [group_by]
    prices_optimums = get_prices_optimums(df, test_start_datetime)
    gb = prices_optimums.groupby(group_by)
    res = {}
    for x in gb.groups:
        dct = dict(gb.get_group(x).value_counts(['cls'], normalize=True))
        dct = {k[0]: dct[k] for k in dct}
        res[x] = dct
    return res


def get_day_hilos(arr: pd.Series):
    arr = arr.to_numpy()
    night_low = np.argmin(arr[:10])
    morning_high = np.argmax(arr[night_low:12]) + night_low
    day_low = np.argmin(arr[morning_high:18]) + morning_high
    evening_high = np.argmax(arr[day_low:]) + day_low
    return night_low, morning_high, day_low, evening_high, \
        arr[night_low], arr[morning_high], arr[day_low], arr[evening_high], \
        analyze_hilos(mh=arr[morning_high], dl=arr[day_low], eh=arr[evening_high])


def analyze_hilos(mh, dl, eh, margin: float = 0.1):

    def gt(a: float, b: float) -> bool:
        return a > b and not math.isclose(a, b, rel_tol=margin, abs_tol=0.0)

    if gt(mh, dl):
        if gt(mh, eh):
            if gt(dl, eh):
                return '321'
            elif gt(eh, dl):
                return '312'
            else:  # dl == eh
                return '211'
        elif gt(eh, mh):
            return '213'
        else:  # mh == eh
            return '212'
    elif gt(dl, mh):
        if gt(mh, eh):
            return '231'
        elif gt(eh, mh):
            if gt(dl, eh):
                return '132'
            elif gt(eh, dl):
                return '123'
            else:  # dl == eh
                return '122'
        else:  # mh == eh
            return '121'
    else:  # mh == dl
        if gt(mh, eh):
            return '221'
        elif gt(eh, mh):
            return '112'
        else:  # mh == eh
            return '111'


def prices_diff_profit_per_mwh(nlv, mhv, dlv, ehv, bat_eff):
    effective_nlv = nlv * (1 / bat_eff)
    effective_dlv = dlv * (1 / bat_eff)
    night_to_morning_profit = mhv - effective_nlv
    night_to_evening_profit = ehv - effective_nlv
    day_to_evening_profit = ehv - effective_dlv
    res = max(
        max(night_to_morning_profit, 0) + max(day_to_evening_profit, 0),
        max(night_to_evening_profit, 0),
    )
    return res


if __name__ == '__main__':
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    df = get_prices_df()
    # avg_price(df)
    daily_prices_ratios()

    # all_prices()
    exit()

    res = get_prices_optimums(df)
    a = res[res['date'] < datetime(year=2020, month=1, day=1) - timedelta(days=30 * 6)]
    print(a)
    a = a.value_counts('cls')
    print(a)
    b = res[res['date'] > datetime(year=2020, month=1, day=1) - timedelta(days=30 * 6)]
    b = b[b['date'] < datetime(year=2020, month=1, day=1)]
    print(b)
    b = b.value_counts('cls')
    print(b)
    # a: pd.DataFrame = get_prices_optimums(df, datetime(year=2021, month=9, day=1))
    # a['potential_profit'] = a.apply(
    #     lambda row: prices_diff_profit_per_mwh(row['optimums'][4], row['optimums'][5], row['optimums'][6], row['optimums'][7], 0.85),
    #     axis=1,
    # )
    # prices_optimums['cls'] = prices_optimums.apply(lambda row: row['optimums'][-1], axis=1)
    # prices_optimums['weekday'] = prices_optimums.apply(lambda row: row['date'].isoweekday(), axis=1)
    # a = a[a['date'] < datetime(year=2021, month=9, day=1)]
    # a = a[a['date'] >= datetime(year=2020, month=7, day=17)]
    # print(len(a))
    # print(np.cumsum(a['potential_profit']) * 0.9 * 0.864)
