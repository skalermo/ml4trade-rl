from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import json

from ml4trade.rendering.charts import render_all

history_file = 'outputs/2022-06-16/21-12-57/env_history.json'


def main():
    with open(history_file) as f:
        history = json.load(f)

    history['datetime'] = list(map(datetime.fromisoformat, history['datetime']))
    render_all(history, 2)


if __name__ == '__main__':
    main()
