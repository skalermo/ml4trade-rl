# ml4trade-rl

Aim of this repository is solving [ml4trade](https://github.com/skalermo/ml4trade)
gym environment using reinforcement learning algorithms (specifically A2C and PPO).

## Requirements

- python 3.7 - 3.10
- pip requirements
- data to process by the environment

The data used by ml4trade by default can be retrieved via provided bash script
[download_all.sh](data/download_all.sh). This is described in **Installation** step.

## Instalation

Pip-install requirements:
```bash
pip install -r requirements.txt
```

Download default required data (~1.5GB in total):
```bash
bash data/download_all.sh
```

Necessary files will be downloaded, unpacked and stored in
automatically created `.data/` directory.

## Testing

You can run tests stored in `test/` directory to see if everything works
as expected:
```bash
python -m unittest discover test
```

## Usage

Start by using `run.py` script. Arguments are handled by
[Hydra](https://hydra.cc/docs/intro/) framework: by default arguments are
taken from [conf/](conf) directory. They are proven to give good results, but
you can override them if needed. Seed can be optionally provided by
`+run.seed=your_seed_here`, otherwise it is acquired from current datetime.

For example, given you want to run `A2C` algorithm for `1,000,000` time steps
with `n_steps` parameter set to `50` and `seed` set to `42`:

```bash
python run.py agent=a2c run.train_steps=1e6 agent.n_steps=50 +run.seed=42
```

All parameters stored in [conf/](conf) are configurable.

Other things worth mentioning:
- results are stored in `outputs/` directory, during each run Hydra creates there
a new directory where logs and artifacts are stored
- artifacts mentioned previously include: saved best model according to the
performance on the test env, history of the actions on the test env, various logs,
performance plots
- consider using TensorBoard, TensorBoard logs are written during runs

### Note

`run.py` was meant to be flexible and frequently changed according to needs.
Feel free to tinker with it.
