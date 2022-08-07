import unittest
import os
import sys
from pathlib import Path
import filecmp
import glob

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import run


class TestRuns(unittest.TestCase):
    def test_runs_are_reproducible(self):
        outputs_path = Path(os.getcwd()) / 'outputs'
        seed = 42
        sys.argv = []
        sys.argv.extend([
            '',
            'agent=a2c',
            'agent.n_steps=2',
            'run.render_all=False',
            'run.train_steps=10',
            'run.eval_freq=10',
            f'+run.seed={seed}',
            'tag=_reproducibility-test',
        ])
        run.main()
        run.main()
        history_files = glob.glob(f'{outputs_path}/**/env_history.json', recursive=True)
        progress_files = glob.glob(f'{outputs_path}/**/progress.json', recursive=True)
        history_files.sort()
        progress_files.sort()
        history1, history2 = history_files[-2:]
        progress1, progress2 = progress_files[-2:]
        self.assertTrue(filecmp.cmp(history1, history2))
        self.assertTrue(filecmp.cmp(progress1, progress2))


if __name__ == '__main__':
    unittest.main()
