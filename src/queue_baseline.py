"""Wait for the existing official loader process; start training only after M0 succeeds."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .baseline import write_status


def main() -> None:
    """Wait without downloading twice; never start on an incomplete dataset inventory."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-pid', type=int, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    status = Path('outputs/logs/training_status.json')
    write_status(status, stage='waiting_for_data', download_pid=args.download_pid)
    while True:
        try:
            os.kill(args.download_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(30)
    try:
        manifest = json.loads(Path('outputs/logs/data_provenance.json').read_text())
        inventory = json.loads(Path('outputs/logs/dataset_inventory.json').read_text())
        if not manifest['files'] or inventory['quickstart_subset']['n_cells'] <= 0:
            raise ValueError('Missing verified cache or empty subset')
    except (OSError, ValueError, KeyError) as exc:
        write_status(status, stage='data_failed', error=repr(exc),
                     log='outputs/logs/data_run.txt')
        raise SystemExit('M0 did not finish successfully; training will not start') from exc
    with Path('outputs/logs/baseline_run.txt').open('w') as log:
        completed = subprocess.run([sys.executable, '-m', 'src.baseline',
                                    '--data-dir', str(args.data_dir), '--epochs', '1500'],
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)

if __name__ == '__main__':
    main()
