"""Continue the predeclared five-PC comparison once teacher training completes."""
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
    """Queue ordered prep → drift → diffusion → validation → qualified observed-data proxy."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-pid',type=int,required=True)
    args=parser.parse_args()
    status=Path('outputs/logs/campaign_pc5_status.json')
    write_status(status,stage='waiting_for_training',training_pid=args.training_pid)
    while True:
        try:os.kill(args.training_pid,0)
        except ProcessLookupError:break
        time.sleep(30)
    trained=json.loads(Path('outputs/logs/training_status_pc5.json').read_text())
    if trained['stage']!='training_finished':
        write_status(status,stage='failed',error='Five-PC teacher training did not finish')
        raise SystemExit(1)
    state='outputs/distillation_pc5';sindy='outputs/sindy_pc5'
    checkpoint='outputs/baseline_pc5/official/teacher.ckpt'
    commands=[
        ('prepare',['src.prepare','--components','5','--state-dir',state,'--checkpoint',checkpoint,'--figure-prefix','pc5_']),
        ('distillation',['src.run_distillation','--config','configs/experiment_pc5.json']),
        ('validation',['src.validate','--state-dir',state,'--sindy-dir',sindy,'--checkpoint',checkpoint,
                       '--output-dir','outputs/validation_pc5','--figure-prefix','pc5_']),
        ('direct_baseline',['src.direct_baseline','--state-dir',state,'--sindy-dir',sindy,'--checkpoint',checkpoint,
                            '--output-dir','outputs/direct_baseline_pc5'])]
    for stage,command in commands:
        write_status(status,stage=stage,command=command)
        with Path(f'outputs/logs/{stage}_pc5_run.txt').open('w') as log:
            result=subprocess.run([sys.executable,'-m',*command],stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            write_status(status,stage='failed',failed_stage=stage,returncode=result.returncode)
            raise SystemExit(result.returncode)
    write_status(status,stage='finished',caveat='Review scientific results and create final executed notebooks before marking goal complete')

if __name__=='__main__':main()
