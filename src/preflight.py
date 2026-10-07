"""Read-only environment and official dataset metadata diagnostics."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
import subprocess
from urllib.request import urlopen

PACKAGES = ('torch', 'scdiffeq', 'torchsde', 'anndata', 'scanpy', 'numpy')
RECORD = 'https://zenodo.org/api/records/21947161'


def inspect_environment(data_dir: Path) -> dict:
    """Return installed versions, storage and download provenance; no data download."""
    data_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        'recorded_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'python': platform.python_version(), 'packages': versions,
        'data_dir': str(data_dir.resolve()),
        'free_bytes': shutil.disk_usage(data_dir).free,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'cuda_runtime': None,
    }
    if versions['torch']:
        import torch
        result['cuda_runtime'] = torch.version.cuda
        result['cuda_available'] = torch.cuda.is_available()
    if shutil.which('nvidia-smi'):
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
        result['nvidia_smi'] = gpu.stdout or gpu.stderr
    else:
        result['nvidia_smi'] = 'not available'
    with urlopen(RECORD, timeout=30) as response:
        record = json.load(response)
    result['source'] = RECORD
    result['files'] = [{k: f[k] for k in ('key', 'size', 'checksum', 'links')}
                       for f in record['files'] if f['key'].startswith('larry')]
    result['minimum_download_bytes'] = next(f['size'] for f in result['files'] if f['key'] == 'larry.h5ad')
    result['ready'] = bool(all(versions.values()) and tuple(map(int, platform.python_version_tuple())) >= (3, 11, 0)
                       and result['free_bytes'] >= 20_000_000_000)
    return result


def main() -> None:
    """Save preflight diagnostics, failing explicitly when prerequisites are missing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    result = inspect_environment(args.data_dir)
    logs = Path('outputs/logs')
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'preflight.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['ready']:
        raise SystemExit('Preflight incomplete: see outputs/logs/preflight.json')

if __name__ == '__main__':
    main()
