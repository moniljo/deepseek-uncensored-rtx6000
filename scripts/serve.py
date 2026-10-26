"""Start the configured local API in this terminal; Ctrl+C stops it."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'.work/local/server.json')
    ap.add_argument('--run',action='store_true')
    a = ap.parse_args()
    if not a.run:
        print(json.dumps({'plan_only':True,'config':str(a.config.resolve()),'host':'127.0.0.1',
                          'interface':'OpenAI/Anthropic APIs, chat and monitor in the same browser page'}))
        return 0
    config = json.loads(a.config.read_text())
    source = Path(config['cwd'])
    if config.get('host') != '127.0.0.1' or not config.get('api_key'):
        raise RuntimeError('Use the authenticated loopback configuration produced by configure.py')
    command = [sys.executable,str(source/'serve/server.py'),'--engine','strata','--config',str(a.config.resolve()),
               '--host','127.0.0.1','--port',str(config['port'])]
    return subprocess.call(command,cwd=source)


if __name__ == '__main__':
    raise SystemExit(main())
