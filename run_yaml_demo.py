#!/usr/bin/env python3
"""Run main.py using a YAML config file by converting YAML keys to CLI args.
Usage:
    python tools/run_yaml_demo.py yaml/Case1.yml
"""
import sys
import shlex
import subprocess
import yaml

if len(sys.argv) < 2:
    print("Usage: python tools/run_yaml_demo.py <path_to_yaml>")
    sys.exit(1)

cfg_path = sys.argv[1]
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

args = []
for k, v in cfg.items():
    # boolean flags
    if isinstance(v, bool):
        if v:
            # boolean flags are passed as presence-only long options
            args.append(f"--{k}")
    else:
        # use --key=value form to avoid shell/argparse prefix ambiguities
        args.append(f"--{k}={v}")

cmd = [sys.executable, 'main.py'] + args
print('Running:', ' '.join(shlex.quote(c) for c in cmd))
ret = subprocess.call(cmd)
if ret != 0:
    sys.exit(ret)
