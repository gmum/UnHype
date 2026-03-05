#!/usr/bin/env python3
import argparse
import sys

from core.prompts import load_config
from backends import detect_backend


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True)
    args, remaining = parser.parse_known_args()

    config, _ = load_config(args.config)
    backend = detect_backend(config)
    print(f"Detected backend: {backend}")

    sys.argv = [sys.argv[0]] + ['--config', args.config] + remaining

    if backend == 'sd':
        from backends.sd_train import main as sd_main
        sd_main()
    elif backend == 'flux':
        from backends.flux_train import main as flux_main
        flux_main()
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == '__main__':
    main()
