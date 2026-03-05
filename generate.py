#!/usr/bin/env python3
import argparse
import sys

from core.prompts import load_config
from backends import detect_backend


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--backend', type=str, choices=['sd', 'flux'], default=None)
    args, remaining = parser.parse_known_args()

    if args.backend:
        backend = args.backend
    else:
        config, _ = load_config(args.config)
        backend = detect_backend(config)

    print(f"Detected backend: {backend}")

    if backend == 'sd':
        sys.argv = [sys.argv[0]] + ['--config', args.config] + remaining
        from backends.sd_generate import main as sd_main
        sd_main()
    elif backend == 'flux':
        sys.argv = [sys.argv[0]] + ['--config', args.config] + remaining
        from backends.flux_generate import main as flux_main
        flux_main()
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == '__main__':
    main()
