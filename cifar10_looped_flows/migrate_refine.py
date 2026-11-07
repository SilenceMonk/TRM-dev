"""Compatibility entry point for the archived convolutional experiment."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    directory = Path(__file__).resolve().parents[1] / 'cifar10_looped_flows_conv_archive'
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / 'migrate_refine.py'), run_name='__main__')
