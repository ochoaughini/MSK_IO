import argparse
from pathlib import Path
from .api import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='Run MSK_IO pipeline')
    parser.add_argument('dicom_dir', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('vault', type=Path)
    args = parser.parse_args()
    result = run_pipeline(args.dicom_dir, args.config, args.vault)
    print(result)

if __name__ == '__main__':
    main()
