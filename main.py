""" Assignment 2 Experiment Runner

This script will run experiments for the following problems:
	1. Four Peaks with RHC, SA, GA, and MIMIC
    2. Continuous Peaks with RHC, SA, GA, and MIMIC
    3. Knapsack with SA, GA, and MIMIC
    4. NN
"""

import argparse
import warnings
from experiments import run_experiments

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("-e", help = "Run specific experiment: fp, cp, om")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.e.lower() not in ['fp', 'cp', 'om']:
        raise ValueError("Invalid experiment, please select from following: fp, cp, om")
    else:
        run_experiments(experiment=args.e)

if __name__ == "__main__":
    main()