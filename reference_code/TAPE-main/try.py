import argparse


parser = argparse.ArgumentParser(description='KGTL')
parser.add_argument("--model", type=str, required=True, help="choose model")
parser.add_argument("--dataset", type=str, required=True, help="choose dataset")
parser.add_argument('--config', default="", metavar="FILE", help="Path to config file") 
args = parser.parse_args()
