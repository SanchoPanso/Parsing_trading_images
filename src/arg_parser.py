import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default="")
    parser.add_argument('--url', '-u', default="")
    parser.add_argument('--output', default="output.json")
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    print(get_args())
