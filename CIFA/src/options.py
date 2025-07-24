import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='uav', help="uav，uav_or, bearing")
    parser.add_argument('--bs', type=int, default=33, help="batch size")
    parser.add_argument('--ep', type=int, default=50, help="epochs")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    # parser.add_argument('--csv', type=str, default='.csv', help="result file")
    parser.add_argument('--test', type=int, default=2, help="0,1,2")


    args = parser.parse_args()
    return args