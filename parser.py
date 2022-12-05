import argparse
import pathlib

def getArgs():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset',
    #                     choices=('mnist', 'cifar-10', 'cifar-100', 'fashion'),
    #                     default='cifar-10',
    #                     dest='dataset',
    #                     help='provide dataset, CIFAR-10 used by default',
    #                     type=str
    #                     )
    parser.add_argument('--seed',
                        default=2222,
                        dest='seed',
                        help='provide seed number',
                        type=int
                        )
    parser.add_argument('--ex',
                        default=2000,
                        dest='exemplars',
                        help='provide exemplar memory size',
                        type=int
                        )
    parser.add_argument('--moles',
                        action='store_true',
                        dest='moles',
                        help='if model will be poisoned by moles',
                        )

    return parser.parse_args()