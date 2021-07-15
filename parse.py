import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int,
                        default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('-sz', '--data_size', type=int,
                        default=None,
                        help='number of pairs for the dataset')
    parser.add_argument('-aug', '--augmented', type=str2bool,
                        default=True,
                        help='use augmented data or not (default: True)')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=16,
                        help='batch size (default: 16)')
    parser.add_argument('-l', '--nb_layers', type=int,
                        default=3,
                        help='number of layers in the MLP of SyncI3d (default: 3)')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.1,
                        help='learning rate (default: 0.1')
    parser.add_argument('-rec', '--record', type=str2bool,
                        default=True,
                        help='record in TensorBoard or not (default: True')
    parser.add_argument('-chp', '--do_chkpts', type=str2bool,
                        default=False,
                        help='do checkpoints or not (default: False')
    parser.add_argument('-chpd', '--chp_delay', type=int,
                        default=10,
                        help='checkpoint delay in epochs (default: 10')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
