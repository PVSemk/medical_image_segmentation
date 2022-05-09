from train import TrainSession
import os


if __name__ == '__main__':
    """
    Run as:
    python train.py --config config.yaml --gpu 0
    """
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess BraTS data.')
    parser.add_argument('--config', type=str, required=True, help='.yaml config file')
    parser.add_argument('--gpu', type=str, required=False, default="0", help='CUDA device id')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sess = TrainSession(config_file=args.config)
    config_net = sess.config.get("network")
    sess.val_epoch(-1)