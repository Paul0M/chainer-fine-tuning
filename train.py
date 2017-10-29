import argparse

import chainer
from chainer import training
from chainer.training import extensions

from net import VGG


def main():
    parser = argparse.ArgumentParser(description='Chainer example: ')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', default='filelist.txt',
                        help='List file of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console')
    parser.add_argument('--train_size', '-s', type=int, default=0)
    parser.add_argument('--pretrained_model', '-m',
                        help='Path to pretrained model file')
    args = parser.parse_args()

    # Load dataset
    train, test = chainer.datasets.get_cifar100(ndim=3)
    class_labels = 100
    if args.train_size > 0:
        train, _ = chainer.datasets.split_dataset_random(train, args.train_size, 0)
    print('train samples: {}'.format(len(train)))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Setup a model
    if args.pretrained_model:
        print('Use pretrained model: {}'.format(args.pretrained_model))
        model = VGG(class_labels, args.pretrained_model)
    else:
        print('Not use pretrained model')
        model = VGG(class_labels)
    if args.gpu >= 0:
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)

    # Freeze parameters
    if args.pretrained_model:
        model.base.disable_update()

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    display_interval = (args.display_interval, 'epoch')

    trainer.extend(extensions.Evaluator(
        test_iter, model, device=args.gpu), trigger=display_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], trigger=display_interval))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run
    trainer.run()

if __name__ == '__main__':
    main()
