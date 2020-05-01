import os
import sys
import time

import chainer
import chainer.functions as F
import numpy as np
from chainer import optimizers, cuda, serializers, training
from chainer.dataset import concat_examples
from chainer.iterators import SerialIterator
from chainer.training import extensions
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet
from .OthelloNNet import OthelloNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'device': 0 if chainer.cuda.available else -1,  # GPU device id for training model, -1 indicates to use CPU.
    'num_channels': 512,
    'out': 'result_chainer',  # Output directory for chainer
    'train_mode': 'trainer'  # 'trainer' or 'custom_loop' supported.
})


def converter(batch, device=None):
    """Convert arrays to float32"""
    batch_list = concat_examples(batch, device=device)
    xp = cuda.get_array_module(batch_list[0])
    batch = tuple([xp.asarray(elem, dtype=xp.float32) for elem in batch_list])
    return batch


class NNetWrapper(NeuralNet):

    def __init__(self, game):
        super(NNetWrapper, self).__init__(game)
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        device = args.device
        if device >= 0:
            chainer.cuda.get_device_from_id(device).use()  # Make a specified GPU current
            self.nnet.to_gpu()

    def train(self, examples):
        if args.train_mode == 'trainer':
            self._train_trainer(examples)
        elif args.train_mode == 'custom_loop':
            self._train_custom_loop(examples)
        else:
            raise ValueError("[ERROR] Unexpected value args.train_mode={}"
                             .format(args.train_mode))

    def _train_trainer(self, examples):
        """Training with chainer trainer module"""
        train_iter = SerialIterator(examples, args.batch_size)
        optimizer = optimizers.Adam(alpha=args.lr)
        optimizer.setup(self.nnet)

        def loss_func(boards, target_pis, target_vs):
            out_pi, out_v = self.nnet(boards)
            l_pi = self.loss_pi(target_pis, out_pi)
            l_v = self.loss_v(target_vs, out_v)
            total_loss = l_pi + l_v
            chainer.reporter.report({
                'loss': total_loss,
                'loss_pi': l_pi,
                'loss_v': l_v,
            }, observer=self.nnet)
            return total_loss

        updater = training.StandardUpdater(
            train_iter, optimizer, device=args.device, loss_func=loss_func, converter=converter)
        # Set up the trainer.
        trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.out)
        # trainer.extend(extensions.snapshot(), trigger=(args.epochs, 'epoch'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'main/loss_pi', 'main/loss_v', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.run()

    def _train_custom_loop(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optimizers.Adam(alpha=args.lr)
        optimizer.setup(self.nnet)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            # self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                xp = self.nnet.xp
                boards = xp.array(boards, dtype=xp.float32)
                target_pis = xp.array(pis, dtype=xp.float32)
                target_vs = xp.array(vs, dtype=xp.float32)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_loss = l_pi.data
                v_loss = l_v.data
                pi_losses.update(cuda.to_cpu(pi_loss), boards.shape[0])
                v_losses.update(cuda.to_cpu(v_loss), boards.shape[0])
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                self.nnet.cleargrads()
                total_loss.backward()
                optimizer.update()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        xp = self.nnet.xp
        board = xp.array(board, dtype=xp.float32)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            board = xp.reshape(board, (1, self.board_x, self.board_y))
            pi, v = self.nnet(board)
        return np.exp(cuda.to_cpu(pi.array)[0]), cuda.to_cpu(v.array)[0]

    def loss_pi(self, targets, outputs):
        return -F.sum(targets * outputs) / targets.shape[0]

    def loss_v(self, targets, outputs):
        return F.mean_squared_error(targets[:, None], outputs)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            # print("Checkpoint Directory exists! ")
            pass
        print('Saving model at {}'.format(filepath))
        serializers.save_npz(filepath, self.nnet)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        serializers.load_npz(filepath, self.nnet)
