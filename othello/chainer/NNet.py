import os
import time

import chainer
import chainer.functions as F
from chainer import optimizers, cuda, serializers

import numpy as np
import sys
sys.path.append('../../')
from utils import dotdict
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet


from .OthelloNNet import OthelloNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    # 'cuda': torch.cuda.is_available(),
    'device': 1,  # GPU device id for training model, -1 indicates to use CPU.
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):

    def __init__(self, game):
        super(NNetWrapper, self).__init__(game)
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # if args.cuda:
        device = args.device
        if device >= 0:
            chainer.cuda.get_device_from_id(device).use()  # Make a specified GPU current
            self.nnet.to_gpu()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optimizers.Adam(alpha=args.lr)
        optimizer.setup(self.nnet)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            # self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                # target_pis = torch.FloatTensor(np.array(pis))
                # target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                xp = self.nnet.xp
                boards = xp.array(boards, dtype=xp.float32)
                target_pis = xp.array(pis, dtype=xp.float32)
                target_vs = xp.array(vs, dtype=xp.float32)

                # predict
                # if args.cuda:
                #     boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                # boards, target_pis, target_vs = Variable(boards), Variable(target_pis), Variable(target_vs)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                try:
                    pi_loss = l_pi.data
                    v_loss = l_v.data
                    # if len(pi_loss.shape()) > 0:
                    #     pi_loss = pi_loss[0]
                    # if len(v_loss.shape()) > 0:
                    #     v_loss = v_loss[0]
                    pi_losses.update(cuda.to_cpu(pi_loss), boards.shape[0])
                    v_losses.update(cuda.to_cpu(v_loss), boards.shape[0])
                except Exception as e:
                    print('Error at pi_losses!!-------------')
                    import IPython; IPython.embed()

                # compute gradient and do SGD step
                self.nnet.cleargrads()
                # optimizer.zero_grad()
                total_loss.backward()
                optimizer.update()
                # optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                             '| Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                              batch=batch_idx,
                              size=int(len(examples)/args.batch_size),
                              data=data_time.avg,
                              bt=batch_time.avg,
                              total=bar.elapsed_td,
                              eta=bar.eta_td,
                              lpi=pi_losses.avg,
                              lv=v_losses.avg,
                              )
                bar.next()
            bar.finish()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        # board = torch.FloatTensor(board.astype(np.float64))
        # if args.cuda: board = board.contiguous().cuda()
        xp = self.nnet.xp
        board = xp.array(board, dtype=xp.float32)
        # board = Variable(board, volatile=True)
        # with torch.no_grad():
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            # board = Variable(board)
            board = xp.reshape(board, (1, self.board_x, self.board_y))
            # self.nnet.eval()
            pi, v = self.nnet(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        # return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        return np.exp(cuda.to_cpu(pi.array)[0]), cuda.to_cpu(v.array)[0]

    def loss_pi(self, targets, outputs):
        # return -torch.sum(targets * outputs) / targets.size()[0]
        return -F.sum(targets * outputs) / targets.shape[0]

    def loss_v(self, targets, outputs):
        # return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return F.mean_squared_error(targets[:, None], outputs)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        serializers.save_npz(filepath, self.nnet)
        # torch.save({
        #     'state_dict' : self.nnet.state_dict(),
        # }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        serializers.load_npz(filepath, self.nnet)
        # checkpoint = torch.load(filepath)
        # self.nnet.load_state_dict(checkpoint['state_dict'])
