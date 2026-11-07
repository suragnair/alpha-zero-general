import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from ..TicTacToe4DLogic import forced_moves
from .TicTacToe4DNNet import TicTacToe4DNNet as t4dnet

args = dotdict({
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 32,
    'num_blocks': 4,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = t4dnet(game, args)
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float32))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        squares, color = forced_moves(board)
        board = torch.FloatTensor(board.astype(np.float32))
        if args.cuda:
            board = board.cuda()
        board = board.view(1, *self.board_size)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        pi, v = torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        if len(squares):
            # With 256 squares and few simulations the search can miss such a move, and the pretrained
            # network was not taught to find it: its training engine played these moves without asking it.
            pi = np.zeros_like(pi)
            pi[squares] = 1 / len(squares)
            if color == 1:
                v = 1.0  # the line is completed on this move
            elif len(squares) > 1:
                v = -1.0  # only one of the two lines can be blocked
        return pi, v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.makedirs(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
