import os
import sys
import time
import copy

import numpy as np
import logging
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.utils.benchmark as benchmark

from .GomokuNNet import GomokuNNet as gonet

log = logging.getLogger(__name__)

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'mps': torch.backends.mps.is_available(),
    'num_channels': 128,
    'input_channels': 2,
    'early_stoping_patience': 2 # stop training if the test loss does not decrease for 2 epoch. This is to avoid overfitting.
})


class NNetWrapper(NeuralNet):
    def __init__(self, game, input_channels = 2, num_channels = 512):
        self.args = copy.copy(args)
        self.args.input_channels = input_channels
        self.args.num_channels = num_channels
        self.game = game
        self.nnet = gonet(game, self.args)
        # cpu_nnet is a copy stored on CPU to do prediction.
        self.cpu_nnet = gonet(game, self.args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        device = 'cuda' if args.cuda else 'mps' if args.mps else 'cpu'
        self.device = torch.device(device)
        self.nnet.to(self.device)  # Move the network to the chosen device

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), self.args.lr)
        example_count = len(examples)

        # Separate training and test examples
        test_sample_ids = np.random.choice(example_count, size=example_count // 10 + 1, replace=False)
        all_indices = np.arange(example_count)
        train_sample_ids = np.setdiff1d(all_indices, test_sample_ids)

        # Create training and test sets
        train_examples = [examples[i] for i in train_sample_ids]
        test_examples = [examples[i] for i in test_sample_ids]
        
        # early stoping to avoid overfitting
        best_test_loss = float('inf')
        best_epoch = 0
        no_improve_epochs = 0
        best_nnet = gonet(self.game, self.args)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = len(train_examples) // args.batch_size
            t = tqdm(range(batch_count), desc='Training Net')

            for _ in t:
                sample_ids = np.random.choice(len(train_examples), size=self.args.batch_size, replace=False)
                batch = [train_examples[i] for i in sample_ids]

                boards, pis, vs = zip(*batch)

                boards = np.array(boards, dtype=np.float32)
                pis = np.array(pis, dtype=np.float32)
                vs = np.array(vs, dtype=np.float32)
                
                black_stones = (boards == 1).astype(np.float32)
                white_stones = (boards == -1).astype(np.float32)
                boards = np.stack([black_stones, white_stones], axis=1)  # Shape: (batch_size, 2, board_x, board_y)

                # Move data to the appropriate device
                boards = torch.tensor(boards, device=self.device)
                target_pis = torch.tensor(pis, device=self.device)
                target_vs = torch.tensor(vs, device=self.device)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            pi_losses, v_losses = self.evaluate_test_set(test_examples)
            log.info(f"Test Losses - Policy Loss: {pi_losses:.4f}, Value Loss: {v_losses:.4f}")
            if pi_losses + v_losses < best_test_loss:
                best_test_loss = pi_losses + v_losses
                no_improve_epochs = 0
                # save the current best model
                best_nnet.load_state_dict(copy.deepcopy(self.nnet.state_dict()))
                best_epoch = epoch
            else:
                no_improve_epochs += 1
                if no_improve_epochs > self.args.early_stoping_patience:
                    log.info(f"early stop at epoch {epoch + 1}; saving network trained at epoch {best_epoch + 1}")
                    # early stop
                    break

        self.nnet.load_state_dict(copy.deepcopy(best_nnet.state_dict()))
        self.cpu_nnet.load_state_dict(copy.deepcopy(self.nnet.state_dict()))
        self.cpu_nnet.to('cpu')

    def evaluate_test_set(self, test_examples):
            """
            Evaluate the model on the test examples and calculate average losses.
            """
            self.nnet.eval()  # Set the network to evaluation mode

            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            with torch.no_grad():  # No gradient computation for evaluation
                for board, pi, v in test_examples:
                    board = np.array(board, dtype=np.float32)
                    black_stones = (board == 1).astype(np.float32)
                    white_stones = (board == -1).astype(np.float32)
                    board = np.stack([black_stones, white_stones], axis=0)  # Shape: (2, board_x, board_y)

                    # Move data to the appropriate device
                    board = torch.tensor(board, device=self.device).unsqueeze(0)  # Add batch dimension
                    target_pi = torch.tensor(pi, device=self.device).unsqueeze(0)  # Add batch dimension
                    target_v = torch.tensor(v, device=self.device).unsqueeze(0)  # Add batch dimension

                    # Compute output
                    out_pi, out_v = self.nnet(board)
                    l_pi = self.loss_pi(target_pi, out_pi)
                    l_v = self.loss_v(target_v, out_v)

                    # Record loss
                    pi_losses.update(l_pi.item(), board.size(0))
                    v_losses.update(l_v.item(), board.size(0))

            return pi_losses.avg, v_losses.avg
            
    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        # Preprocess the board into two channels
        # TODO: optimize it. move it to MPS first? then convert.
        if self.args.input_channels == 2:
            black_stones = (board == 1).astype(np.float32)
            white_stones = (board == -1).astype(np.float32)
            board = np.stack([black_stones, white_stones], axis=0)  # Shape: (2, board_x, board_y)
        
        # board = torch.tensor(board, dtype=torch.float32, device=self.device)
        board = torch.tensor(board, dtype=torch.float32)
        board = board.view(self.args.input_channels, self.board_x, self.board_y)
        self.cpu_nnet.eval()
        with torch.no_grad():
            pi, v = self.cpu_nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        # return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]
        return torch.exp(pi).data.numpy()[0], v.data.numpy()[0][0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None 
        if args.cuda:
            map_location = 'cuda'
        elif args.mps:
            map_location = 'mps'
        else:
            map_location = 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(filepath, map_location='cpu')        
        self.nnet.load_state_dict(checkpoint['state_dict'])
