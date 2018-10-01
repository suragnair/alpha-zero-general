import time
# noinspection PyUnresolvedReferences
import unreal_engine as ue
# noinspection PyUnresolvedReferences
from TFPluginAPI import TFPluginAPI
from tensorflow.python.keras.backend import clear_session

from MCTS import MCTS
from td2020.TD2020Game import TD2020Game
from td2020.keras.NNet import NNetWrapper as NNet
from td2020.src.config import NUM_ACTS, ACTS_REV
from utils import dotdict
import numpy as np


class TD2020LearnAPI(TFPluginAPI):

    # expected api: setup your model for your use cases
    def onSetup(self):
        # setup or load your model and pass it into stored

        # Usually store session, graph, and model if using keras

        import os
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, 'temp/')

        self.g = TD2020Game(8)

        # nnet players
        self.n1 = NNet(self.g)
        self.n1.load_checkpoint(dirname, 'temp.pth.tar')
        args = dotdict({'numMCTSSims': 500000, 'cpuct': 1.0})
        self.mcts = MCTS(self.g, self.n1, args)

        self.recommended_act = None

    # expected api: storedModel and session, json inputs
    def onJsonInput(self, jsonInput):


        if self.recommended_act:
            act1 = self.recommended_act
            self.recommended_act = None
            return act1


        """
        this function is synced with game
        :param jsonInput:
        :return:
        """
        # ue.print_string(jsonInput)
        # now parse this input:
        encoded_actors = jsonInput['data']

        ue.print_string(encoded_actors)

        initial_board_config = []
        for encoded_actor in encoded_actors:
            ue.print_string(encoded_actor)
            initial_board_config.append(
                dotdict({
                    'x': encoded_actor['x'],
                    'y': encoded_actor['y'],
                    'player': encoded_actor['player'],
                    'a_type': encoded_actor['actorType'],
                    'health': encoded_actor['health'],
                    'carry': encoded_actor['carry'],
                    'gold': encoded_actor['money'],
                    'timeout': 100
                })
                # TODO - TODO -THIS TIMEOUT IS HARDCODED
            )

        self.initial_board_config = initial_board_config



        t = time.time()
        self.g.setInitBoard(initial_board_config)
        b = self.g.getInitBoard()

        n1p = lambda x: np.argmax(self.mcts.getActionProb(x, temp=0))

        recommended_act = n1p(self.g.getCanonicalForm(b, 1))

        print("took", time.time() - t, recommended_act)


        return {"action": str(recommended_act)}


    # expected api: no params forwarded for training? TBC
    def onBeginTraining(self):
        """
        this function is Async
        :return:
        """
        clear_session()

        t = time.time()
        self.g.setInitBoard(self.initial_board_config)
        b = self.g.getInitBoard()

        n1p = lambda x: np.argmax(self.mcts.getActionProb(x, temp=0))

        recommended_act = n1p(self.g.getCanonicalForm(b, 1))


        y, x, action_index = np.unravel_index(recommended_act, [b.shape[0], b.shape[0], NUM_ACTS])

        # print("numpy action index", np.ravel_multi_index((y, x, action_index), (n, n, NUM_ACTS)))

        ue.print_string("took "+ str(time.time() - t))

        self.recommended_act = {"x": str(x), "y": str(y), "action": ACTS_REV[action_index]}



        return ""

    def run(self, args):
        pass


# required function to get our api
def getApi():
    return TD2020LearnAPI.getInstance()
