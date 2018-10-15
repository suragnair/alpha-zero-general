import numpy as np
# noinspection PyUnresolvedReferences
import unreal_engine as ue
# noinspection PyUnresolvedReferences
from TFPluginAPI import TFPluginAPI
from tensorflow.python.keras.backend import clear_session

from MCTS import MCTS
from td2020.TD2020Game import TD2020Game
from td2020.keras.NNet import NNetWrapper as NNet
from td2020.src.config import ACTS_REV, NUM_ACTS
from utils import dotdict


class TD2020LearnAPI(TFPluginAPI):

    # expected api: setup your model for your use cases
    def onSetup(self):
        # setup or load your model and pass it into stored

        # Usually store session, graph, and model if using keras
        self.recommended_act = None

    # expected api: storedModel and session, json inputs
    def onJsonInput(self, jsonInput):
        # this function is synced with game


        if self.recommended_act:
            act1 = self.recommended_act
            self.recommended_act = None
            return act1



        # ue.print_string(jsonInput)
        # now parse this input:
        encoded_actors = jsonInput['data']

        # ue.print_string(encoded_actors)

        initial_board_config = []
        for encoded_actor in encoded_actors:
            # ue.print_string(encoded_actor)
            initial_board_config.append(
                dotdict({
                    'x': encoded_actor['x'],
                    'y': encoded_actor['y'],
                    'player': encoded_actor['player'],
                    'a_type': encoded_actor['actorType'],
                    'health': encoded_actor['health'],
                    'carry': encoded_actor['carry'],
                    'gold': encoded_actor['money'],
                    'timeout': encoded_actor['remaining']
                })
            )
        self.initial_board_config = initial_board_config
        self.owning_player = jsonInput['player']




    # expected api: no params forwarded for training? TBC
    def onBeginTraining(self):

        clear_session()
        import os
        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, 'temp/')

        g = TD2020Game(8)

        # nnet players
        n1 = NNet(g)
        n1.load_checkpoint(dirname, 'temp.pth.tar')
        # args = dotdict({'numMCTSSims': 500000, 'cpuct': 1.0})
        args = dotdict({'numMCTSSims': 5000, 'cpuct': 1.0})
        mcts = MCTS(g, n1, args)

        g.setInitBoard(self.initial_board_config)
        b = g.getInitBoard()

        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        # canonicalBoard = g.getCanonicalForm(b, 1)
        print("todo - check if this 'owning player' is ok")
        canonicalBoard = g.getCanonicalForm(b, self.owning_player)

        # self.n1.nnet.model._make_predict_function()

        recommended_act = n1p(canonicalBoard)

        # return {"action": str(recommended_act)}

        #recommended_act = n1p(g.getCanonicalForm(b, self.owning_player))

        y, x, action_index = np.unravel_index(recommended_act, [b.shape[0], b.shape[0], NUM_ACTS])

        # print("numpy action index", np.ravel_multi_index((y, x, action_index), (n, n, NUM_ACTS)))


        self.recommended_act = {"x": str(x), "y": str(y), "action": ACTS_REV[action_index]}
        print("PRINTING RECOMM ACT", self.recommended_act)
        return "{}"

    def run(self, args):
        pass


# required function to get our api
def getApi():
    return TD2020LearnAPI.getInstance()

