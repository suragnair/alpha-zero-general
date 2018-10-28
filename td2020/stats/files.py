import os

import numpy as np
from numpy.ma import count

from td2020.src.config import PATH


class Stats:
    @staticmethod
    def clear():
        files = ["gameEndTurn",
                 "unitKilledBy",
                 "actionExecuted",
                 "sum"
                 ]
        for file_name in files:
            file_count = len(os.listdir(PATH + "\\..\\stats\\" + file_name))
            f = open(PATH + "\\..\\stats\\" + file_name + "\\" + str("file_" + str(file_count + 1)) + ".txt", 'w')

            from td2020.learn import args as learn_args
            from td2020.keras.NNet import args as nnet_args
            from td2020.src.config import MONEY_INC, USE_TIMEOUT, HEAL_AMOUNT, HEAL_COST, DAMAGE, USE_ONE_HOT_ENCODER
            param_line = str(MONEY_INC) + " " + str(USE_TIMEOUT) + " " + str(HEAL_AMOUNT) + " " + str(HEAL_COST) + " " + str(DAMAGE) + " " + str(USE_ONE_HOT_ENCODER) + " "
            param_line += str(learn_args.numIters) + " " + str(learn_args.numEps) + " " + str(learn_args.numMCTSSims) + " " + str(learn_args.arenaCompare) + " " + str(learn_args.cpuct) + " "
            param_line += str(nnet_args.lr) + " " + str(nnet_args.epochs) + " " + str(nnet_args.batch_size)
            f.write(param_line + "\n")
            f.close()

    @staticmethod
    def game_end(tick: int, player_name: int, reason: str):
        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "gameEndTurn"))
        with open(PATH + "\\..\\stats\\" + "gameEndTurn" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + str(player_name) + " " + reason + "\n")

    @staticmethod
    def killed_by(tick: int, unit_type: int, kill_reason: str):
        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "unitKilledBy"))
        with open(PATH + "\\..\\stats\\" + "unitKilledBy" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + str(unit_type) + " " + kill_reason + "\n")

    @staticmethod
    def action(tick: int, action_name: str):
        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "actionExecuted"))
        with open(PATH + "\\..\\stats\\" + "actionExecuted" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + action_name + "\n")

    @staticmethod
    def sum(tick: int, board):
        from td2020.src.config import P_NAME_IDX, HEALTH_IDX, A_TYPE_IDX, MONEY_IDX

        sum_health_p1 = 0
        player_money_p1 = 0
        sum_health_p2 = 0
        player_money_p2 = 0
        for y in range(board.n):
            for x in range(board.n):
                if board[x][y][P_NAME_IDX] == 1:
                    # add all actor health if not gold
                    if board[x][y][A_TYPE_IDX] != 1:
                        sum_health_p1 += board[x][y][HEALTH_IDX]
                    player_money_p1 = board[x][y][MONEY_IDX]
                if board[x][y][P_NAME_IDX] == -1:
                    # add all actor health if not gold
                    if board[x][y][A_TYPE_IDX] != 1:
                        sum_health_p2 += board[x][y][HEALTH_IDX]
                    player_money_p2 = board[x][y][MONEY_IDX]

        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "sum"))
        with open(PATH + "\\..\\stats\\" + "sum" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + str(sum_health_p1) + " " + str(player_money_p1) + " " + str(sum_health_p2) + " " + str(player_money_p2) + "\n")

    @staticmethod
    def plot_game_end():
        import matplotlib.pyplot as plt

        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "gameEndTurn"))
        with open(PATH + "\\..\\stats\\" + "gameEndTurn" + "\\file_" + str(file_count) + ".txt", "r") as f:
            x = f.readlines()
            params = x[0]
            x = x[1:]
            ticks = []
            player_names = []
            reasons = []
            all = []
            for line in x:
                tick, player_name, reason = line.split()
                ticks.append(int(tick))
                player_names.append(int(player_name))
                reasons.append(reason)
                all.append([int(tick), int(player_name), reason])
            plt.title("Game end " + params)
            plt.scatter(ticks, player_names, c=reasons)
            plt.xlim((0, max(ticks)))
            plt.show()

    @staticmethod
    def plot_killed_by():
        import matplotlib.pyplot as plt

        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "unitKilledBy"))
        with open(PATH + "\\..\\stats\\" + "unitKilledBy" + "\\file_" + str(file_count) + ".txt", "r") as f:
            x = f.readlines()
            params = x[0]

            x = x[1:]

            ticks = []
            unit_types = []
            kill_reasons = []
            all = []
            for line in x:
                tick, unit_type, kill_reason = line.split()
                ticks.append(int(tick))
                unit_types.append(unit_type)
                kill_reasons.append(kill_reason)
                all.append([int(tick), unit_type, kill_reason])
            plt.title("Killed by " + params)

            # plt.hist(kill_reasons)  # use this one to display how many of what happened there
            plt.plot(kill_reasons)  # ta pa za čez čas kok se je kj izbolšval bolj kot je prot vrhu, bolš je
            plt.show()

    @staticmethod
    def plot_action():
        import matplotlib.pyplot as plt

        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "actionExecuted"))
        with open(PATH + "\\..\\stats\\" + "actionExecuted" + "\\file_" + str(file_count) + ".txt", "r") as f:
            x = f.readlines()
            params = x[0]
            x = x[1:]

            ticks = []
            action_names = []
            all = []
            for line in x:
                tick, action_name = line.split()
                ticks.append(int(tick))
                action_names.append(action_name)

                all.append([int(tick), action_name])

            # plt.plot(ticks, action_names, 'ro') # tale je zakon - pove pr kerih ticking se kšna stvar dela

            # display prvih 10,000 in zadnih 10,000
            """
            fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
            axs[0].plot(ticks[-10000:], action_names[-10000:], 'ro')  # tale je zakon - pove pr kerih ticking se kšna stvar dela
            axs[1].plot(ticks[:10000], action_names[:10000], 'ro')  # tale je zakon - pove pr kerih ticking se kšna stvar dela
            fig.suptitle("Actions " + params)
            """

            # plt.plot(action_names, 'ro') # to pa pove skoz iteracije kok se je česa kj delal skoz čas

            plt.show()

    @staticmethod
    def plot_sum():
        import matplotlib.pyplot as plt

        print("NAROBE JE PLAYER KER JE VEČINA Z +1 KER JE BOARD CANONICAL K SE NARDI SUM")

        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "sum"))
        with open(PATH + "\\..\\stats\\" + "sum" + "\\file_" + str(file_count) + ".txt", "r") as f:
            x = f.readlines()
            params = x[0]
            x = x[1:]
            ticks = []
            sum_healths_p1 = []
            player_moneys_p1 = []
            sum_healths_p2 = []
            player_moneys_p2 = []
            all = []
            for line in x:
                tick, sum_health_p1, player_money_p1, sum_health_p2, player_money_p2 = line.split()
                ticks.append(int(tick))
                sum_healths_p1.append(int(sum_health_p1))
                player_moneys_p1.append(int(player_money_p1))
                sum_healths_p2.append(int(sum_health_p2))
                player_moneys_p2.append(int(player_money_p2))

                all.append([int(tick), int(sum_health_p1), int(player_money_p1), int(sum_health_p2), int(player_money_p2)])
            plt.title("Summaries" + params)

            width = 1  # the width of the bars: can also be len(x) sequence
            ind = np.arange(count(ticks))

            """
            compare health between 2 players
            p1 = plt.bar(ind, sum_healths_p1, width)
            p2 = plt.bar(ind, sum_healths_p2, width, bottom=sum_healths_p1)
             plt.ylabel('Health')
            plt.title("Healths " + params)
            plt.legend((p1[0], p2[0]), ('Player1', 'Player-1'))

            """

            plt.show()

# Stats.plot_sum()
