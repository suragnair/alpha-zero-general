import os

import matplotlib.pyplot as plt

from td2020.src.config import PATH, MAKE_STATS


class Stats:
    @staticmethod
    def clear():
        if not MAKE_STATS:
            return
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
            from td2020.src.config import EXCLUDE_IDLE, MONEY_INC, USE_TIMEOUT, HEAL_AMOUNT, HEAL_COST, DAMAGE, USE_ONE_HOT_ENCODER
            param_line = str(EXCLUDE_IDLE) + " " + str(MONEY_INC) + " " + str(USE_TIMEOUT) + " " + str(HEAL_AMOUNT) + " " + str(HEAL_COST) + " " + str(DAMAGE) + " " + str(USE_ONE_HOT_ENCODER) + " "
            param_line += str(learn_args.numIters) + " " + str(learn_args.numEps) + " " + str(learn_args.numMCTSSims) + " " + str(learn_args.arenaCompare) + " " + str(learn_args.cpuct) + " "
            param_line += str(nnet_args.lr) + " " + str(nnet_args.epochs) + " " + str(nnet_args.batch_size)
            f.write(param_line + "\n")
            f.close()

    @staticmethod
    def game_end(tick: int, player_name: int, reason: str):
        if not MAKE_STATS:
            return
        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "gameEndTurn"))
        with open(PATH + "\\..\\stats\\" + "gameEndTurn" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + str(player_name) + " " + reason + "\n")

    @staticmethod
    def killed_by(tick: int, unit_type: int, kill_reason: str):
        if not MAKE_STATS:
            return
        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "unitKilledBy"))
        with open(PATH + "\\..\\stats\\" + "unitKilledBy" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + str(unit_type) + " " + kill_reason + "\n")

    @staticmethod
    def action(tick: int, action_name: str):
        if not MAKE_STATS:
            return
        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "actionExecuted"))
        with open(PATH + "\\..\\stats\\" + "actionExecuted" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + action_name + "\n")

    @staticmethod
    def sum(tick: int, board, player_name: int):
        if not MAKE_STATS:
            return
        from td2020.src.config import P_NAME_IDX, HEALTH_IDX, A_TYPE_IDX, MONEY_IDX

        sum_health = 0
        player_money = 0
        for y in range(board.n):
            for x in range(board.n):
                if board[x][y][P_NAME_IDX] == player_name:
                    # add all actor health if not gold
                    if board[x][y][A_TYPE_IDX] != 1:
                        sum_health += board[x][y][HEALTH_IDX]
                    player_money = board[x][y][MONEY_IDX]

        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "sum"))
        with open(PATH + "\\..\\stats\\" + "sum" + "\\file_" + str(file_count) + ".txt", "a") as f:
            f.write(str(tick) + " " + str(sum_health) + " " + str(player_money) + " " + str(player_name) + "\n")

    @staticmethod
    def plot_game_end():
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
            # plt.plot(ticks)
            # plt.plot(player_names, 'r--')
            # plt.plot(reasons, 'b--')
            plt.plot(all)
            plt.show()

    @staticmethod
    def plot_killed_by():
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
            # plt.plot(unit_types)
            # plt.plot(kill_reasons, 'r--')
            plt.plot(kill_reasons, 'ro')
            plt.show()

    @staticmethod
    def plot_action():
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
            plt.title("Actions " + params)
            # plt.plot(ticks)
            # plt.plot(action_names, 'r--')
            plt.plot(ticks, action_names, 'ro')
            plt.show()

    @staticmethod
    def plot_sum():
        file_count = len(os.listdir(PATH + "\\..\\stats\\" + "sum"))
        with open(PATH + "\\..\\stats\\" + "sum" + "\\file_" + str(file_count) + ".txt", "r") as f:
            x = f.readlines()
            params = x[0]
            x = x[1:]
            ticks = []
            sum_healths = []
            player_moneys = []
            player_names = []

            all = []
            for line in x:
                tick, sum_health, player_money, player_name = line.split()
                ticks.append(int(tick))
                sum_healths.append(int(sum_health))
                player_moneys.append(int(player_money))
                player_names.append(int(player_name))

                all.append([int(tick), int(sum_health), int(player_money), int(player_name)])
            plt.title("Summaries" + params)
            # plt.plot(ticks)
            # plt.plot(player_names, 'r--')
            # plt.plot(sum_healths, 'b--')
            # plt.plot(player_moneys, 'g--')

            plt.plot(player_moneys, 'ro')
            plt.show()


Stats.plot_killed_by()
