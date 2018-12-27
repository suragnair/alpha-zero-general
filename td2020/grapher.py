import numpy as np
import pandas as pd
import pylab as plt

_skiprows = 73


def fun1(file1, file2):
    df = pd.read_csv(file1, skiprows=_skiprows)

    score = df['score']
    score1 = score[(df.player == 1) & (df.game_ep == 1)]
    score2 = score[(df.player == -1) & (df.game_ep == 1)]
    arrange_both = np.arange(len(score1))
    plt.plot(arrange_both, score1, 'r--')
    plt.plot(arrange_both, score2, 'b--')

    # next file
    df1 = pd.read_csv(file2, skiprows=_skiprows)
    score = df1['score']
    score3 = score[(df.player == 1) & (df.game_ep == 1)]
    plt.plot(arrange_both, score3, 'g--')
    plt.title("Primerjava točk treh igralcev")
    plt.ylabel("Število točk")
    plt.xlabel("Iteracija igre")
    plt.legend(["Numerični*", "One-Hot*", "Naključni*"])  # todo pazi vrstni red
    plt.show()


def fun2(file):
    df = pd.read_csv(file, skiprows=_skiprows)

    player = df['player']
    act = df['act_rev']

    move1 = act[((df.act_rev == 'up') | (df.act_rev == 'down') | (df.act_rev == 'left') | (df.act_rev == 'right')) & (player == 1)]
    move2 = act[((df.act_rev == 'up') | (df.act_rev == 'down') | (df.act_rev == 'left') | (df.act_rev == 'right')) & (player == -1)]

    mine1 = act[(df.act_rev == 'mine_resources') & (player == 1)]
    mine2 = act[(df.act_rev == 'mine_resources') & (player == -1)]

    return_resources1 = act[(df.act_rev == 'return_resources') & (player == 1)]
    return_resources2 = act[(df.act_rev == 'return_resources') & (player == -1)]

    attack1 = act[((df.act_rev == 'attack_up') | (df.act_rev == 'attack_down') | (df.act_rev == 'attack_right') | (df.act_rev == 'attack_left')) & (player == 1)]
    attack2 = act[((df.act_rev == 'attack_up') | (df.act_rev == 'attack_down') | (df.act_rev == 'attack_right') | (df.act_rev == 'attack_left')) & (player == -1)]

    npc1 = act[((df.act_rev == 'npc_up') | (df.act_rev == 'npc_down') | (df.act_rev == 'npc_right') | (df.act_rev == 'npc_left')) & (player == 1)]
    npc2 = act[((df.act_rev == 'npc_up') | (df.act_rev == 'npc_down') | (df.act_rev == 'npc_right') | (df.act_rev == 'npc_left')) & (player == -1)]

    rifl1 = act[((df.act_rev == 'rifle_infantry_up') | (df.act_rev == 'rifle_infantry_down') | (df.act_rev == 'rifle_infantry_right') | (df.act_rev == 'rifle_infantry_left')) & (player == 1)]
    rifl2 = act[((df.act_rev == 'rifle_infantry_up') | (df.act_rev == 'rifle_infantry_down') | (df.act_rev == 'rifle_infantry_right') | (df.act_rev == 'rifle_infantry_left')) & (player == -1)]

    barr1 = act[((df.act_rev == 'barracks_up') | (df.act_rev == 'barracks_down') | (df.act_rev == 'barracks_right') | (df.act_rev == 'barracks_left')) & (player == 1)]
    barr2 = act[((df.act_rev == 'barracks_up') | (df.act_rev == 'barracks_down') | (df.act_rev == 'barracks_right') | (df.act_rev == 'barracks_left')) & (player == -1)]

    th1 = act[((df.act_rev == 'town_hall_up') | (df.act_rev == 'town_hall_down') | (df.act_rev == 'town_hall_right') | (df.act_rev == 'town_hall_left')) & (player == 1)]
    th2 = act[((df.act_rev == 'town_hall_up') | (df.act_rev == 'town_hall_down') | (df.act_rev == 'town_hall_right') | (df.act_rev == 'town_hall_left')) & (player == -1)]

    heal1 = act[((df.act_rev == 'heal_up') | (df.act_rev == 'heal_down') | (df.act_rev == 'heal_right') | (df.act_rev == 'heal_left')) & (player == 1)]
    heal2 = act[((df.act_rev == 'heal_up') | (df.act_rev == 'heal_down') | (df.act_rev == 'heal_right') | (df.act_rev == 'heal_left')) & (player == -1)]

    acts = ["move", "mine", "return", "attack", "npc", "infantry", "barracks", "town_hall", "heal"]

    plt.bar(acts, [len(move1), len(mine1), len(return_resources1), len(attack1), len(npc1), len(rifl1), len(barr1), len(th1), len(heal1)], width=0.8)
    plt.bar(acts, [len(move2), len(mine2), len(return_resources2), len(attack2), len(npc2), len(rifl2), len(barr2), len(th2), len(heal2)], width=0.8, bottom=[len(move1), len(mine1), len(return_resources1), len(attack1), len(npc1), len(rifl1), len(barr1), len(th1), len(heal1)])
    plt.title("Izvajanje akcij dveh igralcev")
    plt.xlabel("Akcije")
    plt.ylabel("Število izvedenih akcij")
    plt.xticks(np.arange(8), rotation=20, labels=("Premik", "Nabiranje", "Vračanje", "Napad", "Delavec", "Vojak", "Vojašnica", "Zdravljenje"))
    plt.legend(["Igralec 1", "Igralec -1"])

    plt.show()


def to_bins(x, bins):
    x = x.astype(int)
    ar = []
    for i in range(0, len(x), bins):
        ar.append(sum(x[i:i + bins]))
    return ar


def fun3(file):
    df = pd.read_csv(file, skiprows=_skiprows)
    num_bins = 10

    player = df['player']
    act = df['act_rev']

    move1 = (((df.act_rev == 'up') | (df.act_rev == 'down') | (df.act_rev == 'left') | (df.act_rev == 'right')) & (player == 1))
    move1_bin = to_bins(move1, bins=num_bins)

    mine1 = ((df.act_rev == 'mine_resources') & (player == 1))
    mine1_bin = to_bins(mine1, bins=num_bins)

    return_resources1 = ((df.act_rev == 'return_resources') & (player == 1))
    return_resources1_bin = to_bins(return_resources1, bins=num_bins)

    attack1 = (((df.act_rev == 'attack_up') | (df.act_rev == 'attack_down') | (df.act_rev == 'attack_right') | (df.act_rev == 'attack_left')) & (player == 1))
    attack1_bin = to_bins(attack1, bins=num_bins)

    npc1 = (((df.act_rev == 'npc_up') | (df.act_rev == 'npc_down') | (df.act_rev == 'npc_right') | (df.act_rev == 'npc_left')) & (player == 1))
    npc1_bin = to_bins(npc1, bins=num_bins)

    rifl1 = (((df.act_rev == 'rifle_infantry_up') | (df.act_rev == 'rifle_infantry_down') | (df.act_rev == 'rifle_infantry_right') | (df.act_rev == 'rifle_infantry_left')) & (player == 1))
    rifl1_bin = to_bins(rifl1, bins=num_bins)

    barr1 = (((df.act_rev == 'barracks_up') | (df.act_rev == 'barracks_down') | (df.act_rev == 'barracks_right') | (df.act_rev == 'barracks_left')) & (player == 1))
    barr1_bin = to_bins(barr1, bins=num_bins)

    th1 = (((df.act_rev == 'town_hall_up') | (df.act_rev == 'town_hall_down') | (df.act_rev == 'town_hall_right') | (df.act_rev == 'town_hall_left')) & (player == 1))
    th1_bin = to_bins(th1, bins=num_bins)

    heal1 = (((df.act_rev == 'heal_up') | (df.act_rev == 'heal_down') | (df.act_rev == 'heal_right') | (df.act_rev == 'heal_left')) & (player == 1))
    heal1_bin = to_bins(heal1, bins=num_bins)

    all_arrange = np.arange(len(act) / num_bins)

    sum_move_mine = np.add(move1_bin, mine1_bin)
    sum_move_mine_return = np.add(sum_move_mine, return_resources1_bin)
    sum_move_mine_return_attack = np.add(sum_move_mine_return, attack1_bin)
    sum_move_mine_return_attack_npc = np.add(sum_move_mine_return_attack, npc1_bin)
    sum_move_mine_return_attack_npc_rifl = np.add(sum_move_mine_return_attack_npc, rifl1_bin)
    sum_move_mine_return_attack_npc_rifl_barr = np.add(sum_move_mine_return_attack_npc_rifl, barr1_bin)
    sum_move_mine_return_attack_npc_rifl_barr_th = np.add(sum_move_mine_return_attack_npc_rifl_barr, th1_bin)
    sum_move_mine_return_attack_npc_rifl_barr_th_heal = np.add(sum_move_mine_return_attack_npc_rifl_barr_th, heal1_bin)

    plt.plot(all_arrange, move1_bin, color='aqua')
    plt.plot(all_arrange, sum_move_mine, color='gold')
    plt.plot(all_arrange, sum_move_mine_return, color='magenta')
    plt.plot(all_arrange, sum_move_mine_return_attack, color='green')
    plt.plot(all_arrange, sum_move_mine_return_attack_npc, color='orange')
    plt.plot(all_arrange, sum_move_mine_return_attack_npc_rifl, color='teal')
    plt.plot(all_arrange, sum_move_mine_return_attack_npc_rifl_barr, color='orchid')
    plt.plot(all_arrange, sum_move_mine_return_attack_npc_rifl_barr_th, color='lime')
    plt.plot(all_arrange, sum_move_mine_return_attack_npc_rifl_barr_th_heal, color='lightblue')

    plt.fill_between(all_arrange, sum_move_mine_return_attack_npc_rifl_barr_th_heal, color='lightblue')
    plt.fill_between(all_arrange, sum_move_mine_return_attack_npc_rifl_barr_th, color='lime')
    plt.fill_between(all_arrange, sum_move_mine_return_attack_npc_rifl_barr, color='orchid')
    plt.fill_between(all_arrange, sum_move_mine_return_attack_npc_rifl, color='teal')
    plt.fill_between(all_arrange, sum_move_mine_return_attack_npc, color='orange')
    plt.fill_between(all_arrange, sum_move_mine_return_attack, color='green')
    plt.fill_between(all_arrange, sum_move_mine_return, color='magenta')
    plt.fill_between(all_arrange, sum_move_mine, color='gold')
    plt.fill_between(all_arrange, move1_bin, color='aqua')

    plt.legend(["Premik", "Nabiranje zlatnikov", "Vračanje zlatnikov", "Napad", "Urjenje delavca", "Urjenje vojaške enote", "Izgradnja vojašnice", "Zdravljenje"])
    plt.title("Izvajanje akcij skozi učni postopek igralca 1")
    plt.xlabel("Število izbranih akcij")
    plt.ylabel("Število odigranih akcij v koših po " + str(int(num_bins / 2)))
    plt.show()


def fun4(file):
    df = pd.read_csv(file, skiprows=_skiprows)
    num_bins = 10

    player = df['player']
    player1 = player[player == 1]
    output_directionleft_1 = ((df.output_direction == 'left') & (player == 1)) * -1  # multiply with minus for left
    output_directionright_1 = ((df.output_direction == 'right') & (player == 1))

    output_directionleft_2 = ((df.output_direction == 'left') & (player == -1)) * -1  # multiply with minus for left
    output_directionright_2 = ((df.output_direction == 'right') & (player == -1))

    arrange_both = np.arange(len(player1) / (num_bins / 2))

    output_direction_1 = np.add(output_directionleft_1, output_directionright_1)
    output_direction_2 = np.add(output_directionleft_2, output_directionright_2)

    output_direction1_bin = to_bins(output_direction_1, bins=num_bins)
    output_direction2_bin = to_bins(output_direction_2, bins=num_bins)

    plt.plot(output_direction1_bin, arrange_both, 'r--')
    plt.plot(output_direction2_bin, arrange_both, 'b--')
    plt.legend(["Igralec 1", "Igralec -1"])
    plt.title("Smer akcij (levo, desno) v koših po " + str(int(num_bins / 2)))
    plt.ylabel("Število smernih akcij")
    plt.xlabel("Smer izvajanja akcije")
    plt.show()


fun1(".\\..\\temp\\config_pit.csv", ".\\..\\temp\\config_pit.csv")  # todo
fun2(".\\..\\temp\\config_pit.csv")  # todo
fun3(".\\..\\temp\\config_pit.csv")  # todo
fun4(".\\..\\temp\\config_pit.csv")  # todo
