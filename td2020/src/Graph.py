"""
Graph.py

Helper functions that define damage  and num_destroys formulas for 'kill function' end condition.
Graph can also be displayed, which displays how much damage was applied to how many player actors on certain move
"""


def num_destroys(time):
    return int((time / 256) ** 2 + 1)


def damage(time):
    return int((time / 8) ** 2.718 / (time * 8))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    all_damage_amounts = []
    all_num_destoys_per_turn = []
    for current_time_local in range(1, 2048):
        all_num_destoys_per_turn.append(num_destroys(current_time_local))
        all_damage_amounts.append(damage(current_time_local))
    # print(all_damage_amounts)
    plt.plot(all_damage_amounts)
    plt.plot(all_num_destoys_per_turn, 'r--')
    plt.title("")
    plt.xlabel('Število izvedenih potez')  # Eng. 'Number of moves'
    plt.ylabel('Število poškodovanih figur / Škoda figur')  # Eng. 'Number of damaged actors/ damage of actors'
    plt.axis([0, 2500, 0, 64])  # 64 as in max actors on field
    plt.annotate('Število poškodovanih figur', xy=(1300, num_destroys(1300)), xytext=(1600, 20), arrowprops=dict(facecolor='black', shrink=0.1), )  # Eng. 'Number of damaged actors'
    plt.annotate('Škoda figur', xy=(750, damage(750)), xytext=(150, 45), arrowprops=dict(facecolor='black', shrink=0.1), )  # Eng. 'Damage of actors'
    plt.show()
