import matplotlib.pyplot as plt


def num_destroys(time):
    return int((time / 256) ** 2 + 1)


def damage(time):
    return int((time / 8) ** 2.718 / (time * 8) )


all_damage_amounts = []
all_num_destoys_per_turn = []
for current_time_local in range(1, 8192):
    all_num_destoys_per_turn.append(num_destroys(current_time_local))
    all_damage_amounts.append(damage(current_time_local))
#print(all_damage_amounts)
plt.plot(all_damage_amounts)
plt.plot(all_num_destoys_per_turn, 'r--')
plt.title("")
plt.xlabel('Število izvedenih potez')
plt.ylabel('Število poškodovanih figur / Škoda figur')
plt.axis([0, 2500, 0, 64])  # 64 as in max actors on field
plt.annotate('Število poškodovanih figur', xy=(1300, num_destroys(1300)), xytext=(1600, 20), arrowprops=dict(facecolor='black', shrink=0.1), )
plt.annotate('Škoda figur', xy=(750, damage(750)), xytext=(150, 45), arrowprops=dict(facecolor='black', shrink=0.1), )
plt.show()