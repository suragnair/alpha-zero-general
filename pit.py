from othello.pit import OthelloArenaBuilder
if __name__ == "__main__":
    arena_builder = OthelloArenaBuilder(verbose=True, nr_games=2)
    arena = arena_builder.create(human_vs_cpu=True)
    arena_builder.play(arena)
