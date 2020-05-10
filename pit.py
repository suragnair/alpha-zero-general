from othello.pit import OthelloArenaBuilder
if __name__ == "__main__":
    arena_builder = OthelloArenaBuilder()
    arena = arena_builder.create(True)
    arena_builder.play(arena, 2)
