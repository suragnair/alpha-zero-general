from othello.pit import OthelloArenaBuilder
if __name__ == "__main__":
    factory = OthelloArenaBuilder()
    arena = factory.create(True)
    factory.play(arena, 2)
