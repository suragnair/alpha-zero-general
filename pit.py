from othello.pit import OthelloPitFactory

if __name__ == "__main__":
    factory = OthelloPitFactory()
    arena = factory.create()
    factory.play(arena, 2)
