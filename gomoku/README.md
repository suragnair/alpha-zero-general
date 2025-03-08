# Othello

[Othello](https://en.wikipedia.org/wiki/Reversi) is a two player board game, usually played on an 8x8 board.

### Go Text Protocol Player

[Go Text Protocol](https://en.wikipedia.org/wiki/Go_Text_Protocol) (GTP) is a text protocol that allows Go (the game, not the programming language) programs to play with each other. Due to the similarity in the board representation the protocol is used by Othello/Reversi programs.
`GTPOthelloPlayer` allows a game to be played with external Othello program. This could be useful to test Othello player versus existing Othello programs.

#### Usage

Creating `GTPOthelloPlayer` instance and using it is straightforward:

    player = GTPOthelloPlayer(game, ["/path/to/bin/executable", "-gtp", "-l", "10"])
    player.startGame() # runs he external program
    # the game loop
    player.endGame() # stops the external program

`/path/to/bin/executable` is the absolute path to the executable of the Othello program and the rest of the list are the arguments passed to it during startup time. If you're using `Arena` you must not call `startGame` and `endGame`, as it already does it:

    randomPlayer = RandomPlayer(game).play
    gtpPlayer = GTPOthelloPlayer(game, ["/path/to/bin/executable", "-gtp", "-l", "10"])
    
    arena = Arena(randomPlayer, gtpPlayer, game, display=OthelloGame.display)

Notice that the `GTPOthelloPlayer` instance is passed to `Arena`, not a reference to `play`. This allows `Arena` to call various methods, such as `startGame` and `endGame`, mentioned above.

#### Limitations

The different Othello programs has varying support for GTP, which is not intended to play Othello in first place, so there are some limitations when using this player. The implementation was tested with [Egaroucid](https://github.com/Nyanyan/Egaroucid) and [Edax](https://github.com/abulmo/edax-reversi), but hopefully it would work with other programs as well.

* GTP allows varying board sizes, but both Egaroucid and Edax support only 8x8 board
* Player `1` (not `-1`) always starts the game. This aligns with how Arena handles the games
* There is no way to set the state on each move. To account for that after each move of the opponent `notify` must be called. It will sent the move to the external programs so it has the same state about the game. It must not be called for the move made by `GTPOthelloPlayer` as the external program has already modified it own state with it. `Arena` already does that so you don't have to do it yourself if you're using it.
