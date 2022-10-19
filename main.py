import numpy as np
from easyAI import TwoPlayerGame, Human_Player


class ConnectFour(TwoPlayerGame):
    """
    The game of Connect Four. Description of the game:
    https://en.wikipedia.org/wiki/Connect_Four
    Authors: Bartosz Kamiński s20500, Michał Czerwiak s21356
    The game is between human player and AI.
    It uses EasyAI - an artificial intelligence framework for two-players abstract games.
    To compile it you need to install easyAI and numpy package
    """

    def __init__(self, players, board=None):
        """
        Initialization of the game
        :param players: players from easyAI framework
        :param board: array passed to game
        """
        self.players = players
        self.board = (
            board
            if (board is not None)
            else (np.array([[0 for i in range(7)] for j in range(6)]))
        )
        self.current_player = 1  # player 1 starts.

    def possible_moves(self):
        """
        :return: all moves allowed in the game
        """
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    def make_move(self, column):
        """
        Transforms the game according to the move
        :param column: int
        """
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.current_player

    def show(self):
        """
        Function shows the gameboard and the game
        """
        print("\n" + "\n".join(["0 1 2 3 4 5 6", 13 * "-"]))

        print("\n"
              .join([" "
                    .join([[".", "O", "X"][self.board[5 - j][i]] for i in range(7)]) for j in range(6)]))

    def lose(self):
        """
        :return: return value from check_winner method
        """
        return check_winner(self.board, self.opponent_index)

    def is_over(self):
        """
        Checks whether the game has ended
        """
        return (self.board.min() > 0) or self.lose()

    def scoring(self):
        """
        :return: score to the current game
        """
        return -100 if self.lose() else 0


boardHeight = 5
boardWidth = 6


def check_winner(board, player):
    """
    Function checks fields.
    Returns True if the player has connected  4 (or more).
    """
    # check horizontal spaces
    for y in range(boardHeight):
        for x in range(boardWidth - 3):
            if board[x][y] == player and board[x + 1][y] == player and board[x + 2][y] == player and board[x + 3][
                y] == player:
                return True

    # check vertical spaces
    for x in range(boardWidth):
        for y in range(boardHeight - 3):
            if board[x][y] == player and board[x][y + 1] == player and board[x][y + 2] == player and board[x][
                y + 3] == player:
                return True

    # check / diagonal spaces
    for x in range(boardWidth - 3):
        for y in range(3, boardHeight):
            if board[x][y] == player and board[x + 1][y - 1] == player and board[x + 2][y - 2] == player and \
                    board[x + 3][y - 3] == player:
                return True

    # check \ diagonal spaces
    for x in range(boardWidth - 3):
        for y in range(boardHeight - 3):
            if board[x][y] == player and board[x + 1][y + 1] == player and board[x + 2][y + 2] == player and \
                    board[x + 3][y + 3] == player:
                return True

    return False


if __name__ == "__main__":
    # Start of the game

    from easyAI import AI_Player, Negamax

    """
    The standard AI algorithm of easyAI is Negamax with alpha-beta pruning
    and (optionnally), transposition tables.
    """

    neg_alg = Negamax(5)
    game = ConnectFour([AI_Player(neg_alg), Human_Player()])
    game.play()
    if game.lose():
        print("Player %d wins." % game.opponent_index)
    else:
        print("Looks like we have a draw.")