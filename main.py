import numpy as np

class Game:
    def __init__(self, n, m):
        self.n = n
        self.initialize_game(n, m)

    def initialize_game(self, n, m):
        # Empty cell is represented with a zero
        self.board = np.zeros((n, m))

    def draw_board(self):
        # Convert numbers to pictorial representation
        num_rows = self.board.shape[0]
        num_cols = self.board.shape[1]

        # visual_board = num_rows*[num_cols*["."].copy()].copy()

        visual_row = ["." for _ in range(num_cols)]
        visual_board = [visual_row.copy() for _ in range(num_rows)]

        # Convert ones to Xs and twos to Os

        for (row, col), _ in np.ndenumerate(self.board):

            if self.board[row, col] == 1:
                visual_board[row][col] = "X"
            elif self.board[row, col] == 2:
                visual_board[row][col] = 'O'

        print("The board looks like this: \n")

        # Print index of the column
        print('    ', end = '')
        print(*[col for col in range(num_cols)])

        # Separate colum index with new line
        print(f"\n")
        
        for row_idx, row in enumerate(visual_board):
            # Print index of the row
            print(row_idx, end='   ')
            print(*row, sep= ' ')

myGame = Game(10, 5)

myGame.board[0][0] = 2

myGame.board[-1][:] = 1

print(myGame.board)

myGame.draw_board()

if __name__ == "__main__":
    print("Coursework started!")