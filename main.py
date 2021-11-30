import numpy as np

class Game:
    def __init__(self, n, m, k):
        self.k = k
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

        # Separator line
        print()

    def is_valid(self, board, col):
        """Functions that returns True if adding an x or o into a specific column is valid."""
        # Check if the top cell of a column is empty
        is_top_empty = (board[0, col] == 0)
        
        return is_top_empty

    def is_terminal(self, board):
        """Function that checks if a board state is terminal.
        Return True if board is full or if someone won."""

        # Check if board is full
        is_full = not np.any(board == 0)
        
        # Check if anyone won
        has_anyone_won = self.has_anyone_won(board) is not None

        # If board is full or anyone won return True
        return is_full or has_anyone_won


    def has_anyone_won(self, board):
        """Function that checks if anyone has won the game at this state.
        Returns None if no one won, 1 if Max won and 2 if Min won."""

        num_rows = board.shape[0]
        num_cols = board.shape[1]
        k = self.k

        # Check if there are at least k adjacent elements in a row
        for row in range(num_rows):
            # print(f"Row number {row}")
            for col in range(0, num_cols-k+1):
                # print(f"Col number {col}")
                sub_k_horiz = board[row, col:col+k]
                # print(f"Sub k {sub_k_horiz}")
                # If found k consecutive elements someone has won
                consecutive_found = self.is_k_consecutive(sub_k_horiz)
                # print(f"consecutive_found: {consecutive_found}")
                if consecutive_found is not None:
                    return consecutive_found
    
        # Check if there are at least k adjacent elements in a column
        for col in range(num_cols):
            # print(f"col number {col}")
            for row in range(0, num_rows-k+1):
                # print(f"row number {row}")
                sub_k_vert = board[row:row+k,col]
                # print(f"Sub k {sub_k_horiz}")
                # If found k consecutive elements someone has won
                consecutive_found = self.is_k_consecutive(sub_k_vert)
                # print(f"consecutive_found: {consecutive_found}")
                if consecutive_found is not None:
                    return consecutive_found
 
        # Check diagonals starting from left edge
        for row in range(num_rows):
            # Calculate the length of diagonal starting at cell board[row, 0] and going down and to the right
            diagonal_length = min((num_rows - row), num_cols)
            if diagonal_length < k:
                continue
            else:
                diagonal = np.zeros((diagonal_length,1))
                for step_down in range(diagonal_length):
                    diagonal[step_down] = board[row+step_down, 0+step_down]
                # Check if this diagonal has k consecutive terms
                for i in range(diagonal_length-k+1):
                    sub_k_diag = diagonal[i:i+k]
                    consecutive_found = self.is_k_consecutive(sub_k_diag)
                    if consecutive_found is not None:
                        return consecutive_found

        # Check diagonals starting from top edge
        for col in range(num_cols):
            # Calculate the length of diagonal starting at cell board[0, col] and going down and to the right
            diagonal_length = min((num_cols - col), num_rows)
            if diagonal_length < k:
                continue
            else:
                diagonal = np.zeros((diagonal_length,1))
                for step_down in range(diagonal_length):
                    diagonal[step_down] = board[0+step_down, col+step_down]
                # Check if this diagonal has k consecutive terms
                for i in range(diagonal_length-k+1):
                    sub_k_diag = diagonal[i:i+k]
                    consecutive_found = self.is_k_consecutive(sub_k_diag)
                    if consecutive_found is not None:
                        return consecutive_found

        # If no 4 consecutive found return 0
        return None


    def is_k_consecutive(self, sub_k):
        """ Returns None if no consecutive, 1 if k consecutive ones, 2 if k consecutive 2s. """
        k = self.k

        is_all_ones = np.all(sub_k == 1)
        is_all_twos = np.all(sub_k == 2)

        if is_all_ones:
            return 1

        if is_all_twos:
            return 2
        
        # Otherwise
        return None


def test_draw_board():

    # Initialize game
    myGame = Game(n=10, m=5, k=4)

    # Mark the top left element with an O
    myGame.board[0, 0] = 2

    # Mark all the bottom row with Xs
    myGame.board[-1, :] = 1

    # Draw the board
    myGame.draw_board()

def test_is_valid():

    # Initialize game
    myGame = Game(n=10, m=5, k=4)

    # Mark the top left element with an O
    myGame.board[0,0] = 2

    # Mark all the bottom row with Xs
    myGame.board[-1,:] = 1

    # Draw the board
    myGame.draw_board()

    # Check if inserting something in column zero is valid
    print(myGame.is_valid(myGame.board, 0))

def test_is_terminal():

    # Initialize game
    myGame = Game(n=10, m=5, k=4)

    # Create a full board
    myGame.board[:,:] = 2
    myGame.board[3,4] = 1

    # Draw the board
    myGame.draw_board()

    # Check that it is terminal
    print(myGame.is_terminal(myGame.board))

def test_has_anyone_won():

    # Initialize game
    myGame = Game(n=10, m=6, k=4)

    # Create a full board
    myGame.board[:,:] = 0
    myGame.board[-2,:2] = 2
    myGame.board[-2,2:5] = 1
    myGame.board[:3,-1] = 2

    # Create a diagonal of 2s
    for i in range(4):
        myGame.board[i+2,i+1] = 2

    # Draw the board
    myGame.draw_board()

    # Check if anyone won
    print(myGame.has_anyone_won(myGame.board))

    # Check if it is terminal
    print(myGame.is_terminal(myGame.board))




if __name__ == "__main__":
    # test_draw_board()
    # test_is_valid()
    # test_is_terminal()
    test_has_anyone_won()