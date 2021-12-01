import numpy as np
from numpy.lib.utils import who

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

        # Check if anyone won
        who_won = self.has_anyone_won(board)
        has_anyone_won =  who_won is not None

        # Check if board is full
        is_full = not np.any(board == 0)
        
        # If board is full or anyone won return True and who won as a second output
        return ((is_full or has_anyone_won), who_won)


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


    def actions(self, board):
        """ Returns allowed actions as a list of cells that can be played from this board state."""
        num_cols = board.shape[1]
        # Check if dropping element in a column is valid
        valid_cols = []
        for col in range(num_cols):
            if self.is_valid(board, col):
                valid_cols.append(col)

        # Once identified the valid column check in which row the element will drop
        valid_rows = []
        for col in valid_cols:
            idx_empty = np.where(board[:,col] == 0)
            # Find lowest row with an empty cell
            lowest_empty_row = np.max(idx_empty)
            valid_rows.append(lowest_empty_row)
        
        # print(f"Valid rows: {valid_rows}")
        # print(f"Valid cols: {valid_cols}")

        valid_cells = list(zip(valid_rows, valid_cols))

        # print(f"Valid cells: {valid_cells}")

        return valid_cells

    def action_result(self, board, cell, content):
        """Returns the board resulting from filling the cell."""

        # Only allow to fill cell with a 1 (X) or 2 (O).
        assert content == 1 or content == 2, "You can only fill a cell with a 1 or a 2."

        resultant_board = board.copy()
        resultant_board[cell] = content

        return resultant_board

    def max(self, board, alpha, beta):
        """Returns maximum score for max among all the states achievable from the current state. """

        # Check if the board is in a terminal state (leaf node)
        if self.is_terminal(board)[0]:
            who_won = self.is_terminal(board)[1]
            # Return utility 1 if Max won and -1 if Min won, 0 if it is a draw
            if who_won == 1:
                return 1
            elif who_won == 2:
                return -1
            else:
                return 0

        # Define v as the maximum score you can get from this state if both play optimally.
        v = -np.inf

        valid_actions = self.actions(board)


        # valid_actions = [valid_actions[0]]
        # print(f"valid actions: {valid_actions}")

        for action in valid_actions:
            # print(f"action max: {action}")
            # resulting board of filling cell with an 1 (X)
            res_board = self.action_result(board, action, 1)
            # print(f"res board: \n {res_board}")
            res_score = self.min(res_board, alpha, beta)
            # print(f"res score: {res_score}")
            # resulting score if min playes next turn
            v = max(v, res_score)
             
            # Do alpha beta pruning
            if v >= beta:
                return v
            alpha = max(alpha, v)
            # print(f"res_score type: {type(res_score)}")

        #print("I checked maxs action {action}")
            

        return v


    def min(self, board, alpha, beta):
        """Returns minimum score for max among all the states achievable from the current state. """

        # Check if the board is in a terminal state (leaf node)
        if self.is_terminal(board)[0]:
            who_won = self.is_terminal(board)[1]
            # Return utility 1 if Max won and -1 if Min won, 0 if it is a draw
            if who_won == 1:
                return 1
            elif who_won == 2:
                return -1
            else:
                return 0

        # Define v as the minimum score you can get from this state if both play optimally.
        v = np.inf

        valid_actions = self.actions(board)

        # valid_actions = [valid_actions[0]]

        for action in valid_actions:
            # print(f"action min: {action}")
            # resulting board of filling cell with an 2 (O)
            res_board = self.action_result(board, action, 2)
            # print(f"res board: \n {res_board}")
            # resulting score if max plays next turn
            res_score = self.max(res_board, alpha, beta)
            # print(f"res score: {res_score}")
            v = min(v, res_score)

            # Do alpha beta pruning
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def minimax_decision(self, board, player = "Max"):
        """Recommend action assuming it is Max turn by default."""
        # Do a terminal check
        if self.is_terminal(board)[0]:
            print("It is terminal!")
            return None

        valid_actions = self.actions(board)

        if player == "Max":

            children_min_value = []
            for action in valid_actions:
                res_board = self.action_result(board, action, 1)
                res_score = self.min(res_board, -np.inf, +np.inf)
                children_min_value.append(res_score)

            # print(f"Children min value for max: {children_min_value}")
            max_value = max(children_min_value)
            idx = children_min_value.index(max_value)
            recommended_action = valid_actions[idx]

        if player == "Min":

            children_max_value = []
            for action in valid_actions:
                res_board = self.action_result(board, action, 2)
                res_score = self.max(res_board, -np.inf, +np.inf)
                children_max_value.append(res_score)

            # print(f"Children max value for min: {children_max_value}")
            min_value = min(children_max_value)
            idx = children_max_value.index(min_value)
            recommended_action = valid_actions[idx]


        return recommended_action

    def play(self):
        """Start the game assuming it is maxs turn."""

        has_game_finished = False

        # Helper function terminal check and announce
        def terminal_check_announce(board):
            # Do a terminal check
            has_game_finished = self.is_terminal(self.board)[0]
            if has_game_finished:
                who_won = self.is_terminal(self.board)[1]
                if who_won == 1:
                    print("Congratulations Max won!")
                elif who_won == 2:
                    print("Unfortunaltely Min won.")
                else:
                    print("It''s a draw!")
                return True
            # Otherwise
            return False

        while not has_game_finished:

            # Draw board
            self.draw_board()

            # Compute the minimax strategy for max.
            recommended_decision =  self.minimax_decision(self.board)

            # Recommend decision
            print(f"Max should play {recommended_decision} now.")

            # Prompt the user to move
            max_move = input("Enter column where to drop the X: ")

            # Convert to tuple
            user_col = int(max_move)

            # Check if is valid
            is_user_move_valid = self.is_valid(self.board, user_col)

            # Check valid actions of board
            valid_actions = self.actions(self.board)

            # Find resultant action of dropping element in user_col
            for row, col in valid_actions:
                if col == user_col:
                    tuple_move = (row, col)

            # Change the board
            self.board = self.action_result(self.board, tuple_move, 1)

            # Draw board
            print(f"After your move the board looks like this: ")
            self.draw_board()

            # Do a terminal check
            if terminal_check_announce(self.board):
                has_game_finished = True
                break

            # Calculate min's move after user's input
            # Compute the minimax strategy for min.
            recommended_decision_min =  self.minimax_decision(self.board, player="Min")
            print(f"Min played: {recommended_decision_min}")

            # Change the board
            self.board = self.action_result(self.board, recommended_decision_min, 2)

            # Draw board
            print(f"After Min''s move the board looks like this: ")
            self.draw_board()
            
            # Do a terminal check
            if terminal_check_announce(self.board):
                has_game_finished = True


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

def test_max():

    # Initialize game
    myGame = Game(n=4, m=4, k=4)

    # Create a full board
    # myGame.board[:,:] = 0

    # myGame.board[7:,0:2] = 2
    # myGame.board[3:7,0:2] = 1
    # myGame.board[2:3,0:2] = 2

    # myGame.board[7:,-2:] = 2
    # myGame.board[3:7,-2:] = 1
    # myGame.board[2:3,-2:] = 2

    # # myGame.board[7:,0] = 2
    # # myGame.board[4:7,0] = 1
    # # myGame.board[1:4,0] = 2


    # myGame.board[8:,3] = 2

    # # Draw the board
    # myGame.draw_board()

    # res_board = myGame.action_result(myGame.board, (1,5), 1)

    # print(res_board)

    # Check if it is terminal
    # print(myGame.min(myGame.board))
   
    #print(myGame.actions(myGame.board))

    # Check recommended decision
    # print(myGame.minimax_decision(myGame.board))

    myGame.play()



if __name__ == "__main__":
    # test_draw_board()
    # test_is_valid()
    # test_is_terminal()
    # test_has_anyone_won()
    test_max()