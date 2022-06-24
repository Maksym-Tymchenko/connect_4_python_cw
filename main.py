import numpy as np
from scipy.signal import convolve2d
import time
import math
import os

class Game:
    def __init__(self, m, n, k, do_pruning = True, max_depth = math.inf):
        self.k = k
        self.initialize_game(n, m)
        # Attribute to toggle pruning on and off
        self.do_pruning = do_pruning
        # Attribute to stop search when max depth is reached
        self.max_depth = max_depth

    def initialize_game(self, n, m):
        # Empty cell is represented with a zero
        self.board = np.zeros((n, m))

    def draw_board(self):
        # Convert numbers to pictorial representation
        num_rows = self.board.shape[0]
        num_cols = self.board.shape[1]

        # Represent empty cells with a dot
        visual_row = ["." for _ in range(num_cols)]
        visual_board = [visual_row.copy() for _ in range(num_rows)]

        # Convert ones to Xs and tens to Os

        for (row, col), _ in np.ndenumerate(self.board):

            if self.board[row, col] == 1:
                visual_board[row][col] = "X"
            elif self.board[row, col] == 10:
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

        # Check if move is inside board
        num_cols = board.shape[1]
        range_cols = list(range(num_cols))
        is_in_board = col in range_cols

        if not is_in_board:
            return False

        # Check if the top cell of a column is empty
        is_top_empty = (board[0, col] == 0)

        return is_top_empty

    def is_terminal(self, board):
        """Function that checks if a board state is terminal.
        Return a tuple (bool, int), 
        with the first element being a boolean (True if state is terminal) and the second element specifying who won."""

        # Check if anyone won
        who_won = self.has_anyone_won_efficient(board)
        has_anyone_won =  who_won is not None

        # Check if board is full
        is_full = not np.any(board == 0)
        
        # If board is full or anyone won return True and who won as a second output
        return ((is_full or has_anyone_won), who_won)


    def has_anyone_won_efficient(self,board):
        """Function that checks if anyone has won the game at this state.
        Returns None if no one won, 1 if Max won and 10 if Min won. (efficiently)"""

        num_rows = board.shape[0]
        num_cols = board.shape[1]
        k = self.k

        # Create horizontal, vertical and diagonal kernels to detect k consecutive elements
        horizontal_kernel = np.ones((1,k))
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(k, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        
        for kernel in detection_kernels:
            # Convolve each kernel with board
            convolution = convolve2d(board, kernel, mode="same")

            if (convolution == k).any():
                return 1

            elif (convolution == 10*k).any():
                return 10

        return None


    def actions(self, board):
        """ Returns allowed actions as a list of tuple cells that can be played from this board state."""
        num_cols = board.shape[1]
        # Check if dropping element in a column is valid
        valid_cols = []
        for col in range(num_cols):
            if self.is_valid(board, col):
                valid_cols.append(col)

        # Once identified the valid column check in which row the element will drop
        valid_rows = []
        for col in valid_cols:
            # Find lowest row with an empty cell (where element would physically drop)
            idx_empty = np.where(board[:,col] == 0)
            lowest_empty_row = np.max(idx_empty)
            valid_rows.append(lowest_empty_row)
        
        # print(f"Valid rows: {valid_rows}")
        # print(f"Valid cols: {valid_cols}")

        valid_cells = list(zip(valid_rows, valid_cols))

        # print(f"Valid cells: {valid_cells}")

        return valid_cells

    def action_result(self, board, cell, content):
        """Returns the board resulting from filling the input cell with the input content."""

        # Only allow to fill cell with a 1 (X) or 10 (O).
        assert content == 1 or content == 10, "You can only fill a cell with a 1 or a 10."

        resultant_board = board.copy()
        resultant_board[cell] = content

        return resultant_board

    def max(self, board, alpha, beta, depth, do_pruning = True):
        """Returns maximum score for max among all the states achievable from the current state. """
        depth = depth + 1

        # Do a depth check
        # Stop searching if you looked too deep
        if depth == self.max_depth:
            # Return a heauristic of 0 (draw)
            return 0

        # Check if the board is in a terminal state (leaf node)
        if self.is_terminal(board)[0]:
            who_won = self.is_terminal(board)[1]
            # Return utility 1 if Max won and -1 if Min won, 0 if it is a draw
            if who_won == 1:
                return 1
            elif who_won == 10:
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
            res_score = self.min(res_board, alpha, beta, depth, do_pruning = self.do_pruning)
            # print(f"res score: {res_score}")
            # resulting score if min playes next turn
            v = max(v, res_score)
             
            # Do alpha beta pruning if toggled on
            if do_pruning and v >= beta:
                return v
            alpha = max(alpha, v)
            # print(f"res_score type: {type(res_score)}")

            

        return v


    def min(self, board, alpha, beta, depth, do_pruning = True):
        """Returns minimum score for max among all the states achievable from the current state. """
        depth = depth + 1


        # Do a depth check
        # Stop searching if you looked too deep
        if depth == self.max_depth:
            # Return a heauristic of 0 (draw)
            return 0

        # Check if the board is in a terminal state (leaf node)
        if self.is_terminal(board)[0]:
            who_won = self.is_terminal(board)[1]
            # Return utility 1 if Max won and -1 if Min won, 0 if it is a draw
            if who_won == 1:
                return 1
            elif who_won == 10:
                return -1
            else:
                return 0

        # Define v as the minimum score you can get from this state if both play optimally.
        v = np.inf

        valid_actions = self.actions(board)

        # valid_actions = [valid_actions[0]]

        for action in valid_actions:
            # print(f"action min: {action}")
            # resulting board of filling cell with an 10 (O)
            res_board = self.action_result(board, action, 10)
            # print(f"res board: \n {res_board}")
            # resulting score if max plays next turn
            res_score = self.max(res_board, alpha, beta, depth, do_pruning = self.do_pruning)
            # print(f"res score: {res_score}")
            v = min(v, res_score)

            # Do alpha beta pruning if toggled on
            if do_pruning and v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def minimax_decision(self, board, player = "Max"):
        """Recommend action as a tuple (row, col) representing a cell assuming it is Max turn by default."""

        # Do a terminal check
        if self.is_terminal(board)[0]:
            print("It is terminal!")
            return None

        # Identify valid actions from this state
        valid_actions = self.actions(board)

        if player == "Max":

            children_min_value = []
            for action in valid_actions:
                # Calculate the board resulting from applying a valid action
                res_board = self.action_result(board, action, 1)
                # Calculate the score of the resulting board
                res_score = self.min(res_board, -np.inf, +np.inf, depth = 0, do_pruning = self.do_pruning)
                # Collect all the scores of the children states
                children_min_value.append(res_score)

            # print(f"Children min value for max: {children_min_value}")
            # Pick the best score from the children states
            max_value = max(children_min_value)
            idx = children_min_value.index(max_value)
            # Recommend action that gives the highest score
            recommended_action = valid_actions[idx]

        if player == "Min":

            children_max_value = []
            for action in valid_actions:
                # Calculate the board resulting from applying a valid action
                res_board = self.action_result(board, action, 10)
                # Calculate the score of the resulting board
                res_score = self.max(res_board, -np.inf, +np.inf,  depth = 0, do_pruning = self.do_pruning)
                # Collect all the scores of the children states
                children_max_value.append(res_score)

            # print(f"Children max value for min: {children_max_value}")

            # Pick the lowest score (best for min) from the children states
            min_value = min(children_max_value)
            idx = children_max_value.index(min_value)
            # Recommend action that gives the lowest score (best for min)
            recommended_action = valid_actions[idx]


        return recommended_action

    def play(self, force_move = False):
        """Start the game assuming it is maxs turn."""

        has_game_finished = False

        # Helper function that does terminal check and announces winner
        def terminal_check_announce(board):
            # Do a terminal check
            has_game_finished = self.is_terminal(self.board)[0]
            if has_game_finished:
                who_won = self.is_terminal(self.board)[1]
                if who_won == 1:
                    print("Congratulations Max won!")
                elif who_won == 10:
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
            print(f"Max should play column {recommended_decision[1]} now.")

            is_user_move_valid = False

            while not is_user_move_valid:
                # Prompt the user to move
                if force_move == False:
                    max_move = input("Enter column where to drop the X: ")
                elif force_move == True:
                    max_move = recommended_decision[1] # Force user to use recommended move

                # Convert move to tuple
                user_col = int(max_move)

                # Check if is valid
                is_user_move_valid = self.is_valid(self.board, user_col)

                if not is_user_move_valid:
                    print("Your move is not valid, try again.")

            # Check valid actions of board
            valid_actions = self.actions(self.board)

            # Find resultant action of dropping element in user specified column
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
            print(f"Min played column {recommended_decision_min[1]}")

            # Change the board
            self.board = self.action_result(self.board, recommended_decision_min, 10)

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
    myGame.board[0, 0] = 10

    # Mark all the bottom row with Xs
    myGame.board[-1, :] = 1

    # Draw the board
    myGame.draw_board()

def test_is_valid():

    # Initialize game
    myGame = Game(n=10, m=5, k=4)

    # Mark the top left element with an O
    myGame.board[:,0] = 10

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
    myGame.board[:,:] = 10
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
    myGame.board[-2,:2] = 10
    myGame.board[-2,2:5] = 1
    myGame.board[:3,-1] = 10

    # Create a diagonal of 10s
    for i in range(4):
        myGame.board[i+2,i+1] = 10

    # Draw the board
    myGame.draw_board()

    # Check if anyone won
    start = time.time()
    print(myGame.has_anyone_won_efficient(myGame.board))
    end = time.time()
    elapsed = end-start
    print(f"Checked if anyone won in {elapsed} seconds.")

    # Check if it is terminal
    print(myGame.is_terminal(myGame.board))

def test_play(n = 3, m = 5, k = 3, interactive = True, do_pruning = True):

    # Force the move if not interactive mode chosen
    force_move = not interactive

    # Initialize game
    myGame = Game(n=n, m=m, k=k, max_depth=math.inf, do_pruning=do_pruning)

    start = time.time()
    myGame.play(force_move=force_move)
    end = time.time()
    elapsed = end - start
    print(f"The game took {elapsed} seconds to run for m, n, k of {(m, n, k)}, with do_pruning being {do_pruning}.")

    return elapsed

def test_play_connect_4(interactive = True):

    # Force the move if not interactive mode chosen
    force_move = not interactive

    # Initialize depth
    myGame = Game(n=6, m=7, k=4, max_depth=8, do_pruning=True)

    myGame.play(force_move=force_move)

def time_pruning_improvement():

    max_depth = math.inf

    # no pruning
    start = time.time()
    myGame_no_pruning = Game(n=3, m=4, k=3, do_pruning = False, max_depth= max_depth)
    myGame_no_pruning.max(myGame_no_pruning.board, -np.inf, np.inf, depth = 0)
    end = time.time()
    elapsed_no_pruning = end-start

    print(f"Minimax value search without pruning took {elapsed_no_pruning} seconds.")

    # with pruning
    start = time.time()
    myGame_with_pruning = Game(n=3, m=4, k=3, do_pruning = True, max_depth= max_depth)
    myGame_with_pruning.max(myGame_with_pruning.board, -np.inf, np.inf, depth = 0)
    end = time.time()
    elapsed_pruning = end-start

    print(f"Minimax value search with pruning took {elapsed_pruning} seconds.")

    print(f"Pruned search was {elapsed_no_pruning/ elapsed_pruning} times faster.")

def iterate_m_n_k(m_list = [], n_list = [], k_list = [], do_pruning = True):
    running_times = np.zeros( (max(m_list)+1, max(n_list)+1, max(k_list)+1) )
    print(running_times.shape)
    for m in m_list:
        for n in n_list:
            for k in k_list:
                
                running_times[m, n, k] = test_play(m = m, n = n, k = k, interactive=False, do_pruning = do_pruning)

    return running_times

def write_table_to_file(matrix, k):

    folder_name = "tables"

    cwd = os.getcwd()
    dir = os.path.join(cwd,folder_name)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    # Write confusion matrix to file
    np.savetxt(dir + "/table" + f"{k}" + ".csv", matrix, delimiter = " & ", fmt='%.3f')

if __name__ == "__main__":
    # test_draw_board()
    # test_is_valid()
    # test_is_terminal()
    # test_has_anyone_won()
    test_play(n = 4, m = 4, k = 4, interactive = False, do_pruning = False) # Check if it takes more than 3600
    # test_play_connect_4(interactive = True)
    # time_pruning_improvement()

    # Calculate running time with pruning
    # running_times_table = iterate_m_n_k(m_list = [3, 4], n_list = [3, 4], k_list = [3, 4], do_pruning = True)

    # # Extract running times for k = 3 with pruning
    # write_table_to_file(running_times_table[:,:,3], "3_with_pruning")

    # # Extract running times for k = 4 with pruning
    # write_table_to_file(running_times_table[:,:,4], "4_with_pruning")


    # # Calculate running time with no pruning
    # running_times_table = iterate_m_n_k(m_list = [3, 4], n_list = [3, 4], k_list = [3, 4], do_pruning = False)

    # # Extract running times for k = 3 with no pruning
    # write_table_to_file(running_times_table[:,:,3], "3_no_pruning")

    # # Extract running times for k = 4 with no pruning
    # write_table_to_file(running_times_table[:,:,4], "4_no_pruning")


