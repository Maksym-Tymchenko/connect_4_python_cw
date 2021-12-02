    def has_anyone_won(self, board):
        """Intial implementation (not used in code) of function that checks if anyone has won the game at this state.
        Returns None if no one won, 1 if Max won and 10 if Min won."""

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

        # If no k consecutive found return 0
        return None


    def is_k_consecutive(self, sub_k):
        """ Helper function of has_anyone_won (not used in code).
        Returns None if no consecutive, 1 if k consecutive ones, 10 if k consecutive 10s. """
        k = self.k

        is_all_ones = np.all(sub_k == 1)
        is_all_tens = np.all(sub_k == 10)

        if is_all_ones:
            return 1

        if is_all_tens:
            return 10
        
        # Otherwise
        return None
