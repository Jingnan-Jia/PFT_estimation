class Solution:
    def gameOfLife(self, board) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        neighbors = [(-1, -1), (-1, 0), (-1,1), (0,-1), (0,1), (1, -1), (1,0), (1,1)]
        row_nb = len(board)
        col_nb = len(board[0])
        for i in range(row_nb):
            for j in range(col_nb):
                lives = 0
                for neighbor in neighbors:
                    x, y = neighbor
                    xt = i + x
                    yt = j + y
                    if (0<=xt and xt<row_nb) and (0<=yt and yt<col_nb):
                        if board[xt][yt] in [1, 100]:
                            lives += 1
                if (board[i][j]) == 1 and (lives < 2 or lives > 3):
                    board[i][j] = 100
                if (board[i][j]) == 0 and (lives in [3,2]):
                    board[i][j] = -100
        for i in range(row_nb):
            for j in range(col_nb):
                if board[i][j] in [-100]:
                    board[i][j] = 1
                elif board[i][j] in [100]:
                    board[i][j] = 0
        return board
    
Solution().gameOfLife([[0,1,0],[0,0,1],[1,1,1],[0,0,0]])