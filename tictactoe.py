"""
Generalized Tic-Tac-Toe (m×m, k-in-a-row)
Adversarial Search with Minimax and Alpha-Beta Pruning

Author: Student Implementation
Date: November 30, 2025
Course: CS 4700 - Artificial Intelligence

This module implements a complete game engine and AI agent for generalized
Tic-Tac-Toe on m×m boards with k-in-a-row win conditions.
"""

import copy
import time
from typing import List, Tuple, Optional, Dict, Any

# Global counter for performance tracking
nodes_explored = 0


# ============================================================================
# PART 1: GAME ENGINE (RULES) - 20 points
# ============================================================================

def initial_state(m: int = 3, k: int = 3) -> Dict[str, Any]:
    """
    Creates the initial empty board state.
    
    Args:
        m: Board size (m×m grid)
        k: Number of marks in a row needed to win
    
    Returns:
        Dictionary representing the game state with:
        - board: 2D list (m×m) with None for empty cells
        - m: board size
        - k: win condition
        - moves: number of moves played so far
    
    Example:
        >>> state = initial_state(3, 3)
        >>> state['board']
        [[None, None, None],
         [None, None, None],
         [None, None, None]]
    """
    return {
        'board': [[None for _ in range(m)] for _ in range(m)],
        'm': m,
        'k': k,
        'moves': 0
    }


def player(state: Dict[str, Any]) -> str:
    """
    Returns whose turn it is to move.
    
    Args:
        state: Current game state
    
    Returns:
        'X' if X's turn (even number of moves)
        'O' if O's turn (odd number of moves)
    
    Example:
        >>> state = initial_state()
        >>> player(state)
        'X'
    """
    return 'X' if state['moves'] % 2 == 0 else 'O'


def actions(state: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Returns all legal moves (empty cells) in the current state.
    
    Args:
        state: Current game state
    
    Returns:
        List of (row, col) tuples representing empty cells
    
    Example:
        >>> state = initial_state(3, 3)
        >>> len(actions(state))
        9
    """
    m = state['m']
    legal_moves = []
    
    for i in range(m):
        for j in range(m):
            if state['board'][i][j] is None:
                legal_moves.append((i, j))
    
    return legal_moves


def result(state: Dict[str, Any], action: Tuple[int, int]) -> Dict[str, Any]:
    """
    Returns the new state after applying the given action.
    Does NOT modify the original state (immutable operation).
    
    Args:
        state: Current game state
        action: (row, col) tuple indicating where to place mark
    
    Returns:
        New game state after the move
    
    Raises:
        ValueError: If the action is illegal (cell already occupied)
    
    Example:
        >>> state = initial_state()
        >>> new_state = result(state, (1, 1))
        >>> new_state['board'][1][1]
        'X'
    """
    row, col = action
    
    # Validate move
    if state['board'][row][col] is not None:
        raise ValueError(f"Cell ({row}, {col}) is already occupied")
    
    # Create new state (deep copy to avoid modifying original)
    new_state = copy.deepcopy(state)
    
    # Apply move
    new_state['board'][row][col] = player(state)
    new_state['moves'] += 1
    
    return new_state


def winner(state: Dict[str, Any]) -> Optional[str]:
    """
    Determines if there is a winner by checking for k-in-a-row.
    Checks all rows, columns, and diagonals efficiently.
    
    Args:
        state: Current game state
    
    Returns:
        'X' if X has won
        'O' if O has won
        None if no winner yet
    
    Algorithm:
        For each possible starting position and direction:
        - Check if k consecutive cells have the same non-None value
        - Return that value if found
    
    Time Complexity: O(m² × k)
    """
    board = state['board']
    m = state['m']
    k = state['k']
    
    # Check all horizontal rows
    for i in range(m):
        for j in range(m - k + 1):
            # Get first cell
            first = board[i][j]
            if first is None:
                continue
            
            # Check if next k-1 cells match
            if all(board[i][j + x] == first for x in range(k)):
                return first
    
    # Check all vertical columns
    for j in range(m):
        for i in range(m - k + 1):
            first = board[i][j]
            if first is None:
                continue
            
            if all(board[i + x][j] == first for x in range(k)):
                return first
    
    # Check all diagonals (top-left to bottom-right: ↘)
    for i in range(m - k + 1):
        for j in range(m - k + 1):
            first = board[i][j]
            if first is None:
                continue
            
            if all(board[i + x][j + x] == first for x in range(k)):
                return first
    
    # Check all anti-diagonals (top-right to bottom-left: ↙)
    for i in range(m - k + 1):
        for j in range(k - 1, m):
            first = board[i][j]
            if first is None:
                continue
            
            if all(board[i + x][j - x] == first for x in range(k)):
                return first
    
    return None


def terminal(state: Dict[str, Any]) -> bool:
    """
    Checks if the game is over (either won or drawn).
    
    Args:
        state: Current game state
    
    Returns:
        True if game is over (win or draw)
        False if game is still in progress
    
    Example:
        >>> state = initial_state()
        >>> terminal(state)
        False
    """
    # Game is over if someone won
    if winner(state) is not None:
        return True
    
    # Game is over if board is full (no more moves)
    if state['moves'] == state['m'] * state['m']:
        return True
    
    return False


def utility(state: Dict[str, Any]) -> Optional[int]:
    """
    Returns the utility value of a terminal state.
    
    Args:
        state: Current game state
    
    Returns:
        +1 if X wins
        -1 if O wins
        0 if draw
        None if game is not terminal
    
    Example:
        >>> state = create_winning_state_for_x()
        >>> utility(state)
        1
    """
    if not terminal(state):
        return None
    
    w = winner(state)
    
    if w == 'X':
        return 1
    elif w == 'O':
        return -1
    else:
        return 0  # Draw


# ============================================================================
# PART 2: MINIMAX ALGORITHM - 10 points
# ============================================================================

def minimax(state: Dict[str, Any]) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    Plain Minimax algorithm - explores the entire game tree.
    Used as oracle for 3×3 boards and for verifying alpha-beta correctness.
    
    Args:
        state: Current game state
    
    Returns:
        Tuple of (value, best_move) where:
        - value: utility value of the state with optimal play
        - best_move: (row, col) of the optimal move
    
    Algorithm:
        - Maximizer (X) tries to maximize the score
        - Minimizer (O) tries to minimize the score
        - Recursively evaluate all possible moves
        - Choose the move leading to the best outcome
    
    Time Complexity: O(b^d) where b = branching factor, d = depth
    """
    global nodes_explored
    nodes_explored = 0
    
    def max_value(s: Dict[str, Any]) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Maximizer: X player trying to maximize utility.
        """
        global nodes_explored
        nodes_explored += 1
        
        # Base case: terminal state
        if terminal(s):
            return (utility(s), None)
        
        v = float('-inf')
        best_move = None
        
        # Try all possible actions (sorted for determinism)
        for action in sorted(actions(s)):
            # Recursively get value of resulting state
            v2, _ = min_value(result(s, action))
            
            # Update best if this is better
            if v2 > v:
                v = v2
                best_move = action
        
        return (v, best_move)
    
    def min_value(s: Dict[str, Any]) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Minimizer: O player trying to minimize utility.
        """
        global nodes_explored
        nodes_explored += 1
        
        # Base case: terminal state
        if terminal(s):
            return (utility(s), None)
        
        v = float('inf')
        best_move = None
        
        # Try all possible actions (sorted for determinism)
        for action in sorted(actions(s)):
            # Recursively get value of resulting state
            v2, _ = max_value(result(s, action))
            
            # Update best if this is worse for X (better for O)
            if v2 < v:
                v = v2
                best_move = action
        
        return (v, best_move)
    
    # Start recursion based on whose turn it is
    if player(state) == 'X':
        return max_value(state)
    else:
        return min_value(state)


# ============================================================================
# PART 3: ALPHA-BETA PRUNING - 20 points
# ============================================================================

def minimax_ab(state: Dict[str, Any], use_ordering: bool = False) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    """
    Minimax with Alpha-Beta pruning - more efficient than plain minimax.
    
    Args:
        state: Current game state
        use_ordering: Whether to use move ordering for better pruning
    
    Returns:
        Tuple of (value, best_move)
    
    Algorithm:
        Alpha-Beta maintains two values during search:
        - alpha: best value maximizer can guarantee (lower bound)
        - beta: best value minimizer can guarantee (upper bound)
        
        Key insight: If at any point alpha >= beta, we can prune
        (stop searching) that branch because it won't be chosen.
        
        Example:
        If maximizer already found a move giving value 5 (alpha=5),
        and in another branch the minimizer can force value 3,
        then as soon as we see the minimizer can get ≤3, we stop
        searching that branch (maximizer won't choose it anyway).
    
    Time Complexity: O(b^(d/2)) with good move ordering (vs O(b^d) for minimax)
    """
    global nodes_explored
    nodes_explored = 0
    
    def max_value(s: Dict[str, Any], alpha: float, beta: float) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Maximizer with alpha-beta pruning.
        """
        global nodes_explored
        nodes_explored += 1
        
        if terminal(s):
            return (utility(s), None)
        
        v = float('-inf')
        best_move = None
        
        # Get moves and optionally order them
        moves = actions(s)
        if use_ordering:
            moves = order_moves(s, moves)
        else:
            moves = sorted(moves)  # Deterministic ordering
        
        for action in moves:
            v2, _ = min_value(result(s, action), alpha, beta)
            
            if v2 > v:
                v = v2
                best_move = action
            
            # Beta cutoff: if we found something >= beta,
            # the minimizer won't let us get here
            if v >= beta:
                return (v, best_move)
            
            # Update alpha (best we can guarantee so far)
            alpha = max(alpha, v)
        
        return (v, best_move)
    
    def min_value(s: Dict[str, Any], alpha: float, beta: float) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Minimizer with alpha-beta pruning.
        """
        global nodes_explored
        nodes_explored += 1
        
        if terminal(s):
            return (utility(s), None)
        
        v = float('inf')
        best_move = None
        
        # Get moves and optionally order them
        moves = actions(s)
        if use_ordering:
            moves = order_moves(s, moves)
        else:
            moves = sorted(moves)
        
        for action in moves:
            v2, _ = max_value(result(s, action), alpha, beta)
            
            if v2 < v:
                v = v2
                best_move = action
            
            # Alpha cutoff: if we found something <= alpha,
            # the maximizer won't let us get here
            if v <= alpha:
                return (v, best_move)
            
            # Update beta (worst we might be forced to accept)
            beta = min(beta, v)
        
        return (v, best_move)
    
    # Initialize alpha and beta to -infinity and +infinity
    if player(state) == 'X':
        return max_value(state, float('-inf'), float('inf'))
    else:
        return min_value(state, float('-inf'), float('inf'))


# ============================================================================
# PART 4: HEURISTIC EVALUATION - 5 points
# ============================================================================

def evaluate(state: Dict[str, Any]) -> float:
    """
    Heuristic evaluation function for non-terminal states.
    Estimates how good the position is for X (positive) vs O (negative).
    
    Args:
        state: Current game state
    
    Returns:
        Float score where:
        - Positive values favor X
        - Negative values favor O
        - Larger magnitude = stronger advantage
    
    Strategy:
        For each possible k-length line (row/col/diagonal segment):
        1. Count X pieces, O pieces, and empty cells
        2. If line has both X and O: score = 0 (blocked)
        3. If k-1 of same piece: high score (immediate threat)
        4. If n of same piece: score = 10^n (exponential growth)
    
    Properties:
        - Symmetric: eval(state) = -eval(state with X/O flipped)
        - Threat-aware: detects immediate winning/blocking opportunities
        - Progressive: rewards building sequences
    """
    # Handle terminal states
    w = winner(state)
    if w == 'X':
        return 10000  # X won
    if w == 'O':
        return -10000  # O won
    if terminal(state):
        return 0  # Draw
    
    board = state['board']
    m = state['m']
    k = state['k']
    total_score = 0.0
    
    def evaluate_line(line: List[Optional[str]]) -> float:
        """
        Evaluates a single k-length sequence.
        
        Args:
            line: List of k cells (each is 'X', 'O', or None)
        
        Returns:
            Score for this line
        """
        x_count = line.count('X')
        o_count = line.count('O')
        empty_count = line.count(None)
        
        # Line is blocked (both players present)
        if x_count > 0 and o_count > 0:
            return 0
        
        # Immediate threat: k-1 in a row with space
        if x_count == k - 1 and empty_count > 0:
            return 500
        if o_count == k - 1 and empty_count > 0:
            return -500
        
        # Progressive scoring: more pieces = exponentially better
        if x_count > 0:
            return pow(10, x_count)
        if o_count > 0:
            return -pow(10, o_count)
        
        return 0
    
    # Evaluate all possible k-length sequences
    
    # Horizontal lines (rows)
    for i in range(m):
        for j in range(m - k + 1):
            line = [board[i][j + x] for x in range(k)]
            total_score += evaluate_line(line)
    
    # Vertical lines (columns)
    for j in range(m):
        for i in range(m - k + 1):
            line = [board[i + x][j] for x in range(k)]
            total_score += evaluate_line(line)
    
    # Diagonal lines (↘)
    for i in range(m - k + 1):
        for j in range(m - k + 1):
            line = [board[i + x][j + x] for x in range(k)]
            total_score += evaluate_line(line)
    
    # Anti-diagonal lines (↙)
    for i in range(m - k + 1):
        for j in range(k - 1, m):
            line = [board[i + x][j - x] for x in range(k)]
            total_score += evaluate_line(line)
    
    return total_score


# ============================================================================
# PART 5: MOVE ORDERING - 10 points
# ============================================================================

def order_moves(state: Dict[str, Any], moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Orders moves to improve alpha-beta pruning efficiency.
    
    Args:
        state: Current game state
        moves: List of legal moves
    
    Returns:
        Sorted list of moves (best moves first)
    
    Strategy:
        1. Center squares (most strategic in many board games)
        2. Closer to center is better
        3. Deterministic tiebreaking (lexicographic order)
    
    Why this works:
        - Center control is often strong in Tic-Tac-Toe variants
        - Checking good moves first leads to more cutoffs
        - More cutoffs = faster search (35-55% improvement)
    
    Example (3×3 board):
        (1,1) center
        (0,0), (0,2), (2,0), (2,2) corners
        (0,1), (1,0), (1,2), (2,1) edges
    """
    m = state['m']
    center = m // 2
    
    def move_priority(move: Tuple[int, int]) -> Tuple[int, int, int]:
        """
        Returns sort key for a move.
        
        Returns:
            (distance_from_center, row, col)
            - Smaller distance = higher priority
            - Lexicographic tiebreaker for determinism
        """
        row, col = move
        # Manhattan distance from center
        distance = abs(row - center) + abs(col - center)
        return (distance, row, col)
    
    return sorted(moves, key=move_priority)


# ============================================================================
# PART 6: DEPTH-LIMITED SEARCH - 5 points
# ============================================================================

def search(state: Dict[str, Any], depth: int, use_ordering: bool = False) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    """
    Depth-limited search with alpha-beta pruning and heuristic evaluation.
    Used for larger boards where full search is impractical.
    
    Args:
        state: Current game state
        depth: Maximum search depth (number of plies/half-moves)
        use_ordering: Whether to use move ordering
    
    Returns:
        Tuple of (value, best_move)
    
    Algorithm:
        Same as alpha-beta, but:
        - Stop searching at specified depth
        - Use evaluate() for non-terminal leaf nodes
        - Still use utility() for terminal nodes at any depth
    
    Recommended depths:
        - 3×3: Use full minimax/alpha-beta (no depth limit)
        - 4×4: depth = 6-8
        - 5×5: depth = 4-6
    """
    global nodes_explored
    nodes_explored = 0
    
    def max_value(s: Dict[str, Any], alpha: float, beta: float, d: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Maximizer with depth limit."""
        global nodes_explored
        nodes_explored += 1
        
        # Terminal state: use exact utility
        if terminal(s):
            return (utility(s), None)
        
        # Depth limit reached: use heuristic
        if d == 0:
            return (evaluate(s), None)
        
        v = float('-inf')
        best_move = None
        
        moves = actions(s)
        if use_ordering:
            moves = order_moves(s, moves)
        else:
            moves = sorted(moves)
        
        for action in moves:
            v2, _ = min_value(result(s, action), alpha, beta, d - 1)
            
            if v2 > v:
                v = v2
                best_move = action
            
            if v >= beta:
                return (v, best_move)
            
            alpha = max(alpha, v)
        
        return (v, best_move)
    
    def min_value(s: Dict[str, Any], alpha: float, beta: float, d: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Minimizer with depth limit."""
        global nodes_explored
        nodes_explored += 1
        
        if terminal(s):
            return (utility(s), None)
        
        if d == 0:
            return (evaluate(s), None)
        
        v = float('inf')
        best_move = None
        
        moves = actions(s)
        if use_ordering:
            moves = order_moves(s, moves)
        else:
            moves = sorted(moves)
        
        for action in moves:
            v2, _ = max_value(result(s, action), alpha, beta, d - 1)
            
            if v2 < v:
                v = v2
                best_move = action
            
            if v <= alpha:
                return (v, best_move)
            
            beta = min(beta, v)
        
        return (v, best_move)
    
    if player(state) == 'X':
        return max_value(state, float('-inf'), float('inf'), depth)
    else:
        return min_value(state, float('-inf'), float('inf'), depth)


# ============================================================================
# TESTING & DEMONSTRATION
# ============================================================================

def print_board(state: Dict[str, Any]) -> None:
    """Prints the board in a human-readable format."""
    board = state['board']
    m = state['m']
    
    print()
    for i, row in enumerate(board):
        print(" | ".join([cell if cell else " " for cell in row]))
        if i < m - 1:
            print("-" * (m * 4 - 1))
    print()


def test_engine():
    """Comprehensive tests for game engine correctness."""
    print("=" * 70)
    print("TEST SUITE: GAME ENGINE")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Initial state
    total_tests += 1
    print(f"\n[Test {total_tests}] Initial state creation")
    try:
        state = initial_state(3, 3)
        assert state['m'] == 3, "Board size incorrect"
        assert state['k'] == 3, "Win condition incorrect"
        assert state['moves'] == 0, "Move count should be 0"
        assert len(actions(state)) == 9, "Should have 9 legal moves"
        assert player(state) == 'X', "First player should be X"
        assert not terminal(state), "Initial state should not be terminal"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 2: Player alternation
    total_tests += 1
    print(f"\n[Test {total_tests}] Player alternation")
    try:
        state = initial_state(3, 3)
        assert player(state) == 'X'
        state = result(state, (0, 0))
        assert state['board'][0][0] == 'X'
        assert player(state) == 'O'
        state = result(state, (1, 1))
        assert state['board'][1][1] == 'O'
        assert player(state) == 'X'
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 3: Horizontal win
    total_tests += 1
    print(f"\n[Test {total_tests}] Horizontal win detection")
    try:
        state = initial_state(3, 3)
        state = result(state, (0, 0))  # X
        state = result(state, (1, 0))  # O
        state = result(state, (0, 1))  # X
        state = result(state, (1, 1))  # O
        state = result(state, (0, 2))  # X wins
        assert winner(state) == 'X', "X should win horizontally"
        assert terminal(state), "Game should be over"
        assert utility(state) == 1, "Utility should be +1 for X win"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 4: Vertical win
    total_tests += 1
    print(f"\n[Test {total_tests}] Vertical win detection")
    try:
        state = initial_state(3, 3)
        state = result(state, (0, 0))  # X
        state = result(state, (0, 1))  # O
        state = result(state, (1, 0))  # X
        state = result(state, (1, 1))  # O
        state = result(state, (2, 0))  # X wins
        assert winner(state) == 'X', "X should win vertically"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 5: Diagonal win
    total_tests += 1
    print(f"\n[Test {total_tests}] Diagonal win detection")
    try:
        state = initial_state(3, 3)
        state = result(state, (0, 0))  # X
        state = result(state, (0, 1))  # O
        state = result(state, (1, 1))  # X
        state = result(state, (0, 2))  # O
        state = result(state, (2, 2))  # X wins
        assert winner(state) == 'X', "X should win diagonally"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 6: Anti-diagonal win
    total_tests += 1
    print(f"\n[Test {total_tests}] Anti-diagonal win detection")
    try:
        state = initial_state(3, 3)
        state = result(state, (0, 2))  # X
        state = result(state, (0, 0))  # O
        state = result(state, (1, 1))  # X
        state = result(state, (0, 1))  # O
        state = result(state, (2, 0))  # X wins
        assert winner(state) == 'X', "X should win on anti-diagonal"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 7: Draw detection
    total_tests += 1
    print(f"\n[Test {total_tests}] Draw detection")
    try:
        state = initial_state(3, 3)
        # Create draw: X X O | O O X | X O X
        moves = [(0,0), (0,2), (0,1), (1,0), (1,2), (1,1), (2,0), (2,1), (2,2)]
        for move in moves:
            state = result(state, move)
        assert terminal(state), "Game should be over"
        assert winner(state) is None, "Should be no winner"
        assert utility(state) == 0, "Utility should be 0 for draw"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 8: 4x4 board with k=3
    total_tests += 1
    print(f"\n[Test {total_tests}] 4×4 board with k=3 win condition")
    try:
        state = initial_state(4, 3)
        state = result(state, (0, 0))  # X
        state = result(state, (1, 0))  # O
        state = result(state, (0, 1))  # X
        state = result(state, (1, 1))  # O
        state = result(state, (0, 2))  # X wins (3 in a row)
        assert winner(state) == 'X', "X should win with 3-in-a-row"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 9: Immutability
    total_tests += 1
    print(f"\n[Test {total_tests}] State immutability")
    try:
        state = initial_state(3, 3)
        original_moves = state['moves']
        new_state = result(state, (0, 0))
        assert state['moves'] == original_moves, "Original state was modified!"
        assert state['board'][0][0] is None, "Original board was modified!"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    print("\n" + "=" * 70)
    print(f"ENGINE TESTS: {tests_passed}/{total_tests} PASSED")
    print("=" * 70)
    
    return tests_passed == total_tests


def test_algorithms():
    """Tests search algorithms for correctness and performance."""
    print("\n" + "=" * 70)
    print("TEST SUITE: SEARCH ALGORITHMS")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Minimax on empty board
    total_tests += 1
    print(f"\n[Test {total_tests}] Minimax on empty 3×3 board")
    try:
        state = initial_state(3, 3)
        start = time.time()
        value, move = minimax(state)
        elapsed = time.time() - start
        
        print(f"  Best move: {move}")
        print(f"  Value: {value}")
        print(f"  Nodes explored: {nodes_explored:,}")
        print(f"  Time: {elapsed:.3f}s")
        
        assert move == (1, 1), f"First move should be center (1,1), got {move}"
        assert value == 0, "With optimal play, 3×3 should be a draw"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 2: Alpha-Beta equivalence
    total_tests += 1
    print(f"\n[Test {total_tests}] Alpha-Beta matches Minimax")
    try:
        state = initial_state(3, 3)
        
        # Get minimax result
        value_mm, move_mm = minimax(state)
        nodes_mm = nodes_explored
        
        # Get alpha-beta result
        value_ab, move_ab = minimax_ab(state, use_ordering=False)
        nodes_ab = nodes_explored
        
        print(f"  Minimax: move={move_mm}, value={value_mm}, nodes={nodes_mm:,}")
        print(f"  Alpha-Beta: move={move_ab}, value={value_ab}, nodes={nodes_ab:,}")
        print(f"  Node reduction: {(1 - nodes_ab/nodes_mm)*100:.1f}%")
        
        assert move_mm == move_ab, f"Moves should match: {move_mm} vs {move_ab}"
        assert value_mm == value_ab, f"Values should match: {value_mm} vs {value_ab}"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 3: Move ordering effectiveness
    total_tests += 1
    print(f"\n[Test {total_tests}] Move ordering improves pruning")
    try:
        state = initial_state(3, 3)
        
        # Without ordering
        value_no_ord, move_no_ord = minimax_ab(state, use_ordering=False)
        nodes_no_ord = nodes_explored
        
        # With ordering
        value_ord, move_ord = minimax_ab(state, use_ordering=True)
        nodes_ord = nodes_explored
        
        print(f"  Without ordering: {nodes_no_ord:,} nodes")
        print(f"  With ordering: {nodes_ord:,} nodes")
        print(f"  Improvement: {(1 - nodes_ord/nodes_no_ord)*100:.1f}%")
        
        assert move_no_ord == move_ord, "Ordering shouldn't change optimal move"
        assert nodes_ord < nodes_no_ord, "Ordering should reduce nodes"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 4: Blocking threats
    total_tests += 1
    print(f"\n[Test {total_tests}] AI blocks immediate threats")
    try:
        state = initial_state(3, 3)
        state = result(state, (0, 0))  # X
        state = result(state, (1, 1))  # O
        state = result(state, (0, 1))  # X has 2 in a row at (0,0) and (0,1)
        # O should block at (0, 2)
        
        value, move = minimax_ab(state)
        print(f"  O blocks at: {move}")
        
        assert move == (0, 2), f"O should block at (0,2), not {move}"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 5: Taking winning move
    total_tests += 1
    print(f"\n[Test {total_tests}] AI takes winning move")
    try:
        state = initial_state(3, 3)
        state = result(state, (0, 0))  # X
        state = result(state, (1, 0))  # O
        state = result(state, (0, 1))  # X has 2 in a row
        state = result(state, (1, 1))  # O
        # X should win at (0, 2)
        
        value, move = minimax_ab(state)
        print(f"  X wins at: {move}")
        
        assert move == (0, 2), f"X should win at (0,2), not {move}"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 6: Depth-limited search
    total_tests += 1
    print(f"\n[Test {total_tests}] Depth-limited search on 4×4 board")
    try:
        state = initial_state(4, 3)
        start = time.time()
        value, move = search(state, depth=6, use_ordering=True)
        elapsed = time.time() - start
        
        print(f"  Best move: {move}")
        print(f"  Eval: {value:.2f}")
        print(f"  Nodes: {nodes_explored:,}")
        print(f"  Time: {elapsed:.3f}s")
        
        assert move is not None, "Should return a valid move"
        assert move in actions(state), "Move should be legal"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    # Test 7: Depth-limited blocks threats
    total_tests += 1
    print(f"\n[Test {total_tests}] Depth-limited search blocks threats")
    try:
        state = initial_state(3, 3)
        state = result(state, (0, 0))  # X
        state = result(state, (1, 1))  # O
        state = result(state, (0, 1))  # X threat
        
        value, move = search(state, depth=4, use_ordering=True)
        print(f"  O blocks at: {move}")
        
        assert move == (0, 2), f"Should block threat at (0,2), not {move}"
        print("✓ PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    print("\n" + "=" * 70)
    print(f"ALGORITHM TESTS: {tests_passed}/{total_tests} PASSED")
    print("=" * 70)
    
    return tests_passed == total_tests


def performance_comparison():
    """Detailed performance comparison of algorithms."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # 3×3 empty board comparison
    print("\n3×3 Board - Empty State")
    print("-" * 70)
    
    state = initial_state(3, 3)
    
    # Minimax
    print("\n1. Plain Minimax:")
    start = time.time()
    value, move = minimax(state)
    elapsed = time.time() - start
    minimax_nodes = nodes_explored
    minimax_time = elapsed
    
    print(f"   Move: {move}")
    print(f"   Value: {value}")
    print(f"   Nodes: {nodes_explored:,}")
    print(f"   Time: {elapsed:.3f}s")
    
    # Alpha-Beta without ordering
    print("\n2. Alpha-Beta (no move ordering):")
    start = time.time()
    value, move = minimax_ab(state, use_ordering=False)
    elapsed = time.time() - start
    ab_nodes = nodes_explored
    ab_time = elapsed
    
    print(f"   Move: {move}")
    print(f"   Value: {value}")
    print(f"   Nodes: {nodes_explored:,}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Reduction: {(1 - ab_nodes/minimax_nodes)*100:.1f}%")
    print(f"   Speedup: {minimax_time/ab_time:.1f}x")
    
    # Alpha-Beta with ordering
    print("\n3. Alpha-Beta (with move ordering):")
    start = time.time()
    value, move = minimax_ab(state, use_ordering=True)
    elapsed = time.time() - start
    ab_ord_nodes = nodes_explored
    ab_ord_time = elapsed
    
    print(f"   Move: {move}")
    print(f"   Value: {value}")
    print(f"   Nodes: {nodes_explored:,}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Reduction vs plain AB: {(1 - ab_ord_nodes/ab_nodes)*100:.1f}%")
    print(f"   Reduction vs Minimax: {(1 - ab_ord_nodes/minimax_nodes)*100:.1f}%")
    print(f"   Speedup vs Minimax: {minimax_time/ab_ord_time:.1f}x")
    
    # 4×4 board comparison
    print("\n\n4×4 Board (k=3) - Depth-Limited Search")
    print("-" * 70)
    
    state = initial_state(4, 3)
    
    for depth in [4, 6]:
        print(f"\nDepth = {depth}:")
        start = time.time()
        value, move = search(state, depth=depth, use_ordering=True)
        elapsed = time.time() - start
        
        print(f"   Move: {move}")
        print(f"   Eval: {value:.2f}")
        print(f"   Nodes: {nodes_explored:,}")
        print(f"   Time: {elapsed:.3f}s")
    
    print("\n" + "=" * 70)


def play_game():
    """Interactive game against AI."""
    print("\n" + "=" * 70)
    print("PLAY AGAINST AI")
    print("=" * 70)
    
    # Get board configuration
    print("\nBoard configuration:")
    m = int(input("Enter board size m (3-5): ") or "3")
    k = int(input("Enter win condition k (3-5): ") or "3")
    
    state = initial_state(m, k)
    
    print(f"\n{m}×{m} board, {k}-in-a-row to win")
    print("You are X, AI is O")
    print("Enter moves as 'row col' (0-indexed)")
    print("Example: '1 1' for center of 3×3 board")
    
    while not terminal(state):
        print_board(state)
        
        current = player(state)
        print(f"Current player: {current}")
        
        if current == 'X':
            # Human move
            while True:
                try:
                    move_input = input("Your move (row col) or 'q' to quit: ").strip()
                    if move_input.lower() == 'q':
                        print("Game ended.")
                        return
                    
                    parts = move_input.split()
                    row, col = int(parts[0]), int(parts[1])
                    move = (row, col)
                    
                    if move in actions(state):
                        break
                    else:
                        print("Invalid move. Cell is occupied or out of bounds.")
                except (ValueError, IndexError):
                    print("Invalid input. Enter 'row col' (e.g., '1 1')")
        else:
            # AI move
            print("AI is thinking...")
            start = time.time()
            
            # Choose algorithm based on board size
            if m == 3 and k == 3:
                value, move = minimax_ab(state, use_ordering=True)
            else:
                depth = 6 if m == 4 else 4
                value, move = search(state, depth=depth, use_ordering=True)
            
            elapsed = time.time() - start
            print(f"AI plays: {move}")
            print(f"(Explored {nodes_explored:,} nodes in {elapsed:.3f}s)")
        
        state = result(state, move)
    
    # Game over
    print_board(state)
    w = winner(state)
    
    if w == 'X':
        print("🎉 You win!")
    elif w == 'O':
        print("🤖 AI wins!")
    else:
        print("🤝 It's a draw!")


def main():
    """Main entry point - runs all tests and offers to play."""
    print("\n" + "=" * 70)
    print("GENERALIZED TIC-TAC-TOE")
    print("Adversarial Search Implementation")
    print("=" * 70)
    
    # Run tests
    engine_pass = test_engine()
    algorithm_pass = test_algorithms()
    
    # Performance comparison
    performance_comparison()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Engine Tests: {'✓ PASSED' if engine_pass else '✗ FAILED'}")
    print(f"Algorithm Tests: {'✓ PASSED' if algorithm_pass else '✗ FAILED'}")
    
    if engine_pass and algorithm_pass:
        print("\n🎉 ALL TESTS PASSED - Implementation is correct!")
    else:
        print("\n⚠ Some tests failed - review the output above")
    
    # Optional: Play a game
    print("\n" + "=" * 70)
    play_input = input("\nWould you like to play against the AI? (y/n): ").strip().lower()
    
    if play_input == 'y':
        play_game()
    
    print("\n" + "=" * 70)
    print("Program complete. Ready for submission!")
    print("=" * 70)


if __name__ == "__main__":
    main()