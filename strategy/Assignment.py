import numpy as np
from typing import List, Dict

def role_assignment(teammate_positions: List[np.ndarray],
                    formation_positions: List[np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Optimized implementation of Gale-Shapley stable matching for role assignment.
    Uses vectorized operations for faster computation of distances and preferences.
    Returns a dict mapping unum -> assigned formation position (np.ndarray([x, y])).

    Args:
        teammate_positions: list of np.ndarray([x, y]), index 0 is unum=1, etc.
        formation_positions: list of np.ndarray([x, y]), index 0 is first role.

    Returns:
        point_preferences: dict[int, np.ndarray] mapping unum (1-based) to assigned formation position.
    """
    # 1. Fast validation
    N = len(teammate_positions)
    if N == 0 or len(formation_positions) != N:
        raise ValueError("Both lists must be present and of the same non-zero length.")
    
    # Convert to numpy arrays for vectorized operations
    try:
        teammates = np.array(teammate_positions)
        formation = np.array(formation_positions)
        if teammates.shape != (N, 2) or formation.shape != (N, 2):
            raise ValueError("All positions must be 2D points")
    except:
        raise ValueError("Invalid input arrays")

    # 2. Vectorized distance calculation
    # Reshape for broadcasting: (N,1,2) - (N,2) -> (N,N)
    D = np.linalg.norm(teammates[:, np.newaxis] - formation, axis=2)

    # 3. Build preference lists with deterministic tie-breaks
    # Players' preferences: for each player, sorted list of role indices
    players_prefs = []
    for p in range(N):
        dists = D[p]
        idxs = np.lexsort((np.arange(N), dists))
        players_prefs.append(list(idxs))

    # Roles' preferences: for each role, sorted list of player indices
    roles_prefs = []
    for r in range(N):
        dists = D[:, r]
        idxs = np.lexsort((np.arange(N), dists))
        roles_prefs.append(list(idxs))

    # For O(1) comparison: role_rank[role][player] = rank of player in role's preference
    role_rank = [dict() for _ in range(N)]
    for r in range(N):
        for rank, p in enumerate(roles_prefs[r]):
            role_rank[r][p] = rank

    # 4. Gale–Shapley (players propose)
    free_players = set(range(N))
    next_proposal = [0] * N  # next role index to propose to for each player
    current_match_role_to_player = [None] * N  # role_index -> player_index or None

    matches_made = 0
    while free_players and matches_made < N:
        p = min(free_players)  # deterministic: lowest index first
        if next_proposal[p] >= N:
            raise RuntimeError(f"Player {p} has run out of roles to propose to.")
        
        r = players_prefs[p, next_proposal[p]]
        next_proposal[p] += 1
        
        if current_match_role_to_player[r] == -1:
            # Role is free
            current_match_role_to_player[r] = p
            free_players.remove(p)
            matches_made += 1
        else:
            q = current_match_role_to_player[r]
            # Role prefers lower rank
            if role_rank[r, p] < role_rank[r, q]:
                current_match_role_to_player[r] = p
                free_players.remove(p)
                free_players.add(q)
                # matches_made stays the same as we just swapped
    
    # 5. Build result efficiently
    point_preferences = {
        p + 1: formation[r].copy()  # unum is 1-based
        for r, p in enumerate(current_match_role_to_player)
    }
    
    return point_preferences

if __name__ == "__main__":
    def _quick_test():
        teammates = [np.array([1,1]), np.array([0,0]), np.array([3,3])]
        roles     = [np.array([2,1]), np.array([0,3]), np.array([4,5])]
        out = role_assignment(teammate_positions=teammates, formation_positions=roles)
        # Validate shape & keys
        assert set(out.keys()) == {1,2,3}
        for v in out.values():
            assert isinstance(v, np.ndarray) and v.shape == (2,)
        # Determinism check (tiny perturbation shouldn't flip rank ties)
        out2 = role_assignment(teammate_positions=[t.copy() for t in teammates],
                               formation_positions=[r.copy() for r in roles])
        assert all((out[k] == out2[k]).all() for k in out)
        print("Basic test OK:", out)

    def test_edge_cases():
        # Single player case
        single_result = role_assignment([np.array([0,0])], [np.array([1,1])])
        assert len(single_result) == 1
        assert 1 in single_result
        assert np.array_equal(single_result[1], np.array([1,1]))
        print("Single player case OK")
        
        # Equal distances case
        teammates = [np.array([0,0]), np.array([2,0])]
        formation = [np.array([1,1]), np.array([1,-1])]
        result = role_assignment(teammates, formation)
        assert len(result) == 2
        assert set(result.keys()) == {1, 2}
        print("Equal distances case OK")
        
        # Multiple players with same position
        teammates = [np.array([0,0]), np.array([0,0]), np.array([2,2])]
        formation = [np.array([1,0]), np.array([0,1]), np.array([2,1])]
        result = role_assignment(teammates, formation)
        assert len(result) == 3
        assert set(result.keys()) == {1, 2, 3}
        print("Same position case OK")

    # Run all tests
    print("\nRunning tests...")
    _quick_test()
    test_edge_cases()