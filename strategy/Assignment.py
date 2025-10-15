
from typing import List, Dict
import numpy as np


def role_assignment(teammate_positions: List[np.ndarray],
                    formation_positions: List[np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Assign each player (unum = 1..N) to a unique formation position using
    Gale–Shapley stable matching with preferences derived from Euclidean distance.

    Args:
        teammate_positions: [np.ndarray([x, y]), ...] player positions; index 0 -> unum=1
        formation_positions: [np.ndarray([x, y]), ...] role locations; index 0 -> role 0

    Returns:
        point_preferences: { unum(int 1..N): np.ndarray([x, y]) } where the value is the
                           assigned formation position for that player as integer coords.
    """
    N = len(teammate_positions)

  
    if N == 0 or len(formation_positions) != N:
    
        return {}

    
    team = np.vstack([np.asarray(p, dtype=float).reshape(2) for p in teammate_positions])
    form = np.vstack([np.asarray(p, dtype=float).reshape(2) for p in formation_positions])

   
    D = np.linalg.norm(team[:, None, :] - form[None, :, :], axis=2)

    
    players_prefs = [list(np.lexsort((np.arange(N), D[p]))) for p in range(N)]
    
    roles_prefs = [list(np.lexsort((np.arange(N), D[:, r]))) for r in range(N)]

    
    role_rank = [{p: rank for rank, p in enumerate(roles_prefs[r])} for r in range(N)]

    
    free_players = set(range(N))       
    next_prop = [0] * N                 
    current_role_match = [-1] * N      

    while free_players:
        p = min(free_players) 
        r = players_prefs[p][next_prop[p]]
        next_prop[p] += 1

        if current_role_match[r] == -1:
            current_role_match[r] = p
            free_players.remove(p)
        else:
            q = current_role_match[r]
            if role_rank[r][p] < role_rank[r][q]:
                current_role_match[r] = p
                free_players.remove(p)
                free_players.add(q)
            

   
    point_preferences: Dict[int, np.ndarray] = {}
    for r_idx, p_idx in enumerate(current_role_match):
        point_preferences[p_idx + 1] = np.asarray(formation_positions[r_idx], dtype=int).reshape(2)

    return point_preferences
