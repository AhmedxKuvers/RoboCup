import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from formation.Formation import GenerateBasicFormation


def _pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> float:
	"""Squared Euclidean distance between two 2D points."""
	d = a - b
	return float(d[0] * d[0] + d[1] * d[1])


def role_assignment(
	teammate_positions: List[Optional[np.ndarray]],
	formation_positions: List[np.ndarray],
) -> Dict[int, np.ndarray]:
	"""
	Compute an assignment from players (by uniform number, starting at 1) to
	desired formation points.

	Inputs
	- teammate_positions: list indexed by unum-1 containing 2D positions (np.ndarray shape (2,))
						  or None when a teammate isn't localized yet. This list may include
						  more than the number of formation positions; only the first K players
						  are considered, where K == len(formation_positions).
	- formation_positions: list of K formation targets (np.ndarray shape (2,)).

	Output
	- dict mapping unum (1-based) -> assigned formation position (np.ndarray (2,)).

	Notes
	- If a teammate position is None, it's treated as a large cost to avoid assignment when
	  better candidates exist.
	- For players beyond K, they are mapped to their current position if available, otherwise
	  to the first formation point as a safe fallback. This avoids KeyErrors for callers that
	  index by unum blindly.
	"""

	K = len(formation_positions)
	N = len(teammate_positions)

	# Ensure numpy arrays for formation targets
	F = [np.asarray(p, dtype=float) for p in formation_positions]

	# Build list of candidate players (first K players by unum 1..K)
	player_indices = list(range(1, min(K, N) + 1))

	# Pre-compute cost matrix (players x formation)
	# Use a high penalty when teammate position is unknown
	BIG = 1e6
	costs = []  # rows correspond to player_indices order
	for unum in player_indices:
		pos = teammate_positions[unum - 1]
		if pos is None:
			row = [BIG] * K
		else:
			A = np.asarray(pos, dtype=float)
			row = [_pairwise_sq_dist(A, F[j]) for j in range(K)]
		costs.append(row)

	# Solve the linear assignment problem for small K via brute-force permutations.
	# This is fine for K<=11 (11! gets big, but we use K from formation list which is 5 in this repo).
	best_perm = None
	best_cost = float("inf")
	for perm in itertools.permutations(range(K), r=len(player_indices)):
		c = 0.0
		for i, j in enumerate(perm):
			c += costs[i][j]
			if c >= best_cost:
				break  # pruning
		if c < best_cost:
			best_cost = c
			best_perm = perm

	assignment: Dict[int, np.ndarray] = {}

	if best_perm is not None:
		for idx, unum in enumerate(player_indices):
			j = best_perm[idx]
			assignment[unum] = F[j]

	# For players beyond K, keep their current position to avoid KeyErrors downstream
	for unum in range(len(player_indices) + 1, N + 1):
		current = teammate_positions[unum - 1]
		if current is None:
			assignment[unum] = F[0]  # safe fallback
		else:
			assignment[unum] = np.asarray(current, dtype=float)

	return assignment


# ---------------- Additional helpers used by Agent.select_skill ----------------

def get_formation(ball_x: float) -> List[np.ndarray]:
	"""
	Produce a dynamic formation based on the ball's x position.
	Players advance or retreat depending on ball position.
	Maintains good spacing to avoid clustering.
	
	ball_x is expected roughly in [-15, 15].
	"""
	base = GenerateBasicFormation()  # 5 positions
	f = float(np.clip(ball_x / 15.0, -1.0, 1.0))  # -1 (our half) .. 1 (their half)

	out: List[np.ndarray] = []
	for i, p in enumerate(base):
		p = np.asarray(p, dtype=float).copy()
		if i == 0:  # GK: minimal movement
			p[0] = np.clip(p[0] + f * 0.5, -14.5, -11.0)
		elif i <= 2:  # Defenders: moderate movement
			p[0] = np.clip(p[0] + f * 3.0, -13.0, 3.0)
		else:  # Forwards: aggressive movement
			p[0] = np.clip(p[0] + f * 5.0, -5.0, 14.0)
		
		out.append(p)
	
	# Add spacing enforcement to avoid clustering
	min_spacing = 2.5  # Minimum distance between players
	for i in range(len(out)):
		for j in range(i + 1, len(out)):
			dist = np.linalg.norm(out[i] - out[j])
			if dist < min_spacing:
				# Push players apart slightly
				direction = out[j] - out[i]
				if np.linalg.norm(direction) > 0.01:
					direction = direction / np.linalg.norm(direction)
					push = direction * (min_spacing - dist) * 0.5
					out[j] = out[j] + push
					out[i] = out[i] - push
	
	return out


def pass_reciever_selector(
	player_unum: int,
	teammate_positions: List[Optional[Tuple[float, float]]],
	default_target: Tuple[float, float] = (15.0, 0.0),
) -> np.ndarray:
	"""
	Choose between shooting at goal or passing to a teammate:
	- Shoot at goal if in attacking position (x > 5) or no better teammate ahead
	- Pass to a teammate significantly ahead (at least 2m further forward)
	- Otherwise shoot at goal
	Returns a 2D numpy array target.
	"""
	my_idx = player_unum - 1
	if my_idx < 0 or my_idx >= len(teammate_positions):
		return np.asarray(default_target, dtype=float)

	me = teammate_positions[my_idx]
	if me is None:
		return np.asarray(default_target, dtype=float)

	me = np.asarray(me, dtype=float)

	# If I'm in a good shooting position (past halfway, x > 5), prioritize shooting
	if me[0] > 5.0:
		# Only pass if a teammate is SIGNIFICANTLY ahead (2+ meters)
		for i, pos in enumerate(teammate_positions):
			if pos is None or i == my_idx:
				continue
			P = np.asarray(pos, dtype=float)
			if P[0] > me[0] + 2.0:  # Teammate is 2m+ ahead
				return P
		# No one significantly ahead, shoot!
		return np.asarray(default_target, dtype=float)

	# In defensive/midfield: look for better positioned teammates
	ahead: List[np.ndarray] = []
	for i, pos in enumerate(teammate_positions):
		if pos is None or i == my_idx:
			continue
		P = np.asarray(pos, dtype=float)
		if P[0] > me[0] + 1.0:  # At least 1m ahead
			ahead.append(P)

	def _nearest(cands: List[np.ndarray]) -> Optional[np.ndarray]:
		if not cands:
			return None
		d2 = [np.sum((c - me) ** 2) for c in cands]
		return cands[int(np.argmin(d2))]

	target = _nearest(ahead)
	if target is None:
		# No one ahead, shoot at goal
		return np.asarray(default_target, dtype=float)
	return target
