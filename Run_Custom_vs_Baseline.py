#!/usr/bin/env python3
"""
Run Custom Agent vs Baseline Agent
This script runs a match between your custom agent team and the baseline/vanilla agent team.
- Team 1 (Left side): Custom Agent (from agent.Agent)
- Team 2 (Right side): Baseline Agent (from agent.baseline.Agent_Baseline)
"""

from scripts.commons.Script import Script
script = Script() # Initialize: load config file, parse arguments, build cpp modules
a = script.args

# Import both agent types
from agent.Agent import Agent as CustomAgent
from agent.baseline.Agent_Baseline import Agent as BaselineAgent

# Team 1: Custom Agent (your implementation)
# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw
team1_args = ((a.i, a.p, a.m, u, a.t, True, True) for u in range(1, 6))
script.batch_create(CustomAgent, team1_args)

# Team 2: Baseline Agent (vanilla opponent)
# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw
team2_args = ((a.i, a.p, a.m, u, "Baseline", True, False) for u in range(1, 6))
script.batch_create(BaselineAgent, team2_args)

print("\n" + "="*60)
print("Match Setup:")
print(f"  Team 1 (Left):  {a.t} - Custom Agent")
print(f"  Team 2 (Right): Baseline - Vanilla Agent")
print("="*60 + "\n")

# Main loop
while True:
    script.batch_execute_agent()
    script.batch_receive()
