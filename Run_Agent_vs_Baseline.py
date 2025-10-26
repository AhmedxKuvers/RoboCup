#!/usr/bin/env python3
"""
Run your custom Agent against the Baseline Agent
Team 1: Your custom agent implementation (agent.Agent)
Team 2: Baseline/Vanilla agent implementation (agent.baseline.Baseline_Agent)
"""

from scripts.commons.Script import Script
script = Script() # Initialize: load config file, parse arguments, build cpp modules
a = script.args

from agent.Agent import Agent
from agent.baseline.Baseline_Agent import Baseline_Agent

# Team 1: Your custom agent (team name from arguments)
team_args = ((a.i, a.p, a.m, u, a.t, True, True) for u in range(1,6))
script.batch_create(Agent, team_args)

# Team 2: Baseline/Vanilla opponent agent
baseline_team_args = ((a.i, a.p, a.m, u, "Baseline", True, False) for u in range(1,6))
script.batch_create(Baseline_Agent, baseline_team_args)

# Main execution loop
while True:
    script.batch_execute_agent()
    script.batch_receive()
