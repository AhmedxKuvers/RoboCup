from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



        



    def select_skill(self,strategyData):
        #--------------------------------------- 2. Decide action
        drawer = self.world.draw

        # -----------------------------------------------------------------
        # --- STEP 5: "MAIN BRAIN" - FIX KICK-OFF RULES ---
        # -----------------------------------------------------------------
        
        # --- Handle Our KickOff ---
        if strategyData.play_mode == self.world.M_OUR_KICKOFF:
            drawer.annotation((0,10.5), "GAME MODE: Our KickOff" , drawer.Color.white, "status")
            
            # Check if I am the active player (the one at the center)
            if strategyData.active_player_unum == strategyData.robot_model.unum:
                # --- FIX for Double-Touch Foul ---
                # Pass to our Midfielder (Player 4) instead of kicking into space
                # strategyData.teammate_positions is 0-indexed, so unum 4 is index 3
                target = strategyData.teammate_positions[3] 
                drawer.annotation((0,8.5), "Passing to Player 4" , drawer.Color.cyan, "pass_target")
                drawer.line(strategyData.mypos, target, 2, drawer.Color.cyan, "attack_line")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                # I am support: Hold my initial position
                drawer.clear("pass_target")
                return self.move(self.init_pos)

        # --- Handle Their KickOff ---
        elif strategyData.play_mode == self.world.M_THEIR_KICKOFF:
            drawer.annotation((0,10.5), "GAME MODE: Their KickOff" , drawer.Color.white, "status")
            # --- FIX for Illegal Position ---
            # Everyone holds their safe, initial position.
            return self.move(self.init_pos)

        # --- Handle Our Set Plays ---
        elif strategyData.PM_GROUP == self.world.MG_OUR_KICK:
            # Determine specific type of set play
            if strategyData.play_mode == self.world.M_OUR_CORNER_KICK:
                return self.handle_our_corner_kick(strategyData)
            elif strategyData.play_mode == self.world.M_OUR_FREE_KICK or strategyData.play_mode == self.world.M_OUR_DIR_FREE_KICK:
                return self.handle_our_free_kick(strategyData)
            elif strategyData.play_mode == self.world.M_OUR_KICK_IN:
                return self.handle_our_kick_in(strategyData)
            else:
                # Other set plays (goal kick, etc.) use normal strategy
                drawer.annotation((0,10.5), "GAME MODE: Our Set Play" , drawer.Color.white, "status")
                return self.run_play_on_strategy(strategyData)
        
        # --- Handle Their Set Plays ---
        elif strategyData.PM_GROUP == self.world.MG_THEIR_KICK:
            drawer.annotation((0,10.5), "GAME MODE: Their Set Play - DEFEND" , drawer.Color.white, "status")
            # Defensive positioning for opponent's set plays
            return self.handle_defensive_set_play(strategyData)
            
        # --- Handle Regular Gameplay ---
        elif strategyData.play_mode == self.world.M_PLAY_ON:
            # Run our normal PlayOn logic
            return self.run_play_on_strategy(strategyData)

        # --- Fallback / Do Nothing ---
        else:
            return self.move(strategyData.mypos)


    def handle_our_corner_kick(self, strategyData):
        """
        Specialized strategy for our corner kicks
        - Taker: Player closest to corner
        - Attackers: Move into the box
        - Target: Far post or best positioned player
        """
        drawer = self.world.draw
        drawer.annotation((0,10.5), "CORNER KICK: Our Attack" , drawer.Color.cyan, "status")
        
        ball_pos = strategyData.ball_2d
        
        # Determine which corner (left or right based on ball Y position)
        is_right_corner = ball_pos[1] > 0
        
        # Corner position
        corner_pos = np.array([15, 11 if is_right_corner else -11])
        
        # Check if I'm the taker (closest to ball)
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            # I'm taking the corner
            drawer.annotation((0, 9), "CORNER TAKER: Looking for target" , drawer.Color.green, "role")
            
            # Target positions in the box
            near_post = np.array([15, 3 if is_right_corner else -3])
            far_post = np.array([15, -5 if is_right_corner else 5])
            penalty_spot = np.array([13.5, 0])
            
            # Find best teammate to target (simple version - just closest to goal in box)
            best_target = None
            best_dist_to_goal = 1000
            
            for i, teammate_pos in enumerate(strategyData.teammate_positions):
                teammate_unum = i + 1
                if teammate_unum == strategyData.player_unum:
                    continue
                
                # Prefer players in the box (x > 10, |y| < 7)
                if teammate_pos[0] > 10 and abs(teammate_pos[1]) < 7:
                    dist_to_goal = np.linalg.norm(teammate_pos - np.array([15, 0]))
                    if dist_to_goal < best_dist_to_goal:
                        best_dist_to_goal = dist_to_goal
                        best_target = teammate_pos
            
            # If no teammate in box, aim for far post
            if best_target is None:
                best_target = far_post
            
            drawer.line(ball_pos, best_target, 3, drawer.Color.cyan, "corner_target")
            return self.kickTarget(strategyData, strategyData.mypos, best_target)
        
        else:
            # I'm an attacker - move into the box
            drawer.annotation((0, 9), "CORNER: Moving to box" , drawer.Color.yellow, "role")
            
            # Different positions based on player number
            my_unum = strategyData.player_unum
            
            if my_unum == 1:  # Goalkeeper stays back
                target_pos = np.array([-13, 0])
            elif my_unum == 2:  # Defender stays back for counter-attack defense
                target_pos = np.array([-5, 0])
            elif my_unum == 3:  # Near post
                target_pos = np.array([14.5, 3 if is_right_corner else -3])
            elif my_unum == 4:  # Far post
                target_pos = np.array([14.5, -5 if is_right_corner else 5])
            else:  # Penalty spot area
                target_pos = np.array([13, -2 if (my_unum % 2 == 0) else 2])
            
            drawer.line(strategyData.mypos, target_pos, 2, drawer.Color.green, "corner_move")
            return self.move(target_pos, orientation=M.target_abs_angle(target_pos, ball_pos))


    def handle_our_free_kick(self, strategyData):
        """
        Specialized strategy for our free kicks
        - Close to goal: Direct shot
        - Far from goal: Pass to open player
        """
        drawer = self.world.draw
        drawer.annotation((0,10.5), "FREE KICK: Our Attack" , drawer.Color.cyan, "status")
        
        ball_pos = strategyData.ball_2d
        opponent_goal = np.array([15, 0])
        dist_to_goal = np.linalg.norm(ball_pos - opponent_goal)
        
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            # I'm taking the free kick
            
            if dist_to_goal < 10:  # Close enough for direct shot
                drawer.annotation((0, 9), "FREE KICK: Direct shot!" , drawer.Color.green, "role")
                drawer.line(ball_pos, opponent_goal, 3, drawer.Color.red, "free_kick_shot")
                return self.kickTarget(strategyData, strategyData.mypos, opponent_goal)
            else:
                # Too far - find closest forward teammate
                drawer.annotation((0, 9), "FREE KICK: Looking for pass" , drawer.Color.green, "role")
                
                best_target = None
                best_score = -1000
                
                for i, teammate_pos in enumerate(strategyData.teammate_positions):
                    teammate_unum = i + 1
                    if teammate_unum == strategyData.player_unum:
                        continue
                    
                    # Score based on being forward and not too far
                    score = 0
                    if teammate_pos[0] > ball_pos[0]:  # Ahead of ball
                        score += 100
                    
                    dist_from_ball = np.linalg.norm(teammate_pos - ball_pos)
                    if dist_from_ball < 15:  # Not too far
                        score += (15 - dist_from_ball) * 5
                    
                    dist_to_goal_from_teammate = np.linalg.norm(teammate_pos - opponent_goal)
                    score += (30 - dist_to_goal_from_teammate) * 3
                    
                    if score > best_score:
                        best_score = score
                        best_target = teammate_pos
                
                if best_target is not None:
                    drawer.line(ball_pos, best_target, 3, drawer.Color.cyan, "free_kick_pass")
                    return self.kickTarget(strategyData, strategyData.mypos, best_target)
                else:
                    # No good pass - kick forward
                    forward_target = ball_pos + np.array([5, 0])
                    return self.kickTarget(strategyData, strategyData.mypos, forward_target)
        else:
            # I'm support - create passing options
            drawer.annotation((0, 9), "FREE KICK: Creating space" , drawer.Color.yellow, "role")
            
            # Use basic formation but push forward
            formation_positions = GenerateBasicFormation()
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
            target_pos = point_preferences[strategyData.player_unum]
            
            # Push forward a bit more for free kicks
            if target_pos[0] < 10 and strategyData.player_unum > 2:
                target_pos = target_pos + np.array([3, 0])
            
            drawer.line(strategyData.mypos, target_pos, 2, drawer.Color.green, "free_kick_move")
            return self.move(target_pos, orientation=strategyData.ball_dir)


    def handle_our_kick_in(self, strategyData):
        """
        Specialized strategy for our kick-ins (throw-ins)
        - Quick restart
        - Pass to nearest open teammate
        """
        drawer = self.world.draw
        drawer.annotation((0,10.5), "KICK-IN: Quick restart" , drawer.Color.cyan, "status")
        
        ball_pos = strategyData.ball_2d
        
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            # I'm taking the kick-in
            drawer.annotation((0, 9), "KICK-IN: Finding nearby player" , drawer.Color.green, "role")
            
            # Find closest teammate (simple version)
            best_target = None
            best_dist = 1000
            
            for i, teammate_pos in enumerate(strategyData.teammate_positions):
                teammate_unum = i + 1
                if teammate_unum == strategyData.player_unum:
                    continue
                
                dist = np.linalg.norm(teammate_pos - ball_pos)
                
                # Prefer closer teammates within 8m (quick restart)
                if dist < 8 and dist > 2 and dist < best_dist:  # Not too close, not too far
                    best_dist = dist
                    best_target = teammate_pos
            
            if best_target is not None:
                drawer.line(ball_pos, best_target, 3, drawer.Color.cyan, "kick_in_target")
                return self.kickTarget(strategyData, strategyData.mypos, best_target)
            else:
                # No nearby teammate - kick forward down the line
                forward_target = ball_pos + np.array([5, 0])
                return self.kickTarget(strategyData, strategyData.mypos, forward_target)
        else:
            # I'm support - move to receive
            drawer.annotation((0, 9), "KICK-IN: Ready to receive" , drawer.Color.yellow, "role")
            
            # Move slightly away from the touchline to create space
            target_y = 0 if abs(strategyData.mypos[1]) > 8 else strategyData.mypos[1]
            target_pos = np.array([strategyData.mypos[0], target_y])
            
            # If I'm close to the kick-in, move to receive
            if np.linalg.norm(strategyData.mypos - ball_pos) < 8:
                # Position myself to receive
                receive_pos = ball_pos + np.array([2, -2 if ball_pos[1] > 0 else 2])
                drawer.line(strategyData.mypos, receive_pos, 2, drawer.Color.green, "kick_in_receive")
                return self.move(receive_pos, orientation=strategyData.ball_dir)
            else:
                # Hold position
                return self.move(strategyData.mypos, orientation=strategyData.ball_dir)


    def handle_defensive_set_play(self, strategyData):
        """
        Defensive strategy for opponent's set plays
        - Mark opponents
        - Protect goal
        - Stay compact
        """
        drawer = self.world.draw
        ball_pos = strategyData.ball_2d
        
        # Determine type of defensive set play
        if strategyData.play_mode == self.world.M_THEIR_CORNER_KICK:
            drawer.annotation((0, 9), "DEFEND: Corner kick" , drawer.Color.red, "defend_type")
            
            # Everyone defend the box except goalkeeper
            if strategyData.player_unum == 1:  # Goalkeeper
                target_pos = np.array([-13, 0])
            else:
                # Defend near posts and penalty area
                is_right_corner = ball_pos[1] > 0
                
                if strategyData.player_unum == 2:  # Near post
                    target_pos = np.array([-14.5, 3 if is_right_corner else -3])
                elif strategyData.player_unum == 3:  # Far post
                    target_pos = np.array([-14.5, -4 if is_right_corner else 4])
                else:  # Others in the box
                    target_pos = np.array([-12, -2 + strategyData.player_unum])
            
        elif strategyData.play_mode in (self.world.M_THEIR_FREE_KICK, self.world.M_THEIR_DIR_FREE_KICK):
            drawer.annotation((0, 9), "DEFEND: Free kick" , drawer.Color.red, "defend_type")
            
            dist_to_our_goal = np.linalg.norm(ball_pos - np.array([-15, 0]))
            
            if dist_to_our_goal < 12:  # Dangerous free kick - form wall
                # Form a defensive wall
                if strategyData.player_unum == 1:  # Goalkeeper
                    target_pos = np.array([-13, 0])
                elif strategyData.player_unum in [2, 3, 4]:  # Wall players
                    wall_y = 0 if strategyData.player_unum == 2 else (1.5 if strategyData.player_unum == 3 else -1.5)
                    target_pos = ball_pos + np.array([-2, wall_y])  # 2m from ball
                else:
                    target_pos = np.array([-10, 0])
            else:
                # Regular defensive formation - use basic formation
                formation_positions = GenerateBasicFormation()
                point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
                target_pos = point_preferences[strategyData.player_unum]
        else:
            # General defensive positioning - use basic formation
            formation_positions = GenerateBasicFormation()
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
            target_pos = point_preferences[strategyData.player_unum]
        
        drawer.line(strategyData.mypos, target_pos, 2, drawer.Color.red, "defend_position")
        return self.move(target_pos, orientation=strategyData.ball_dir)


    def run_play_on_strategy(self, strategyData):
        """
        This is the "PlayOn" logic.
        It decides if we are an ATTACKER or SUPPORT player.
        """
        # Check if I am the active player (closest to the ball)
        if strategyData.active_player_unum == strategyData.robot_model.unum: 
            # --- I AM THE ACTIVE PLAYER ---
            drawer = self.world.draw
            drawer.annotation((0,10.5), "ACTIVE: Attacking" , drawer.Color.red, "status")
            target = (15,0) # Opponent goal
            drawer.line(strategyData.mypos, target, 2, drawer.Color.red, "attack_line")
            drawer.clear("target_line") # Clear the blue formation line
            drawer.clear("retreat_line") # Clear the orange retreat line

            return self.kickTarget(strategyData, strategyData.mypos, target)

        else:
            # --- I AM A SUPPORT PLAYER ---
            # Run the dedicated support logic
            return self.run_support_strategy(strategyData)
    
    def run_support_strategy(self, strategyData):
        """
        This is the logic for a support player.
        We've moved it into its own function to be called from multiple game modes.
        """
        drawer = self.world.draw
        
        # --- I AM A SUPPORT PLAYER ---
        drawer.annotation((0,10.5), "SUPPORT: Moving to position" , drawer.Color.yellow, "status")
        
        formation_positions = GenerateBasicFormation()
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]

        # --- Give the Active Player Space ---
        ball_pos = strategyData.ball_2d
        assigned_pos = strategyData.my_desired_position
        
        dist_to_ball = np.linalg.norm(assigned_pos - ball_pos)
        MIN_SUPPORT_DIST = 2.5 
        
        final_target = assigned_pos
        drawer.clear("retreat_line") 
        
        if dist_to_ball < MIN_SUPPORT_DIST:
            vec_from_ball = assigned_pos - ball_pos
            if np.linalg.norm(vec_from_ball) > 0.1:
                norm_vec = vec_from_ball / np.linalg.norm(vec_from_ball)
                final_target = ball_pos + (norm_vec * MIN_SUPPORT_DIST)
            else:
                final_target = np.array([ball_pos[0] - MIN_SUPPORT_DIST, ball_pos[1]])
            drawer.line(assigned_pos, final_target, 1, drawer.Color.orange, "retreat_line")

        drawer.line(strategyData.mypos, final_target, 2, drawer.Color.blue, "target_line")
        drawer.clear("attack_line") 
        
        # Move to the final target, but face the ball
        return self.move(final_target, orientation=strategyData.ball_dir)
































    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")