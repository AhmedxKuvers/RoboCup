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
        # --- STEP 4: "MAIN BRAIN" - SMART KICK-OFF ---
        # -----------------------------------------------------------------
        
        # --- Handle Our KickOff ---
        if strategyData.play_mode == self.world.M_OUR_KICKOFF:
            drawer.annotation((0,10.5), "GAME MODE: Our KickOff" , drawer.Color.white, "status")
            
            # Check if I am the active player (the one at the center)
            if strategyData.active_player_unum == strategyData.robot_model.unum:
                # I am active: Kick the ball forward to start play
                target = (5, 0) # Kick it 5 meters forward
                drawer.line(strategyData.mypos, target, 2, drawer.Color.red, "attack_line")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                # I am support: Hold my initial position
                return self.move(self.init_pos)

        # --- Handle Their KickOff ---
        elif strategyData.play_mode == self.world.M_THEIR_KICKOFF:
            drawer.annotation((0,10.5), "GAME MODE: Their KickOff" , drawer.Color.white, "status")
            # We are on defense. Everyone acts as a support player.
            # Run to our defensive spots but DO NOT attack the ball.
            return self.run_support_strategy(strategyData)

        # --- Handle Set Plays (KickIn, Corner, etc.) ---
        elif strategyData.PM_GROUP == self.world.MG_OUR_KICK or strategyData.PM_GROUP == self.world.MG_THEIR_KICK:
            drawer.annotation((0,10.5), "GAME MODE: Set Play" , drawer.Color.white, "status")
            # Run our normal PlayOn logic: active player attacks, support players position.
            return self.run_play_on_strategy(strategyData)
            
        # --- Handle Regular Gameplay ---
        elif strategyData.play_mode == self.world.M_PLAY_ON:
            # Run our normal PlayOn logic
            return self.run_play_on_strategy(strategyData)

        # --- Fallback / Do Nothing ---
        else:
            return self.move(strategyData.mypos)


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