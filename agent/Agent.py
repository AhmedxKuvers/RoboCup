from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment, get_formation, pass_reciever_selector
from strategy.Strategy import Strategy 


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



        



    def select_skill(self, strategyData):
        """
        Enhanced decision-making with:
        - Goalkeeper special behavior
        - Game mode awareness (kickoff, kick-ins, corners)
        - Defensive tactics when opponent has possession
        - Better spacing and positioning
        """
        drawer = self.world.draw
        path_draw_options = self.path_manager.draw_options
        my_unum = strategyData.robot_model.unum
        w = self.world
        
        # ==================== GOALKEEPER BEHAVIOR ====================
        if my_unum == 1:
            drawer.annotation((0, 10.5), "Goalkeeper", drawer.Color.cyan, "status")
            
            # GK position between ball and goal
            gk_target = strategyData.get_goalkeeper_position()
            
            # If ball is very close and I'm closest, go for it
            if strategyData.active_player_unum == my_unum and strategyData.ball_dist < 3.0:
                # Clear the ball away from goal
                if strategyData.should_clear_ball():
                    # Kick toward sideline or upfield
                    clear_target = np.array([0.0, np.sign(strategyData.ball_2d[1]) * 8.0])
                    drawer.line(strategyData.mypos, clear_target, 3, drawer.Color.orange, "gk_clear")
                    return self.kickTarget(strategyData, strategyData.mypos, clear_target)
            
            # Normal GK positioning
            drawer.line(strategyData.mypos, gk_target, 2, drawer.Color.cyan, "gk_line")
            drawer.clear("gk_clear")
            return self.move(gk_target, orientation=strategyData.ball_dir)
        
        # ==================== GAME MODE HANDLING ====================
        
        # KICKOFF situations
        if strategyData.play_mode == w.M_OUR_KICKOFF:
            drawer.annotation((0, 10.5), "Our Kickoff", drawer.Color.green, "status")
            kickoff_pos = strategyData.get_kickoff_position(my_unum, self.init_pos, is_our_kickoff=True)
            
            if my_unum == strategyData.active_player_unum:
                # Player taking kickoff - kick forward to teammate
                target = pass_reciever_selector(strategyData.player_unum, strategyData.teammate_positions, (5, 0))
                drawer.line(strategyData.mypos, target, 2, drawer.Color.yellow, "kickoff_pass")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                drawer.clear("kickoff_pass")
                return self.move(kickoff_pos, orientation=strategyData.ball_dir)
        
        if strategyData.play_mode == w.M_THEIR_KICKOFF:
            drawer.annotation((0, 10.5), "Their Kickoff - Defend", drawer.Color.red, "status")
            kickoff_pos = strategyData.get_kickoff_position(my_unum, self.init_pos, is_our_kickoff=False)
            return self.move(kickoff_pos, orientation=strategyData.ball_dir)
        
        # KICK-IN situations
        if strategyData.play_mode in [w.M_OUR_KICK_IN, w.M_OUR_GOAL_KICK]:
            drawer.annotation((0, 10.5), "Our Kick-In", drawer.Color.green, "status")
            kick_in_pos = strategyData.get_kick_in_position(my_unum, is_our_kick_in=True)
            
            if my_unum == strategyData.active_player_unum:
                # Take the kick-in
                target = pass_reciever_selector(strategyData.player_unum, strategyData.teammate_positions, (15, 0))
                drawer.line(strategyData.mypos, target, 2, drawer.Color.green, "kick_in_line")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                drawer.clear("kick_in_line")
                return self.move(kick_in_pos, orientation=strategyData.ball_dir)
        
        if strategyData.play_mode in [w.M_THEIR_KICK_IN, w.M_THEIR_GOAL_KICK]:
            drawer.annotation((0, 10.5), "Their Kick-In - Defend", drawer.Color.red, "status")
            kick_in_pos = strategyData.get_kick_in_position(my_unum, is_our_kick_in=False)
            return self.move(kick_in_pos, orientation=strategyData.ball_dir)
        
        # CORNER KICK situations
        if strategyData.play_mode == w.M_OUR_CORNER_KICK:
            drawer.annotation((0, 10.5), "Our Corner", drawer.Color.green, "status")
            
            if my_unum == strategyData.active_player_unum:
                # Kick toward goal area
                corner_target = np.array([13.0, np.sign(strategyData.ball_2d[1]) * -2.0])
                drawer.line(strategyData.mypos, corner_target, 2, drawer.Color.green, "corner_line")
                return self.kickTarget(strategyData, strategyData.mypos, corner_target)
            else:
                # Position in goal area for header/shot
                attack_pos = np.array([12.0, (my_unum - 3) * 2.5])
                drawer.clear("corner_line")
                return self.move(attack_pos, orientation=strategyData.ball_dir)
        
        if strategyData.play_mode == w.M_THEIR_CORNER_KICK:
            drawer.annotation((0, 10.5), "Their Corner - Defend Goal", drawer.Color.red, "status")
            # Defend the goal area
            defend_pos = np.array([-12.0, (my_unum - 3) * 2.0])
            return self.move(defend_pos, orientation=strategyData.ball_dir)
        
        # ==================== NORMAL PLAY ====================
        drawer.clear("status")
        
        # Active player pursues the ball - does NOT follow formation
        if strategyData.active_player_unum == my_unum:
            # Check if we should clear under pressure
            if strategyData.should_clear_ball():
                drawer.annotation((0, 10.5), "CLEAR BALL!", drawer.Color.orange, "status")
                # Just boot it upfield
                clear_target = np.array([15.0, 0.0])
                drawer.line(strategyData.mypos, clear_target, 3, drawer.Color.orange, "attack_line")
                return self.kickTarget(strategyData, strategyData.mypos, clear_target)
            
            # Normal attack
            drawer.annotation((0, 10.5), "Active - Attack", drawer.Color.yellow, "status")
            target = pass_reciever_selector(strategyData.player_unum, strategyData.teammate_positions, (15, 0))
            drawer.line(strategyData.mypos, target, 2, drawer.Color.red, "attack_line")
            
            # Use aggressive movement and avoid priority teammates
            return self.kickTarget(strategyData, strategyData.mypos, target)
        
        # Non-active players follow formation with defensive adjustments
        drawer.clear("attack_line")
        
        # Dynamic formation based on ball x
        formation_positions = get_formation(strategyData.ball_2d[0])
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        
        # My assigned position
        assigned_pos = point_preferences.get(my_unum, np.asarray(strategyData.mypos))
        
        # Apply defensive adjustment if opponent has possession
        if strategyData.min_opponent_ball_dist + 1.0 < strategyData.min_teammate_ball_dist:
            drawer.annotation((0, 10.5), "Defending", drawer.Color.red, "status")
            assigned_pos = strategyData.get_defensive_position(my_unum, assigned_pos)
        
        strategyData.my_desired_position = assigned_pos
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )
        
        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "formation_line")
        
        # Move to formation position, giving priority to active player
        return self.move(
            strategyData.my_desired_position, 
            orientation=strategyData.ball_dir,
            priority_unums=[strategyData.active_player_unum]
        )
        
































    

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