import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M



class Strategy():
    def __init__(self, world):
        self.play_mode = world.play_mode
        self.robot_model = world.robot  
        self.my_head_pos_2d = self.robot_model.loc_head_position[:2]
        self.player_unum = self.robot_model.unum
        self.mypos = (world.teammates[self.player_unum-1].state_abs_pos[0],world.teammates[self.player_unum-1].state_abs_pos[1])
       
        self.side = 1
        if world.team_side_is_left:
            self.side = 0

        self.teammate_positions = [teammate.state_abs_pos[:2] if teammate.state_abs_pos is not None 
                                    else None
                                    for teammate in world.teammates
                                    ]
        
        self.opponent_positions = [opponent.state_abs_pos[:2] if opponent.state_abs_pos is not None 
                                    else None
                                    for opponent in world.opponents
                                    ]



        

        self.team_dist_to_ball = None
        self.team_dist_to_oppGoal = None
        self.opp_dist_to_ball = None

        self.prev_important_positions_and_values = None
        self.curr_important_positions_and_values = None
        self.point_preferences = None
        self.combined_threat_and_definedPositions = None


        self.my_ori = self.robot_model.imu_torso_orientation
        self.ball_2d = world.ball_abs_pos[:2]
        self.ball_vec = self.ball_2d - self.my_head_pos_2d
        self.ball_dir = M.vector_angle(self.ball_vec)
        self.ball_dist = np.linalg.norm(self.ball_vec)
        self.ball_sq_dist = self.ball_dist * self.ball_dist # for faster comparisons
        self.ball_speed = np.linalg.norm(world.get_ball_abs_vel(6)[:2])
        
        self.goal_dir = M.target_abs_angle(self.ball_2d,(15.05,0))

        self.PM_GROUP = world.play_mode_group

        self.slow_ball_pos = world.get_predicted_ball_pos(0.5) # predicted future 2D ball position when ball speed <= 0.5 m/s

        # list of squared distances between teammates (including self) and slow ball (sq distance is set to 1000 in some conditions)
        self.teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and (world.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                  else 1000 # force large distance if teammate does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in world.teammates ]

        # list of squared distances between opponents and slow ball (sq distance is set to 1000 in some conditions)
        self.opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and world.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                  else 1000 # force large distance if opponent does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in world.opponents ]

        self.min_teammate_ball_sq_dist = min(self.teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(self.min_teammate_ball_sq_dist)   # distance between ball and closest teammate
        self.min_opponent_ball_dist = math.sqrt(min(self.opponents_ball_sq_dist)) # distance between ball and closest opponent

        self.active_player_unum = self.teammates_ball_sq_dist.index(self.min_teammate_ball_sq_dist) + 1

        self.my_desired_position = self.mypos
        self.my_desired_orientation = self.ball_dir


    def GenerateTeamToTargetDistanceArray(self, target, world):
        for teammate in world.teammates:
            pass
        

    def IsFormationReady(self, point_preferences):
        
        is_formation_ready = True
        for i in range(1, 6):
            if i != self.active_player_unum: 
                teammate_pos = self.teammate_positions[i-1]

                if not teammate_pos is None:

                    distance = np.sum((teammate_pos - point_preferences[i]) **2)
                    if(distance > 0.3):
                        is_formation_ready = False

        return is_formation_ready

    def GetDirectionRelativeToMyPositionAndTarget(self,target):
        target_vec = target - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)

        return target_dir
    
    def get_goalkeeper_position(self):
        """
        Calculate optimal goalkeeper position based on ball location.
        GK should stay between ball and goal, with limited movement range.
        """
        ball_x, ball_y = self.ball_2d
        
        # Goal center at (-15, 0)
        goal_x = -15.0
        goal_y = 0.0
        
        # Calculate where GK should position between ball and goal
        # GK stays on a line from ball to goal, but limited range
        if ball_x < -10:  # Ball very close to our goal
            target_x = max(-14.5, ball_x + 1.0)  # Stay slightly in front
            target_y = np.clip(ball_y * 0.3, -2.0, 2.0)  # Limited lateral movement
        else:  # Ball further away
            # Position on the line from ball to goal center
            direction = np.array([goal_x - ball_x, goal_y - ball_y])
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                direction = direction / dist
                # Position 1-2 meters in front of goal
                target_x = goal_x + direction[0] * 1.5
                target_y = goal_y + direction[1] * 1.5
            else:
                target_x = -13.5
                target_y = 0.0
            
            # Clamp to reasonable GK area
            target_x = np.clip(target_x, -14.5, -11.0)
            target_y = np.clip(target_y, -3.0, 3.0)
        
        return np.array([target_x, target_y])
    
    def get_defensive_position(self, my_unum, assigned_formation_pos):
        """
        When opponent has possession, position defensively:
        - Stay between opponent and our goal
        - Mark nearest opponent or cover passing lanes
        """
        # If opponent is much closer to ball, shift back defensively
        if self.min_opponent_ball_dist + 1.0 < self.min_teammate_ball_dist:
            # Opponent has clear possession, fall back
            defensive_shift = np.array([-3.0, 0.0])
            new_pos = assigned_formation_pos + defensive_shift
            new_pos[0] = np.clip(new_pos[0], -14.5, assigned_formation_pos[0])
            return new_pos
        
        return assigned_formation_pos
    
    def get_kickoff_position(self, my_unum, init_pos, is_our_kickoff):
        """
        Handle kickoff positioning rules.
        - Can't cross midfield before kickoff
        - Player taking kickoff positions near ball
        """
        if is_our_kickoff:
            # Player closest to ball takes kickoff
            if my_unum == self.active_player_unum:
                return np.array([-0.5, 0.0])  # Just behind ball
            else:
                # Others stay in own half
                pos = np.array(init_pos)
                pos[0] = np.clip(pos[0], -14.5, -0.5)
                return pos
        else:
            # Opponent kickoff - stay in own half, away from center circle
            pos = np.array(init_pos)
            pos[0] = np.clip(pos[0], -14.5, -3.0)  # Stay back
            
            # Avoid center circle (radius 2.5 from center)
            if np.linalg.norm(pos) < 3.0:
                pos[0] = -3.5
            
            return pos
    
    def get_kick_in_position(self, my_unum, is_our_kick_in):
        """
        Position for kick-in situations.
        - Taking team player near ball
        - Others spread for passing options
        """
        if is_our_kick_in:
            if my_unum == self.active_player_unum:
                # Move to ball to take kick-in
                return self.ball_2d
            else:
                # Position for pass reception
                # Spread out along the field
                y_offset = (my_unum - 3) * 3.0  # Spread players
                return np.array([self.ball_2d[0] + 2.0, np.clip(y_offset, -9.0, 9.0)])
        else:
            # Opponent kick-in: mark their players, stay away from ball
            return self.get_defensive_position(my_unum, np.array(self.mypos))
    
    def should_clear_ball(self):
        """
        Determine if active player should just clear/kick ball away.
        Used when under pressure near our goal.
        """
        ball_x = self.ball_2d[0]
        
        # Ball in our defensive third and opponent is close
        if ball_x < -5.0 and self.min_opponent_ball_dist < 3.0:
            return True
        
        # Ball very close to our goal
        if ball_x < -10.0:
            return True
            
        return False

