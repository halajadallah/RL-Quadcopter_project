import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
    
    def hill(self, x):
        y = 4000000.0/(1+400*(x**2)) 
        return y
    
    def hill_all(self,x):
        return 20000.0/(1+100*(x)) 
    
    def hill_c(self, x, a, b, c):
        return a/(c+b*x)
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        #squared distance from positioin to target 
        dist_sq = np.sum((self.sim.pose[:3] - self.target_pos)**2)
        
        #reward = 100 - 1.5*(self.target_pos[2] - self.sim.pose[2]) - \
        #1.5*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()- 1.0*(abs(self.sim.pose[3:6])).sum() 
        
        z_comp= (self.target_pos[2] - self.sim.pose[2])
        y_comp = (self.target_pos[1] - self.sim.pose[1])
        x_comp = (self.target_pos[0] - self.sim.pose[0])
        
        #reward = self.hill(z_comp)- 2.0*np.log(1+abs(self.target_pos[:2] - self.sim.pose[:2]).sum()) \
        #- 3.0*np.log(1+abs(current_pose[3:6]).sum())
        
        #reward = self.hill(z_comp)- 1.2*np.log(1+abs(self.target_pos[:2] - self.sim.pose[:2]).sum()) \
        #- 1.5*np.log(1+abs(self.sim.pose[3:6]).sum())
        
        z_sq = (self.target_pos[2] - self.sim.pose[2])**2
        xy_sq = (self.target_pos[:2] - self.sim.pose[:2]).sum()**2
        a_sq = abs(self.sim.pose[3:]).sum()**2
        a_velocity_sq = abs(self.sim.angular_v[9:12]).sum()**2
        xy_velocity_sq = abs(self.sim.v[:2]).sum()**2
        z_vel_sq = self.sim.v[2]**2
        #reward = self.hill_all((z_sq+a_sq).sum()) - 1.2*np.log(1+abs(self.target_pos[:2] - self.sim.pose[:2]).sum()) - 1.0*np.log(1+a_sq)- 0.5*np.log(1+a_velocity_sq)-.5*np.log(1+xy_velocity_sq)
        
        #reward = self.hill_all(z_sq) - 1.0*np.log(1+abs(self.target_pos[:2] - self.sim.pose[:2]).sum()**2) - 1.0*np.log(1+a_sq)- 0.8*np.log(1+a_velocity_sq)-.5*np.log(1+xy_velocity_sq)
        
        #reward = self.hill_all(z_sq)- 2*self.hill_all(z_vel_sq)
        #reward = self.hill_c(z_sq,1,60,1) - 2* self.hill_c(z_vel_sq,1,80,1)
        #reward =  -3.0* self.hill_c(self.sim.v[2],2,10, 100)+self.hill_c(z_vel_sq,1,50,1)
        
        #reward = 1.0 - 2.1*self.hill_c(z_sq,1,1,1)-2.2*self.hill_c(z_vel_sq,1,1,1) - 2.55* self.hill_c(xy_sq,1,1,1) - 2.55*self.hill_c(xy_velocity_sq,1,1,1)
        
        reward = 1.0 - 0.2*self.hill_c(dist_sq, 2., 40., 1)+.1*self.hill_c(self.sim.v[2],1,20,50)
        #clipping reward to the interval [-1,1]
        if reward > 1:
            reward = 1
            print('reward clip :', reward)
        elif reward < -1:
            reward = -1
            print('reward clip : ', reward)
        
        #if (abs(self.sim.v[:2]).sum()+abs(self.sim.angular_v).sum()) > 0.5:
        #    reward = -10
        
        if (abs(self.sim.v[0])+abs(self.sim.v[1])).sum() > 0.4:
            reward += -5
        
        if abs(self.sim.angular_v).sum() > 0.4:
            reward += -2
            
        #if abs(self.sim.angular_v[1]) > 0.3:
        #    reward += -1
            
        #if abs(self.sim.angular_v[2]) > 0.3:
        #    reward += -2
            
        
        
        # does not respond well toward positive reward
        #if np.sqrt(dist_sq) < 30:
        #    reward += 80
        #elif np.sqrt(dist_sq) < 50:
        #    reward += 50
        #elif np.sqrt(dist_sq) < 80:
        #    reward += 30
        #elif np.sqrt(dist_sq) < 90:
        #    reward += 30
        #elif np.sqrt(dist_sq) < 95:
        #    reward += 20
        #else:
        #    reward += 10
         
        #if np.sqrt(dist_sq) > 95:
        #    reward += -10
         
        if np.sqrt(dist_sq) < 5:
            reward -= 100
        
        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        #done = bool(((self.sim.pose[:3] - self.target_pos)**2).sum() <= 10) or done
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state