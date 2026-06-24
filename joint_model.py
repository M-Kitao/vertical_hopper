import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os
import mujoco
import math

class Joint_model_Env(MujocoEnv, utils.EzPickle):
    m_torso = 53.5
    m_thigh = 8.5
    m_shank = 3.5
    m_ankle = 0
    m_foot  = 1.25
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    def __init__(self,path):
        utils.EzPickle.__init__(self,)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64)
        # frameskip, observationspace
        MujocoEnv.__init__(self, os.path.join(path, "vertical_hopper.xml"), 4, observation_space=observation_space) ##**kwargs

    
    def fall_judge(self):
        flag = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom2 == self.model.geom('hat_geom').id:
                flag = True
            elif contact.geom2 == self.model.geom('knee_geom').id:
                flag = True
            elif contact.geom2 == self.model.geom('left_knee_geom').id:
                flag = True
            elif contact.geom2 == self.model.geom('hip_geom').id:
                flag = True
        return flag

    def actuator_force(self):
        force = self.data.qfrc_actuator
        actuator_output = np.array([force[7],    force[8], force[9], force[10], force[11],    force[13], force[14], force[15], force[16]]) ###### Be careful!!
        return actuator_output
    
    def model_mass(self):
        mass_sum = self.m_torso + 2*self.m_thigh + 2*self.m_shank + 2*self.m_ankle + 2*self.m_foot
        return mass_sum


    def step(self, action, num=0, vel=False, vel_flag=False):
        ######################################
        self.do_simulation(action, self.frame_skip)
        ######################################
        self.data.qpos[self.model.joint('camerax').id] = self.data.qpos[self.model.joint('rootx').id]

        height = self.data.qpos[self.model.joint('rootz').id]

        ####reward####
        reward = 1  ### not used, but for the future research
        fall_flag = self.fall_judge()
        if num>700 and vel_flag:
            if abs(np.average(vel[-700:]))<0.05:
                stop_flag = True
            else:
                stop_flag = False
        else:
            stop_flag = False
        done = not (height > -0.9 and fall_flag==False and stop_flag==False)
        ob = self._get_obs()
        
        self.render_mode = "human"
        if self.render_mode == "human":
            self.render()
        return ob, reward, done, {}



    def _get_obs(self):
        angles = self.joints_angles()
        vels = self.joints_vels()

        _, _, right_x_N, left_x_N, right_z_N, left_z_N = self.grf()

        obs = np.array([
            angles[0], # torso
            angles[1], # Rhip
            angles[2], # Rknee
            angles[3], # Rankle
            angles[4], # Lhip
            angles[5], # Lknee
            angles[6], # Lankle
            vels[0], # torso
            vels[1], # Rhip
            vels[2], # Rknee
            vels[3], # Rankle
            vels[4], # Lhip
            vels[5], # Lknee
            vels[6], # Lankle
            right_x_N/self.model_mass()/9.8, # normalized by its body weight
            left_x_N/self.model_mass()/9.8,
            right_z_N/self.model_mass()/9.8,
            left_z_N/self.model_mass()/9.8,
            ### for 3D
            angles[7],
            angles[8],
            vels[7],
            vels[8]
            ])
        return obs

    def reset_model(self, initial_state, num=0):
        if initial_state:
            self.data.qpos[self.model.joint('rootx').id] = 0
            self.data.qpos[self.model.joint('rootz').id] = -1.86801687e-02
            self.data.qpos[self.model.joint('rooty').id] = 0
            self.data.qpos[self.model.joint('roll').id] = 0
            self.data.qpos[self.model.joint('pitch').id] = 0
            self.data.qpos[self.model.joint('yaw').id] = 0
            self.data.qpos[self.model.joint('torso_joint').id] = initial_state["init_tor_pos"]
            self.data.qpos[self.model.joint('hip_joint').id] = -1.73920841e-01
            self.data.qpos[self.model.joint('hip_adduction_joint').id] = 0
            self.data.qpos[self.model.joint('knee_joint').id] = 6.23753292e-04
            self.data.qpos[self.model.joint('ankle_joint').id] = 1.73329897e-01
            self.data.qpos[self.model.joint('left_hip_joint').id] = initial_state["init_L_hip_pos"]
            self.data.qpos[self.model.joint('left_hip_adduction_joint').id] = 0
            self.data.qpos[self.model.joint('left_knee_joint').id] = initial_state["init_L_kne_pos"]
            self.data.qpos[self.model.joint('left_ankle_joint').id] = initial_state["init_L_ank_pos"]
    
            self.data.qvel[self.model.joint('rootx').id] = initial_state["init_x_vel"]
            self.data.qvel[self.model.joint('rootz').id] = initial_state["init_z_vel"]
            self.data.qvel[self.model.joint('rooty').id] = initial_state["init_z_vel"]
            self.data.qvel[self.model.joint('roll').id] = initial_state["init_z_vel"]
            self.data.qvel[self.model.joint('pitch').id] = initial_state["init_z_vel"]
            self.data.qvel[self.model.joint('yaw').id] = initial_state["init_z_vel"]
            self.data.qvel[self.model.joint('torso_joint').id] = initial_state["init_tor_vel"]
            self.data.qvel[self.model.joint('hip_joint').id] = initial_state["init_R_hip_vel"]
            self.data.qvel[self.model.joint('hip_adduction_joint').id] = 0
            self.data.qvel[self.model.joint('knee_joint').id] = initial_state["init_R_kne_vel"]
            self.data.qvel[self.model.joint('ankle_joint').id] = initial_state["init_R_ank_vel"]
            self.data.qvel[self.model.joint('left_hip_joint').id] = initial_state["init_L_hip_vel"]
            self.data.qvel[self.model.joint('left_hip_adduction_joint').id] = 0
            self.data.qvel[self.model.joint('left_knee_joint').id] = initial_state["init_L_kne_vel"]
            self.data.qvel[self.model.joint('left_ankle_joint').id] = initial_state["init_L_ank_vel"]
            
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def pos(self):
        return self.data.qpos[self.model.joint('rootx').id]
        
    def vel(self):
        return math.sqrt(self.data.qvel[self.model.joint('rootx').id]*self.data.qvel[self.model.joint('rootx').id] + self.data.qvel[self.model.joint('rooty').id]*self.data.qvel[self.model.joint('rooty').id])


    def state_machine(self, rleg_state, lleg_state):
        if rleg_state == 'stance':
            right_touch_flag, _, _, _, _, _ = self.grf()
            if self.data.qpos[self.model.joint('hip_joint').id]<0 and right_touch_flag==False:
                rleg_state = 'swing'
        elif rleg_state == 'swing':
            right_touch_flag, _, _, _, _, _ = self.grf()
            if self.data.qpos[self.model.joint('hip_joint').id]>0 and right_touch_flag:
                rleg_state = 'stance'

        if lleg_state == 'stance':
            _, left_touch_flag, _, _, _, _ = self.grf()
            if self.data.qpos[self.model.joint('left_hip_joint').id]<0 and left_touch_flag==False:
                lleg_state = 'swing'
        elif lleg_state == 'swing':
            _, left_touch_flag, _, _, _, _ = self.grf()
            if self.data.qpos[self.model.joint('left_hip_joint').id]> 0 and left_touch_flag:
                lleg_state = 'stance'

        return rleg_state, lleg_state
        

    def grf(self):
        right_x_N = 0
        left_x_N = 0
        right_z_N = 0
        left_z_N = 0
        z_N = 0
        x_N = 0
        touch = False
        right_touch = False
        left_touch = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            if contact.geom2 == self.model.geom('right_heel_geom').id:
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            if contact.geom2 == self.model.geom('right_foot_geom').id:
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            if contact.geom2 == self.model.geom('right_toe_geom').id:
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]

            if contact.geom2 == self.model.geom('left_heel_geom').id:
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
            if contact.geom2 == self.model.geom('left_foot_geom').id:
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
            if contact.geom2 == self.model.geom('left_toe_geom').id:
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]

            if contact.geom2 == self.model.geom('hip_geom').id:
                touch = True
                x_N += -c_array[2]
                z_N += c_array[0]
            if contact.geom2 == self.model.geom('knee_geom').id:
                touch = True
                x_N += -c_array[2]
                z_N += c_array[0]
            if contact.geom2 == self.model.geom('shank_geom').id:
                touch = True
                x_N += -c_array[2]
                z_N += c_array[0]
            if contact.geom2 == self.model.geom('footsphere').id:
                touch = True
                x_N += -c_array[2]
                z_N += c_array[0]
                
        return right_touch, left_touch, right_x_N, left_x_N, right_z_N, left_z_N, touch, x_N, z_N
    

    def grf_y(self):
        right_y_N = 0
        left_y_N = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            if contact.geom2 == self.model.geom('right_heel_geom').id:
                right_y_N += c_array[1]
            if contact.geom2 == self.model.geom('right_foot_geom').id:
                right_y_N += c_array[1]
            if contact.geom2 == self.model.geom('right_toe_geom').id:
                right_y_N += c_array[1]

            if contact.geom2 == self.model.geom('left_heel_geom').id:
                left_y_N += -c_array[1]
            if contact.geom2 == self.model.geom('left_foot_geom').id:
                left_y_N += -c_array[1]
            if contact.geom2 == self.model.geom('left_toe_geom').id:
                left_y_N += -c_array[1]
                
        return right_y_N/self.model_mass()/9.8, left_y_N/self.model_mass()/9.8
    

    def joints_angles(self):
        torso = self.data.qpos[self.model.joint('torso_joint').id]
        hip = self.data.qpos[self.model.joint('hip_joint').id] + self.data.qpos[self.model.joint('torso_joint').id]
        knee = self.data.qpos[self.model.joint('knee_joint').id]
        ankle = self.data.qpos[self.model.joint('ankle_joint').id]
        L_hip = self.data.qpos[self.model.joint('left_hip_joint').id] + self.data.qpos[self.model.joint('torso_joint').id]
        L_knee = self.data.qpos[self.model.joint('left_knee_joint').id]
        L_ankle = self.data.qpos[self.model.joint('left_ankle_joint').id]
        hip_adduction_joint = self.data.qpos[self.model.joint('hip_adduction_joint').id]
        L_hip_adduction_joint = self.data.qpos[self.model.joint('left_hip_adduction_joint').id]
        return np.array([torso, hip, knee, ankle, L_hip, L_knee, L_ankle, hip_adduction_joint, L_hip_adduction_joint])

    def joints_vels(self):
        torso = self.data.qvel[self.model.joint('torso_joint').id]
        hip = self.data.qvel[self.model.joint('hip_joint').id] + self.data.qvel[self.model.joint('torso_joint').id]
        knee = self.data.qvel[self.model.joint('knee_joint').id]
        ankle = self.data.qvel[self.model.joint('ankle_joint').id]
        L_hip = self.data.qvel[self.model.joint('left_hip_joint').id] + self.data.qvel[self.model.joint('torso_joint').id]
        L_knee = self.data.qvel[self.model.joint('left_knee_joint').id]
        L_ankle = self.data.qvel[self.model.joint('left_ankle_joint').id]
        hip_adduction_joint = self.data.qvel[self.model.joint('hip_adduction_joint').id]
        L_hip_adduction_joint = self.data.qvel[self.model.joint('left_hip_adduction_joint').id]
        return np.array([torso, hip, knee, ankle, L_hip, L_knee, L_ankle, hip_adduction_joint, L_hip_adduction_joint])


    def torso_pos(self):
        return self.data.qpos[self.model.joint('torso_joint').id]

    def torso_vel(self):
        return self.data.qvel[self.model.joint('torso_joint').id]

    def hip_pos(self):
        return self.data.qpos[self.model.joint('hip_joint').id], self.data.qpos[self.model.joint('left_hip_joint').id]
    
    def hip_adduction_pos(self):
        return self.data.qpos[self.model.joint('hip_adduction_joint').id], self.data.qpos[self.model.joint('left_hip_adduction_joint').id]
    
    def knee_pos(self):
        return self.data.qpos[self.model.joint('knee_joint').id], self.data.qpos[self.model.joint('left_knee_joint').id]
    
    def ankle_pos(self):
        return self.data.qpos[self.model.joint('ankle_joint').id], self.data.qpos[self.model.joint('left_ankle_joint').id]

    
    def roll(self):
        return self.data.qpos[self.model.joint('roll').id]
    
    def pitch(self):
        return self.data.qpos[self.model.joint('pitch').id]
    

    def com_position(self):
        hat_mass = 53.5
        thigh_mass = 8.5
        shank_mass = 3.5
        heel_mass = 0.15
        foot_mass = 0.9
        toe_mass = 0.2
        total_mass = hat_mass + thigh_mass*2 + shank_mass*2 + heel_mass*2 + foot_mass*2 + toe_mass*2
        com = (
            hat_mass*(self.data.site_xpos[self.model.site('hat_geom_site').id])
            + thigh_mass*(self.data.site_xpos[self.model.site('thigh_geom_site').id])
            + shank_mass*(self.data.site_xpos[self.model.site('shank_geom_site').id])
            + heel_mass*(self.data.site_xpos[self.model.site('right_heel_geom_site').id])
            + foot_mass*(self.data.site_xpos[self.model.site('right_foot_geom_site').id])
            + toe_mass*(self.data.site_xpos[self.model.site('right_toe_geom_site').id])
            #########
            + thigh_mass*(self.data.site_xpos[self.model.site('left_thigh_geom_site').id])
            + shank_mass*(self.data.site_xpos[self.model.site('left_shank_geom_site').id])
            + heel_mass*(self.data.site_xpos[self.model.site('left_heel_geom_site').id])
            + foot_mass*(self.data.site_xpos[self.model.site('left_foot_geom_site').id])
            + toe_mass*(self.data.site_xpos[self.model.site('left_toe_geom_site').id])
        ) / total_mass
        # print(com)
        return com
    

    def cop(self):
        com = self.com_position()
        right_z_N = 0
        left_z_N = 0
        right_moment_x = 0
        right_moment_y = 0
        left_moment_x = 0
        left_moment_y = 0
        right_touch = False
        left_touch = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            if contact.geom2 == self.model.geom('right_heel_geom').id:
                right_touch = True
                right_moment_y += c_array[0]*(contact.pos[0]-com[0])
                right_moment_x += c_array[0]*(contact.pos[1]-com[1])
                right_z_N += c_array[0]
            if contact.geom2 == self.model.geom('right_foot_geom').id:
                right_touch = True
                right_moment_y += c_array[0]*(contact.pos[0]-com[0])
                right_moment_x += c_array[0]*(contact.pos[1]-com[1])
                right_z_N += c_array[0]
            if contact.geom2 == self.model.geom('right_toe_geom').id:
                right_touch = True
                right_moment_y += c_array[0]*(contact.pos[0]-com[0])
                right_moment_x += c_array[0]*(contact.pos[1]-com[1])
                right_z_N += c_array[0]

            if contact.geom2 == self.model.geom('left_heel_geom').id:
                left_touch = True
                left_moment_y += c_array[0]*(contact.pos[0]-com[0])
                left_moment_x += c_array[0]*(contact.pos[1]-com[1])
                left_z_N += c_array[0]
            if contact.geom2 == self.model.geom('left_foot_geom').id:
                left_touch = True
                left_moment_y += c_array[0]*(contact.pos[0]-com[0])
                left_moment_x += c_array[0]*(contact.pos[1]-com[1])
                left_z_N += c_array[0]
            if contact.geom2 == self.model.geom('left_toe_geom').id:
                left_touch = True
                left_moment_y += c_array[0]*(contact.pos[0]-com[0])
                left_moment_x += c_array[0]*(contact.pos[1]-com[1])
                left_z_N += c_array[0]

        if right_touch and left_touch==False:
            right_cop_x = right_moment_y / right_z_N
            right_cop_y = right_moment_x / right_z_N
            left_cop_x = 0
            left_cop_y = 0
        elif right_touch==False and left_touch:
            right_cop_x = 0
            right_cop_y = 0
            left_cop_x = left_moment_y / left_z_N
            left_cop_y = left_moment_x / left_z_N
        else:
            right_cop_x = 0
            right_cop_y = 0
            left_cop_x = 0
            left_cop_y = 0
                
        return right_cop_x, right_cop_y, left_cop_x, left_cop_y

    def y_pos(self):
        return self.data.qpos[self.model.joint('rooty').id]