# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, quaternion_to_matrix, compute_heading_and_up, compute_rot, normalize_angle

from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class Khr3(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        # normalization
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["goalpos"] = self.cfg["env"]["learn"]["goalposRewardScale"]
        self.rew_scales["crouching"] = self.cfg["env"]["learn"]["crouchingRewardScale"]
        self.rew_scales["base"] = self.cfg["env"]["learn"]["baseRewardScale"]
        self.rew_scales["up"] = self.cfg["env"]["learn"]["upRewardScale"]
        self.rew_scales["raisebody"] = self.cfg["env"]["learn"]["raisebodyRewardScale"]
        self.rew_scales["footup"] = self.cfg["env"]["learn"]["footupRewardScale"]
        self.rew_scales["footdown"] = self.cfg["env"]["learn"]["footdownCostScale"]
        self.rew_scales["legbend"] = self.cfg["env"]["learn"]["legBendRewardScale"]
        self.rew_scales["stand"] = self.cfg["env"]["learn"]["standRewardScale"]
        self.rew_scales["separate"] = self.cfg["env"]["learn"]["separateCostScale"]
        self.rew_scales["b2hError"] = self.cfg["env"]["learn"]["base2headErrorCostScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueCostScale"]
        self.rew_scales["actionsVel"] = self.cfg["env"]["learn"]["actionsVelCostScale"]
        self.rew_scales["actionsAcc"] = self.cfg["env"]["learn"]["actionsAccCostScale"]
        self.rew_scales["death"] = self.cfg["env"]["learn"]["deathCostScale"]


        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint angles
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # action scaling config
        self.named_center_joint_angles = self.cfg["env"]["centerJointAngles"]
        self.named_joint_scales = self.cfg["env"]["jointScales"]

        # goal dof position
        self.named_standard_joint_angles = self.cfg["env"]["goal"]["standardJointAngles"]
        self.named_crouching_joint_angles = self.cfg["env"]["goal"]["crouchingJointAngles"]

        self.cfg["env"]["numObservations"] = 49
        self.cfg["env"]["numActions"] = 22

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        # camera setting
        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0
        self.initial_root_states[0, 3:7]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.center_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.standard_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False) # for calc reward
        self.crouching_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False) # for calc reward

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            if name in self.named_default_joint_angles.keys():
                self.default_dof_pos[:, i] = self.named_default_joint_angles[name]
            if name in self.named_center_joint_angles.keys():
                self.center_dof_pos[:, i] = self.named_center_joint_angles[name]
            if name in self.named_standard_joint_angles.keys():
                self.standard_dof_pos[:, i] = self.named_standard_joint_angles[name]
            if name in self.named_crouching_joint_angles.keys():
                self.crouching_dof_pos[:, i] = self.named_crouching_joint_angles[name]


        # initialize some data used later on
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # get body states
        self.head_states = []
        self.l_foot_states = []
        self.r_foot_states = []
        for j in range(self.num_envs):
            env_ptr = j * self.num_bodies
            self.head_states.append(self.rigid_body_states[env_ptr + self.head_idx, :])
            self.l_foot_states.append(self.rigid_body_states[env_ptr + self.l_foot_idx, :])
            self.r_foot_states.append(self.rigid_body_states[env_ptr + self.r_foot_idx, :])
        self.head_states = torch.stack(self.head_states)
        self.l_foot_states = torch.stack(self.l_foot_states)
        self.r_foot_states = torch.stack(self.r_foot_states)

        # get dof pos states
        self.dof_pos_states = []
        for k in range(self.num_envs):
            env_ptr = k * self.num_dof
            self.dof_pos_states.append(self.dof_state[env_ptr:env_ptr + self.num_dof, 0])
        self.dof_pos_states = torch.stack(self.dof_pos_states)
        # print("dof_pos",self.dof_pos_states.size())


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        # add Khr3 asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/khr3_gym/xacro/khr3_gym.urdf"

        # Asset Options
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.armature = 0.007
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        khr3_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(khr3_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(khr3_asset)

        # set Dof properties
        dof_props = self.gym.get_asset_dof_properties(khr3_asset)
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
            dof_props['armature'][i] = 0.01
            # set dof limits
            self.dof_limits_lower.append(dof_props['lower'][i])
            self.dof_limits_upper.append(dof_props['upper'][i])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # start_pose.r = gymapi.Quat(*self.base_init_state[3:7])
        start_pose.r = gymapi.Quat(0.0, 0.7071, 0.0, 0.7071) # face down
        # start_pose.r = gymapi.Quat(0.0, -0.7071, 0.0, 0.7071) # face up
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        body_names = self.gym.get_asset_dof_properties(khr3_asset)
        self.dof_names = self.gym.get_asset_dof_names(khr3_asset)

        self.khr3_handles = []
        self.envs = []
        self.head_states = []
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for j in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )
            khr3_handle = self.gym.create_actor(env_ptr, khr3_asset, start_pose, "khr3", j, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, khr3_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, khr3_handle)

            self.envs.append(env_ptr)
            self.khr3_handles.append(khr3_handle)

        # find rigid body indices
        body_idx = self.gym.get_actor_rigid_body_names(self.envs[0], self.khr3_handles[0])
        print(body_idx)
        self.head_idx = 0
        self.head_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.khr3_handles[0], "c_head_a")
        print("head_idx", self.head_idx)
        self.l_foot_idx = 0
        self.l_foot_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.khr3_handles[0], "l_foot")
        print("l_foot_idx", self.l_foot_idx)
        self.r_foot_idx = 0
        self.r_foot_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.khr3_handles[0], "r_foot")
        print("r_foot_idx", self.r_foot_idx)
        self.l_arm_idx = 0
        self.l_arm_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.khr3_handles[0], "l_lowerarm")
        print("l_arm_idx", self.l_arm_idx)
        self.r_arm_idx = 0
        self.r_arm_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.khr3_handles[0], "r_lowerarm")
        print("r_arm_idx", self.r_arm_idx)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = compute_khr3_reward(
            # Tensor
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.torques,
            self.dof_pos_states,
            self.actions,
            self.standard_dof_pos,
            self.crouching_dof_pos,
            self.root_states,
            self.head_states,
            self.l_foot_states,
            self.r_foot_states,
            # Dict
            self.named_joint_scales,
            self.rew_scales,
            # other
            self.max_episode_length
        )
        # # For logdata test
        # if self.reset_buf[0]:
        #     print("@log", 0.00)
        # else:
        #     print("@log", self.obs_buf[0, 0].item()) # dt=0.02s

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.time = self.gym.get_sim_time(self.sim)
        # print("time", self.time)

        self.obs_buf[:], self.head_states[:], \
        self.l_foot_states[:], self.r_foot_states[:], self.dof_pos_states[:] = compute_khr3_observations(
            # Tensor
            self.obs_buf,
            self.root_states,
            self.rigid_body_states,
            self.dof_state,
            self.basis_vec1,
            self.actions,
            # float
            self.dt,
            # int
            self.num_bodies,
            self.num_dof,
            self.head_idx,
            self.l_foot_idx,
            self.r_foot_idx)


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            if name in self.named_joint_scales.keys():
                self.actions[:, i] *= self.named_joint_scales[name]
        targets = (self.action_scale * self.actions) + self.center_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.default_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_khr3_reward(
    # tensors
    obs_buf,
    reset_buf,
    progress_buf,
    torques,
    dof_pos_states,
    actions,
    standard_dof_pos,
    crouching_dof_pos,
    root_states,
    head_states,
    l_foot_states,
    r_foot_states,
    # Dict
    joint_scales,
    rew_scales,
    # other
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # prepare quantities
    base_height = obs_buf[:, 0]
    base_up_prj = obs_buf[:, 1]
    head_height = head_states[:, 2]
    l_foot_up_prj = obs_buf[:, 3]
    r_foot_up_prj = obs_buf[:, 4]
    actions_buf = obs_buf[:, 5:27]
    actions_buf_buf = obs_buf[:, 27:49]

    # === base to foot xy error penalty ===
    base2lfoot_error = torch.norm(root_states[:, 0:2] - l_foot_states[:, 0:2], dim=1)
    base2rfoot_error = torch.norm(root_states[:, 0:2] - r_foot_states[:, 0:2], dim=1)
    separate_val = torch.min(torch.tensor(1.0),torch.exp(50 * ((base2lfoot_error + base2rfoot_error)-0.4)))
    doesSeparate = (base2lfoot_error + base2rfoot_error) > 0.3
    separate_cost = torch.where(doesSeparate, separate_val * rew_scales["separate"], 0.0)

    # === leg bend reward ===
    HIP_P_MAX_BEND = 3.84
    act_l_hip_p = -1.0 * actions[:, 12] / joint_scales["l_hipjoint_pitch"] # Rescaled (-1.0 ~ 1.0)
    act_l_knee_p = actions[:, 13] / joint_scales["l_knee_pitch"]
    act_r_hip_p = -1.0 * actions[:, 18] / joint_scales["r_hipjoint_pitch"]
    act_r_knee_p = actions[:, 19] / joint_scales["r_knee_pitch"]
    leg_bend_val = torch.min(torch.tensor(1.0),torch.exp(10*((act_l_hip_p + act_r_hip_p + act_l_knee_p + act_r_knee_p)-4.)))
    # leg_bend_val = torch.min(torch.tensor(1.0),torch.exp(50*((act_l_hip_p + act_r_hip_p) - HIP_P_MAX_BEND)))
    rew_legBend = torch.where(doesSeparate, leg_bend_val * rew_scales["legbend"], 0.)

    # === reward for base height ===
    base_val = base_height
    isBaseUp = base_up_prj > 0.7
    rew_base = torch.where((~doesSeparate), base_val * rew_scales["base"], -(0.001 * rew_scales["base"]))

    # === reward for raise body ===
    max_err_b2h = 0.099 # KHR's height gap(base2head)
    raise_body_val = torch.min(torch.tensor(1.), torch.exp(5 * ((head_height - base_height) - max_err_b2h)))
    raise_body_val_inv = torch.min(torch.tensor(1.), torch.exp(-5 * (head_height - base_height)))
    doesRaisebody = head_height > base_height
    rew_raisebody = torch.where(doesRaisebody & (~doesSeparate), raise_body_val * rew_scales["raisebody"], raise_body_val_inv * (-0.05))

    # === reward for foot_link upright ===
    feet_up_val = torch.min(torch.tensor(1.), torch.exp(5 * ((l_foot_up_prj + r_foot_up_prj)/2. - 1.)))
    isFeetup = (l_foot_up_prj > 0.7) & (r_foot_up_prj > 0.7)
    rew_feet_up = torch.where((~doesSeparate), feet_up_val * rew_scales["footup"], 0.)

    # === penalty for foot_link wrong posture ===
    feet_down_val = torch.clip(torch.exp(-5 * (l_foot_up_prj + r_foot_up_prj)/2.), 0., 1.)
    isFeetwrong = (l_foot_up_prj < 0.3) | (r_foot_up_prj < 0.3)
    feet_down_cost = torch.where(isFeetwrong, feet_down_val * rew_scales["footdown"], 0.)

    # === reward for base_link upright ===
    up_val = torch.min(torch.tensor(1.0), torch.exp(2 * (base_up_prj - 1.)))
    isUp = base_up_prj > 0.7
    rew_up = torch.where(isUp & (~doesSeparate), up_val * rew_scales["up"], 0.0)

    # === reward for crouching ===
    crouching_err = dof_pos_states[:, 10:22] - crouching_dof_pos[:, 10:22]
    crouching_val = 1.0 / ((5 * torch.norm(torch.arcsin(torch.min(torch.tensor(1.0), torch.norm(crouching_err,p=2,dim=1))))) + 0.1)
    rew_crouching = torch.where((base_up_prj>0.7) & (isFeetup), crouching_val * rew_scales["crouching"], 0.0)

    # === reward for stand ===
    max_base_height = 0.26
    doesStand = (base_up_prj>0.85) & (isFeetup) & (~doesSeparate)
    stand_val = torch.min(torch.tensor(1.), torch.exp(50 * (base_height - max_base_height)))
    rew_stand = torch.where(doesStand, stand_val * rew_scales["stand"], 0.)

    # === reward for goal ===
    goal_err = dof_pos_states[:, 10:22] - standard_dof_pos[:, 10:22]
    goal_val = 1.0 / ((5 * torch.norm(torch.arcsin(torch.min(torch.tensor(1.0), torch.norm(goal_err,p=2,dim=1))))) + 0.1)
    rew_goal = torch.where((base_up_prj>0.93) & (isFeetup) & (~doesSeparate), \
                            goal_val * rew_scales["goalpos"], 0.0)

    # === cost for legBend ===
    leg_bend_val = torch.min(torch.tensor(1.0),torch.exp(10*((act_l_hip_p + act_r_hip_p + act_l_knee_p + act_r_knee_p)-4.)))
    legBend_cost = torch.where((base_up_prj > 0.93) & (isFeetup) & (~doesSeparate), -0.001 * leg_bend_val, 0.)

    # === cost for base2head xy error ===
    base2head_error = torch.norm(root_states[:, 0:2] - head_states[:, 0:2], dim=1)
    b2h_error_val = torch.min(torch.tensor(1.0), torch.exp(5 * ((base2head_error/max_err_b2h) - 1.)))
    b2h_error_cost = torch.where(doesRaisebody & (~doesSeparate), b2h_error_val * rew_scales["b2hError"], 0.0)

    # === torque cost ===
    torque_cost = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # === actions cost ===
    actions_vel = actions - actions_buf
    actions_acc = actions_vel - (actions_buf - actions_buf_buf)
    actions_vel_cost = torch.sum(torch.square(actions_vel), dim=-1) * rew_scales["actionsVel"]
    actions_acc_cost = torch.sum(torch.square(actions_acc), dim=-1) * rew_scales["actionsAcc"]

    # total reward
    total_reward =  rew_goal \
                    + rew_crouching \
                    + rew_base \
                    + rew_stand \
                    + rew_up \
                    + rew_raisebody \
                    + rew_feet_up \
                    + rew_legBend \
                    + rew_stand \
                    + feet_down_cost \
                    + separate_cost \
                    + torque_cost \
                    + actions_vel_cost \
                    + actions_acc_cost 

    # reset agents
    over_flip = base_up_prj < -0.7
    feet_flip =  ((~doesSeparate) & (l_foot_up_prj < -0.7)) | ((~doesSeparate) & (r_foot_up_prj < -0.7))
    # over_bend_legs = ((dof_pos_states[:, 12] + dof_pos_states[:, 13]) > np.pi) | ((dof_pos_states[:, 18] + dof_pos_states[:, 19]) > np.pi)
    over_bend_legs = (dof_pos_states[:, 13] + dof_pos_states[:, 14] >= 3.0) | (dof_pos_states[:, 19] + dof_pos_states[:, 20] >= 3.0)
    time_out = progress_buf >= max_episode_length - 1

    reset = time_out | over_flip | over_bend_legs | feet_flip

    # adjust reward
    total_reward = torch.where(over_flip | over_bend_legs | feet_flip, torch.ones_like(total_reward) * rew_scales["death"], total_reward)
    total_reward = torch.clip(total_reward, 0., None)

    return total_reward, reset


@torch.jit.script
def compute_khr3_observations(
    # Tensor
    obs_buf,
    root_states,
    rigid_body_states,
    dof_state,
    basis_vec1,
    actions,
    # float
    dt,
    # int
    num_bodies,
    num_dof,
    head_idx,
    l_foot_idx,
    r_foot_idx
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, int, int, int, int, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    
    # actions buf
    actions_buf = obs_buf[:, 5:27]
    # base states
    base_pos = root_states[:, 0:3]
    base_quat = root_states[:, 3:7]
    num_envs = base_quat.shape[0]
    # base proj
    base_up_vec = get_basis_vector(base_quat, basis_vec1).view(num_envs, 3)
    base_up_prj = base_up_vec[:, 2]

    # calc body states
    head_states = []
    l_foot_states = []
    r_foot_states = []
    for j in range(num_envs):
        env_ptr = j * num_bodies
        head_states.append(rigid_body_states[env_ptr + head_idx, :])
        l_foot_states.append(rigid_body_states[env_ptr + l_foot_idx, :])
        r_foot_states.append(rigid_body_states[env_ptr + r_foot_idx, :])
    head_states = torch.stack(head_states)
    l_foot_states = torch.stack(l_foot_states)
    r_foot_states = torch.stack(r_foot_states)
    # calc dof states
    dof_pos_states = []
    for k in range(num_envs):
        env_ptr = k * num_dof
        dof_pos_states.append(dof_state[env_ptr:env_ptr + num_dof, 0])
    dof_pos_states = torch.stack(dof_pos_states)
    # body proj
    head_up_vec = get_basis_vector(head_states[:, 3:7], basis_vec1).view(num_envs, 3)
    head_up_prj = head_up_vec[:, 2]
    l_foot_up_vec = get_basis_vector(l_foot_states[:, 3:7], basis_vec1).view(num_envs, 3)
    l_foot_up_prj = l_foot_up_vec[:, 2]
    r_foot_up_vec = get_basis_vector(r_foot_states[:, 3:7], basis_vec1).view(num_envs, 3)
    r_foot_up_prj = r_foot_up_vec[:, 2]
    
    obs = torch.cat((base_pos[:, 2].view(-1, 1),
                     base_up_prj.unsqueeze(-1),
                     head_up_prj.unsqueeze(-1),
                     l_foot_up_prj.unsqueeze(-1),
                     r_foot_up_prj.unsqueeze(-1),
                     actions,
                     actions_buf
                     ), dim=-1)

    return obs, head_states, l_foot_states, r_foot_states, dof_pos_states