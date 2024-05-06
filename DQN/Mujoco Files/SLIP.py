import numpy as np

from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box

class SLIPEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 500,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "SLIP_bipedal.xml", 2, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

        self.R2D = 180.0 / np.pi
        self.backflip = False

    def step(self, a):
        angle_previous = self.sim.data.qpos[2]
        self.do_simulation(np.concatenate((a,a)), self.frame_skip)
        #self.do_simulation(a, self.frame_skip)
        x_pos, height, angle_after = self.sim.data.qpos[0:3]

        # alive_bonus = 10.0
        # reward = 0.05 * (angle_after - angle_previous) * self.R2D / self.dt
        # reward += alive_bonus
        # reward -= 50 * np.square(a)
        # reward -= 8.0 * self.sim.data.qpos[0]
        # s = self.state_vector()
        # terminated = not (
        #     np.isfinite(s).all()
        #     and (height > -2.38)
        #     and (np.abs(s[[0,1,3,4,5,6]]) < 60).all()
        #     and (abs(x_pos) < 30.0)
        # )
        ############## Currently Used ###############

        alive_bonus = 10.
        reward = 0.
        if not self.backflip: reward += 25 * (angle_after - angle_previous) / self.dt
        else: reward -= 5 * abs(angle_after - angle_previous) / self.dt
        reward += alive_bonus
        #reward -= 12 * np.square(a)
        s = self.state_vector()
        terminated = not (
            np.isfinite(s).all()
            and (height > -2.2)
            and (np.abs(s[[0,1]]) < 25).all()
        )

        #print(self.backflip)

        if terminated: reward -= 10000

        if(self.sim.data.ncon): 
            if (abs(angle_after) % 2*np.pi < 0.27): reward += 500
        #     else: reward -= 200
        
        ##################################################################
        # s = self.state_vector()
        # terminated = not (
        #     np.isfinite(s).all()
        #     and (height > -2.38)
        #     and (np.abs(s) < 10).all()
        # )
        # ############################################################
        # #############  Reward Function for Balancing ###############
        # ############################################################
        # alive_bonus = 10.0
        # reward = alive_bonus
        # if abs(angle_after) < 0.5236: reward += 125
        # else: reward -= 500
        # reward -= (15.0 * abs(x_pos))
        # if(terminated): reward -= 1000
        ############### current ######################
        # alive_bonus = 20.
        # reward = 0.
        # reward += alive_bonus
        # #reward -= 15.0 * np.square(a)
        # #reward -= 5.0 * abs(x_pos)
        # s = self.state_vector()
        # terminated = not (
        #     np.isfinite(s).all()
        #     and (height > -2.38)
        #     and (np.abs(s[[0,1]]) < 25).all()
        #     and (abs(angle_after) < 0.6)
        # )

        # if terminated: reward -= 10000

        # if(self.sim.data.ncon): 
        #     if (abs(angle_after) % 2*np.pi < 0.27): reward += 300
        ####################################################################

        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, False, {}

    def _get_obs(self):
        # return np.concatenate(
        #     [self.sim.data.qpos.flat[0:3], np.clip(self.sim.data.qvel.flat[0:3], -15, 15)]
        # )
        if (not self.backflip) and (abs(self.sim.data.qpos.flat[2]) > 6.28): self.backflip = True
        return np.concatenate(
            [self.sim.data.qpos.flat[0:3] + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq - 3), 
            np.clip(self.sim.data.qvel.flat[0:3], -30, 30) + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv - 3), 
            [self.backflip], 
            [self.sim.data.ncon]]
            # [self.sim.data.qpos.flat[0:3] + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq - 1), 
            # np.clip(self.sim.data.qvel.flat[0:3], -30, 30) + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv - 1), 
            # [self.backflip], 
            # [self.sim.data.ncon]]
        )
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        # qpos = self.init_qpos
        # qvel = self.init_qvel
        self.backflip = False
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] = 3
        self.viewer.cam.elevation = -20