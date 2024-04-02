import gym
import numpy as np
import math
import cv2
import pybullet as p
import imageio
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
from simple_driving.resources.obstacle import Obstacle
import matplotlib.pyplot as plt


# 1. Fix obstacles
# 2. Train 1000 atleast 
# 3. Fine Tune policy 
# 4. Learning curve
# 5. Reward update


class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-0.5, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            #xy position of the car, unit xy orientation of the car, xy velocity of the car, and xy position of a target
            low=np.array([-6, -6, -1, -1, -5, -5, -6, -6], dtype=np.float32),
            high=np.array([6, 6, 1, 1, 5, 5, 6, 6], dtype=np.float32))  
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.obstacles = []
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self.prev_wheel_position = 0
        self.update_counter = 0

    def step(self, action):
                #Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Change and update Reward so you're maximizing -distance
        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        dist_to_obstacles = [math.sqrt(((car_ob[0] - obs[0]) ** 2 +
                                        (car_ob[1] - obs[1]) ** 2))
                             for obs in self.obstacles]  # Calculate distance to each obstacle
        
        # Initialize reward
        reward = 0
        
        #If distance > 1, reward is 100/dist_to_goal
        if dist_to_goal < 1.5:
            reward = 200/dist_to_goal  # with a scale factor
        elif dist_to_goal < 2:
            reward = 150/dist_to_goal
        elif dist_to_goal < 3:
            reward = 100/dist_to_goal   
        elif dist_to_goal < 4:
            reward = 50/dist_to_goal
        else :
            reward = 10/dist_to_goal
        
        for dist_to_obstacle in dist_to_obstacles:
            if dist_to_obstacle < 0.7:
                reward -= 250
        
        #reward for choosing the shortest path
        if dist_to_goal < self.prev_dist_to_goal:
            reward += 50
        
        # Done by running off boundaries
        if (car_ob[0] >= 6 or car_ob[0] <= -6 or
                car_ob[1] >= 6 or car_ob[1] <= -6):
            self.done = True
        elif any(dist < 0.5 for dist in dist_to_obstacles):  # Done if any obstacle is hit
            self.done = True
            reward = -500
        # Done by reaching goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 1000

        ob = np.array(car_ob + self.goal, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self.client)
        self.car = Car(self.client)

        #Set the goal to a random target
        # x = (np.random.uniform(5, 5.5) if np.random.randint(2) else
        #      np.random.uniform(-5, -5.5))
        # y = (np.random.uniform(5, 5.5) if np.random.randint(2) else
        #      np.random.uniform(-5, -5.5))
        self.goal = (3, 5)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)
        
        self.obstacles = [(0,3), (2.5,1), (1,2.2)]  # Set obstacle positions
        self.done = False
        for obstacle in self.obstacles:
            Obstacle(self.client, obstacle)  # Create each obstacle
        
        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        return np.array(car_ob + self.goal, dtype=np.float32)
         
    # def render(self, mode='human'): 
    #     #decrease the rendering frequency 
    #     if self.update_counter % 8 != 0: 
    #         self.update_counter += 1 
    #         return 
     
    #     camera_target_position = [0, 0, 0]  # Adjust based on your scene 
    #     camera_distance = 10  # Adjust based on the scale of your environment 
    #     camera_pitch = -90  # Pointing straight down 
    #     camera_yaw = 0  # Northward view, adjust as necessary 
    #     camera_roll = 0  # Usually fine as 0 for bird's eye view 
    #     fov = 70  # Wider FOV for wide-angle view 
    #     aspect = 1  # Aspect ratio, change if necessary
    #     near_val = 1  # Near plane
    #     far_val = 11  # Far plane

    #     view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target_position, camera_distance, camera_yaw, camera_pitch, camera_roll, 2)
    #     projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_val, far_val)

    #     # Obtain the image from the camera view
    #     width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    #         width=512, 
    #         height=512, 
    #         viewMatrix=view_matrix,
    #         projectionMatrix=projection_matrix,
    #         renderer=p.ER_BULLET_HARDWARE_OPENGL
    #     )

    #     if mode == 'human':
    #         # Convert the PyBullet image (which is BGRA) to the correct format
    #         rgb_array = rgb_img[:, :, :3]
    #         # plt.imshow or other methods to display the image if needed
    #         if self.rendered_img is None:
    #             self.rendered_img = plt.imshow(rgb_array)
    #         else:
    #             self.rendered_img.set_data(rgb_array)
    #         plt.draw()
    #         plt.pause(0.001)  # Small pause to update the plots
            
    #     self.update_counter += 1
    
    def render(self, mode='human'): 
        #decrease the rendering frequency 
        if self.update_counter % 8 != 0: 
            self.update_counter += 1 
            return 
     
        camera_target_position = [0, 0, 0]  # Adjust based on your scene 
        camera_distance = 10  # Adjust based on the scale of your environment 
        camera_pitch = -90  # Pointing straight down 
        camera_yaw = 0  # Northward view, adjust as necessary 
        camera_roll = 0  # Usually fine as 0 for bird's eye view 
        fov = 70  # Wider FOV for wide-angle view 
        aspect = 1  # Aspect ratio, change if necessary
        near_val = 1  # Near plane
        far_val = 11  # Far plane

        view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target_position, camera_distance, camera_yaw, camera_pitch, camera_roll, 2)
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_val, far_val)

        # Obtain the image from the camera view
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=512, 
            height=512, 
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        if mode == 'human':
            # Convert the PyBullet image (which is BGRA) to the correct format
            rgb_array = rgb_img[:, :, :3]
            # plt.imshow or other methods to display the image if needed
            if self.rendered_img is None:
                self.rendered_img = plt.imshow(rgb_array)
                
            else:
                self.rendered_img.set_data(rgb_array)
            plt.draw()
            plt.pause(0.001)  # Small pause to update the plots
            
        self.update_counter += 1

    def close(self):
        p.disconnect(self.client)