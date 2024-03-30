import gym
import numpy as np
import math
import pybullet as p
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
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-0.5, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-6, -6, -1, -1, -5, -5, -6, -6], dtype=np.float32),
            high=np.array([6, 6, 1, 1, 5, 5, 6, 6], dtype=np.float32))  #Figure out the bounds
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.obstacles = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self.update_counter = 0

    def step(self, action):
        # # Feed action to the car and get observation of car's state
        # self.car.apply_action(action)
        # p.stepSimulation()
        # car_ob = self.car.get_observation()

        # # Change and update Reward so you're maximizing -distance
        
        # # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
        #                           (car_ob[1] - self.goal[1]) ** 2))
        # # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        # reward = self.prev_dist_to_goal - dist_to_goal
        # self.prev_dist_to_goal = dist_to_goal

        # # Done by running off boundaries
        # if (car_ob[0] >= 6 or car_ob[0] <= -6 or
        #         car_ob[1] >= 6 or car_ob[1] <= -6):
        #     self.done = True
        # # Done by reaching goal
        # elif dist_to_goal < 1:
        #     self.done = True
        #     reward = 50

        # ob = np.array(car_ob + self.goal, dtype=np.float32)
        # return ob, reward, self.done, dict()
    
        #Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Change and update Reward so you're maximizing -distance
        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        dist_to_obstacle = math.sqrt(((car_ob[0] - self.obstacles[0]) ** 2 +
                                  (car_ob[1] - self.obstacles[1]) ** 2))
        
        reward = 50/dist_to_goal  # with a scale factor

        # Done by running off boundaries
        if (car_ob[0] >= 6 or car_ob[0] <= -6 or
                car_ob[1] >= 6 or car_ob[1] <= -6):
            self.done = True
        elif dist_to_obstacle < 1: #Done by hitting an obstacle 
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

        # Set the goal to a random target
        x = (np.random.uniform(5, 5.5) if np.random.randint(2) else
             np.random.uniform(-5, -5.5))
        y = (np.random.uniform(5, 5.5) if np.random.randint(2) else
             np.random.uniform(-5, -5.5))
        self.goal = (x, y)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)
        
        #set the first obstacles
        x1 = (np.random.uniform(1, 5.5) if np.random.randint(2) else
             np.random.uniform(-1, -5.5))
        y1 = (np.random.uniform(1, 5.5) if np.random.randint(2) else
             np.random.uniform(-1, -5.5))
        self.obstacles = (x1,y1)
        self.done = False
        
        #Visual element of the obstacle
        Obstacle(self.client, self.obstacles)
        
        #set the second obstacles
        x2 = (np.random.uniform(2, 5.5) if np.random.randint(2) else
             np.random.uniform(-2, -5.5))
        y2 = (np.random.uniform(2, 5.5) if np.random.randint(2) else
             np.random.uniform(-2, -5.5))
        self.obstacles = (x2,y2)
        self.done = False
        
        #Visual element of the obstacle
        Obstacle(self.client, self.obstacles)
        
        #set the second obstacles
        x2 = (np.random.uniform(3, 5.5) if np.random.randint(2) else
             np.random.uniform(-3, -5.5))
        y2 = (np.random.uniform(3, 5.5) if np.random.randint(2) else
             np.random.uniform(-3, -5.5))
        self.obstacles = (x2,y2)
        self.done = False
        
        #Visual element of the obstacle
        Obstacle(self.client, self.obstacles)
        
        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        return np.array(car_ob + self.goal, dtype=np.float32)
    
    # def render(self, mode='human'):
    #     if self.rendered_img is None:
    #         self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

    #     # Base information
    #     car_id, client_id = self.car.get_ids()
    #     proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
    #                                                nearVal=0.01, farVal=100)
    #     pos, ori = [list(l) for l in
    #                 p.getBasePositionAndOrientation(car_id, client_id)]
    #     pos[2] = 0.2

    #     # Rotate camera direction
    #     rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    #     camera_vec = np.matmul(rot_mat, [1, 0, 0])
    #     up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
    #     view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

    #     # Display image
    #     frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
    #     frame = np.reshape(frame, (100, 100, 4))
    #     self.rendered_img.set_data(frame)
    #     plt.draw()
    #     plt.pause(.00001)

    # def render(self, mode='human'):
    #     if self.update_counter % 5 != 0:
    #         self.update_counter += 1
    #         return
    
    #     camera_target_position = [0, 0, 0]  # Adjust based on your scene
    #     camera_distance = 10  # Adjust based on the scale of your environment
    #     camera_pitch = -90  # Pointing straight down
    #     camera_yaw = 0  # Northward view, adjust as necessary
    #     camera_roll = 0  # Usually fine as 0 for bird's eye view
    #     fov = 70  # Wider FOV for wide-angle view
    #     aspect = 1  # Aspect ratio, change if necessary
    #     near_val = 0.1  # Near plane
    #     far_val = 100  # Far plane

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