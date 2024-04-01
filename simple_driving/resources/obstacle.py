import pybullet as p
import os


class Obstacle:
    def __init__(self, client, base, size=[0.5, 0.5, 0.5]):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleobstacle.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[base[0] + size[0]/2, base[1] + size[1]/2, size[2]/2],
                   physicsClientId=client)


