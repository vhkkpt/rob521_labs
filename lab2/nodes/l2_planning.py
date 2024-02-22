#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag


def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.3 #m/s (Feel free to change!)
        self.rot_vel_max = 1.6 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 2.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        scale = min(1000 / self.map_shape[1], 1000 / self.map_shape[0])
        window_size = (int(self.map_shape[1] * scale), int(self.map_shape[0] * scale))
        self.window = pygame_utils.PygameWindow(
            "Path Planner", window_size, self.occupancy_map.T.shape, self.map_settings_dict, self.goal_point, self.stopping_dist, map_filename)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        # print("TO DO: Sample point to drive towards")
        
        def sample_unif():
            return np.array([
                [np.random.uniform(self.bounds[0, 0], self.bounds[0, 1])],
                [np.random.uniform(self.bounds[1, 0], self.bounds[1, 1])]
            ])

        def sample_non_unif():
            while True:
                x = sample_unif()
                sigma = 0.4
                y = np.random.multivariate_normal(x.flatten(), [[sigma**2, 0], [0, sigma**2]])[:, np.newaxis]
                if not (self.bounds[0, 0] <= y[0, 0] < self.bounds[0, 1] and self.bounds[1, 0] <= y[1, 0] < self.bounds[1, 1]):
                    continue
                x_cell = self.point_to_cell(x).flatten()
                y_cell = self.point_to_cell(y).flatten()
                if (self.occupancy_map[x_cell[0], x_cell[1]] == 1) ^ (self.occupancy_map[y_cell[0], y_cell[1]] == 1):
                    return x if self.occupancy_map[x_cell[0], x_cell[1]] == 1 else y

        k = np.random.uniform(0, 1)

        if k < 0.01:
            return self.goal_point

        if k < 0.55:
            return sample_non_unif()

        return sample_unif()

        # return np.array([
        #     [100],
        #     [100]
        # ])
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        # print("TO DO: Check that nodes are not duplicates")

        duplicated = False

        for node in self.nodes:
            if np.linalg.norm(node.point[:2, :] - point) < self.map_settings_dict["resolution"]:
                duplicated = True
                break

        return duplicated
    
    def get_dist_angle_diff(self, node_i, point):
        dist = np.linalg.norm(point - node_i[:2, :])
        angle = np.arctan2(point[1, 0] - node_i[1, 0], point[0, 0] - node_i[0, 0])
        angle = np.mod(angle - node_i[2, 0], 2 * np.pi)
        angle = angle - 2 * np.pi if angle > np.pi else angle

        return dist, angle

    def closest_node(self, point):
        #Returns the index of the closest node
        # print("TO DO: Implement a method to get the closest node to a sapled point")

        min_cost = np.inf
        min_idx = -1

        for i, node in enumerate(self.nodes):
            # dist, angle = self.get_dist_angle_diff(node.point, point)
            # cost = dist + abs(angle)
            cost = np.sum((node.point[:2, :] - point) ** 2)
            if cost < min_cost:
                min_cost, min_idx = cost, i

        return min_idx
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")
        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(node_i, vel, rot_vel)
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        # dist, angle = self.get_dist_angle_diff(node_i, point_s)

        # if dist < self.map_settings_dict["resolution"]:
        #     v = 0.0
        #     w = 0.0
        # if abs(angle) < 0.2:
        #     v = self.vel_max
        #     w = 0.0
        # else:
        #     v = 0.0
        #     w = self.rot_vel_max if angle > 0 else -self.rot_vel_max

        # v = self.vel_max / 2 # min(max(dist, -self.vel_max), self.vel_max)
        # w = min(max(angle, -self.rot_vel_max), self.rot_vel_max)

        # return v, w

        min_cost = np.inf
        opt_v = 0
        opt_w = 0

        # v_values = np.arange(-self.vel_max, self.vel_max, 0.2)
        # w_values = np.arange(-self.rot_vel_max, self.rot_vel_max, 0.2)
        v_values = np.linspace(-self.vel_max, self.vel_max, 4)
        w_values = np.linspace(-self.rot_vel_max, self.rot_vel_max, 5)

        for v in v_values:
            if v == 0:
                continue
            for w in w_values:
                # newq = self.circular_rollout(node_i, v, w, self.timestep)
                traj = self.trajectory_rollout(node_i, v, w)
                if self.check_trajectory_collision(traj[:2, :]):
                    continue
                cost = np.linalg.norm(traj[:2, -1:] - point_s)
                if cost < min_cost:
                    min_cost, opt_v, opt_w = cost, v, w

        return opt_v, opt_w
    
    def trajectory_rollout(self, node_i, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        dt = self.timestep / self.num_substeps
        traj = np.zeros((3, self.num_substeps))
        q = node_i
        p = np.array([vel, rot_vel])[:, np.newaxis]

        for i in range(self.num_substeps):
            G_q = np.array([
                [np.cos(q[2, 0]), 0],
                [np.sin(q[2, 0]), 0],
                [0, 1]
            ])
            q_dot = G_q @ p
            q = q + q_dot * dt
            traj[:, i] = q[:, 0]

        return traj

    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        origin = np.array([self.bounds[0, 0], self.bounds[1, 0]])
        newpoints = point - origin[:, np.newaxis]
        cells = np.zeros_like(newpoints)
        resol = self.map_settings_dict["resolution"]
        cells[0] = np.floor((self.bounds[1, 1] - self.bounds[1, 0] - newpoints[1]) / resol)
        cells[1] = np.floor(newpoints[0] / resol)

        return cells.astype(np.int64)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")
        cells = self.point_to_cell(point=points)
        res_rr = []
        res_cc = []
        
        for c in cells.T:
            rr, cc = disk((c[0], c[1]), self.robot_radius / self.map_settings_dict["resolution"], shape=self.map_shape)
            res_rr.extend(rr)
            res_cc.extend(cc)

        return res_rr, res_cc
    
    def check_trajectory_collision(self, points):
        rr, cc = self.points_to_robot_circle(points)
        if len(rr) == 0:
            return True
        return not np.all(self.occupancy_map[rr, cc])

    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        path_found = False
        while True: #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            if self.check_if_duplicate(point):
                continue

            self.window.add_point(point[:, 0].copy(), color = (255, 0, 0))

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            # print(trajectory_o)

            #Check for collisions
            # print("TO DO: Check for collisions and add safe points to list of nodes.")            
            if self.check_trajectory_collision(trajectory_o[:2, :]):
                continue

            newpoint = trajectory_o[:, -1:]

            self.window.add_point(newpoint[:-1, 0].copy())

            idx = len(self.nodes)
            self.nodes.append(Node(newpoint, closest_node_id, 0.0))
            self.nodes[closest_node_id].children_ids.append(idx)
            
            #Check if goal has been reached
            # print("TO DO: Check if at goal point.")
            if np.linalg.norm(newpoint[:2, :] - self.goal_point) < self.stopping_dist:
                path_found = True
                break

        print(path_found)
        return self.nodes

    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    # #Set map information
    # map_filename = "willowgarageworld_05res.png" #"simple_map.png"
    # map_setings_filename = "willowgarageworld_05res.yaml"

    # #robot information
    # goal_point = np.array([[42], [-44]]) #m
    # stopping_dist = 0.5 #m

    #Set map information
    map_filename = "myhal.png"
    map_setings_filename = "myhal.yaml"

    #robot information
    goal_point = np.array([[7], [0]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning() #path_planner.rrt_star_planning()
    path = path_planner.recover_path()
    node_path_metric = np.hstack(path)

    for i in range(1, len(path)):
        path_planner.window.add_line(path[i - 1][:2, 0].copy(), path[i][:2, 0].copy(), width = 3, color = (0, 0, 255))

    input("Press enter to exit...")

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
