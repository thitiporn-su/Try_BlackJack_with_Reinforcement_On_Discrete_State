import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Constants
TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 0.01
MAX_STEPS = 100

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_size[0], state_size[1], state_size[2], action_size))
        self.state_size = (10, 10, 10, 10, 10, 10, 10, 10, 10)  

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size) 
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q  
        
        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

class MotionPlanning:
    def __init__(self):
        # Constants
        g = -9.81  # m/s^2
        m = 0.57  # kg
        Ix = 0.02898  # kgm^2
        Iy = 0.04615  # kgm^2
        Iz = 0.00738  # kgm^2

        # Initialize drone, setpoint, obstacle
        self.drone_position = np.array([0.0, 0.0, 0.0])
        self.setpoint = np.random.uniform(low=[0.0, 0.0, 0.0], high=[10.0, 10.0, 10.0])
        self.obstacle_position = np.random.uniform(low=[0.0, 0.0, 0.0], high=[10.0, 10.0, 10.0])
        self.obstacle_radius = np.random.uniform(0.5, 2.0)

        # Store drone's flight path
        self.flight_path = []

        # Initialize RL agent
        self.agent = QLearningAgent(state_size=(10, 10, 10), action_size=6)
        
        # Set grid size
        self.grid_size = 1

        # Define action space
        self.action_space = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1])
        } 
        
        # Continuous-time system matrices
        self.A = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       
        ])

        self.B = np.array([
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [1/m, 0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   1/Ix, 0,    0   ],
            [0,   0,    1/Iy, 0   ],
            [0,   0,    0,    1/Iz]
        ])
        
        # Discrete-time system matrices
        self.dt = 0.1  # Time step
        self.Ad = np.eye(12) + self.dt * self.A
        self.Bd = self.dt * self.B
        
        # Discrete-time Riccati equation solver
        self.Qd = np.eye(12)
        self.Rd = np.eye(4)
        self.Kd = np.linalg.inv(self.Rd + self.Bd.T @ self.Qd @ self.Bd) @ self.Bd.T @ self.Qd @ self.Ad

        self.x = np.zeros(12)
        self.u = np.zeros(4)

        # Initialize plot for motion planning
        self.fig_mp = plt.figure()
        self.ax_mp = self.fig_mp.add_subplot(111, projection='3d')

        # Set plot labels and title
        self.ax_mp.set_xlabel('X [m]')
        self.ax_mp.set_ylabel('Y [m]')
        self.ax_mp.set_zlabel('Z [m]')
        self.ax_mp.set_title('Motion Planning Trajectory')

        # Initialize time display
        self.time_text = self.ax_mp.text(0.02, 0.02, 0.02, "", transform=self.ax_mp.transAxes)

    def calculate_control_input(self):
        self.u = -self.Kd.dot(self.x - np.concatenate((self.setpoint, np.zeros(9))))

    def visualize_trajectory(self):
        self.ax_mp.scatter(self.drone_position[0], self.drone_position[1], self.drone_position[2], color='red', label='Drone')
        self.ax_mp.scatter(self.setpoint[0], self.setpoint[1], self.setpoint[2], color='green', label='Setpoint')
        self.ax_mp.scatter(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2], color='blue', label='Obstacle')

        if self.flight_path:
            flight_path = np.array(self.flight_path)
            self.ax_mp.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], color='black', label='Flight Path')

        self.ax_mp.legend()
        plt.show()

    def get_state(self):
        return tuple(map(int, self.drone_position))

    def update_drone_position(self, frame):

        time_elapsed = frame * self.dt 
        real_time_elapsed = time.time() - self.start_time
              
        state = self.get_state()
        action = self.agent.choose_action(state)
        next_position = self.drone_position + self.action_space[action] * self.grid_size  

        if self.is_valid_position(next_position):
            self.drone_position = next_position
            self.flight_path.append(self.drone_position)
            reward = self.calculate_reward()
            next_state = self.get_state()
            self.agent.learn(state, action, reward, next_state)

        path = self.plan_path()

        if len(path) > 1:
            new_position = np.array(path[1])
            direction = (new_position - self.drone_position) / np.linalg.norm(new_position - self.drone_position)
            self.drone_position += 0.2 * direction

        self.calculate_control_input()
        self.drone_position += self.u[:3]
        self.flight_path.append(self.drone_position)

        epsilon = 0.1
        if np.linalg.norm(self.drone_position - self.setpoint) < epsilon:
            self.stop_animation()

        # Plot to Visualize
        self.ax_mp.clear()
        self.ax_mp.set_xlabel('X [m]')
        self.ax_mp.set_ylabel('Y [m]')
        self.ax_mp.set_zlabel('Z [m]')
        self.ax_mp.set_title('Motion Planning Trajectory')

        self.ax_mp.scatter(self.drone_position[0], self.drone_position[1], self.drone_position[2], c='b', marker='o', label='Drone')
        self.ax_mp.text(self.drone_position[0], self.drone_position[1], self.drone_position[2], "Drone", color='b')

        self.ax_mp.scatter(self.setpoint[0], self.setpoint[1], self.setpoint[2], c='g', marker='*', label='Setpoint')
        self.ax_mp.text(self.setpoint[0], self.setpoint[1], self.setpoint[2], "Setpoint", color='g')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.obstacle_radius * np.outer(np.cos(u), np.sin(v)) + self.obstacle_position[0]
        y = self.obstacle_radius * np.outer(np.sin(u), np.sin(v)) + self.obstacle_position[1]
        z = self.obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.obstacle_position[2]
        self.ax_mp.plot_surface(x, y, z, color='r', alpha=0.5)
        self.ax_mp.text(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2], "Obstacle", color='r')

        if len(self.flight_path) > 1:
            flight_path = np.array(self.flight_path)
            self.ax_mp.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], c='y', label='Flight Path')
            print(self.flight_path)

        if time_elapsed >= MAX_STEPS * self.dt or np.linalg.norm(self.drone_position - self.setpoint) < 0.1:
            self.stop_animation()
            return

        self.ax_mp.text2D(0.02, 0.90, f'Real Time: {real_time_elapsed:.1f} s', transform=self.ax_mp.transAxes)


    def get_action_vector(self, action):
        actions = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1])
        }
        return actions[action]

    def is_valid_position(self, position):
        return (0 <= position[0] < 10) and (0 <= position[1] < 10) and (0 <= position[2] < 10) and not self.check_collision(position)

    def check_collision(self, position):
        distance_to_obstacle = np.linalg.norm(position - self.obstacle_position)
        return distance_to_obstacle < self.obstacle_radius + SAFETY_DISTANCE

    def calculate_reward(self):
        distance_to_setpoint = np.linalg.norm(self.drone_position - self.setpoint)
        if distance_to_setpoint < 0.1:
            return 100  
        elif self.check_collision(self.drone_position):
            return -100 
        else:
            return -distance_to_setpoint

    def plan_path(self):
        state = self.get_state()
        discretized_state = tuple(int(round(s / self.grid_size)) for s in state)

        action = np.argmax(self.agent.q_table[discretized_state])
        next_position = self.drone_position + self.action_space[action] * self.grid_size

        if self.is_valid_position(next_position):
            return [self.drone_position, next_position]
        else:
            return [self.drone_position]

    def start_animation(self):
        self.start_time = time.time()
        self.animation = animation.FuncAnimation(self.fig_mp, self.update_drone_position, frames=np.arange(0, MAX_STEPS), interval=100)
        plt.show()

    def stop_animation(self):
        self.animation.event_source.stop()

if __name__ == "__main__":
    motion_planning = MotionPlanning()
    motion_planning.start_animation()





# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------


# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------
# --------------------------------------------------- a* -----------------------------------------------------------------------------------------------











import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from queue import PriorityQueue


# Define Constants
TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 0.01
MAX_STEPS = 100

class MotionPlanning:
    def __init__(self):
        # Drone parameters
        g = -9.81  # m/s^2
        m = 0.57  # kg
        Ix = 0.02898  # kgm^2
        Iy = 0.04615  # kgm^2
        Iz = 0.00738  # kgm^2
        
        # Initialize drone, setpoint, obstacle
        self.drone_position = np.array([0.0, 0.0, 0.0])
        self.setpoint = np.array([1.0, 8.0, 2.0])
        self.obstacle_position = np.array([6.0, 2.0, 5.0])
        self.obstacle_radius = 1.2

        # Initialize plot for motion planning
        self.fig_mp = plt.figure()
        self.ax_mp = self.fig_mp.add_subplot(111, projection='3d')
        self.ax_mp.set_xlabel('X [m]')
        self.ax_mp.set_ylabel('Y [m]')
        self.ax_mp.set_zlabel('Z [m]')
        self.ax_mp.set_title('Motion Planning Trajectory')

        # Store drone's flight path
        self.flight_path = []
        
        self.dt = 0.1
        
        # Continuous-time system matrices
        self.A = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       
        ])

        self.B = np.array([
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [1/m, 0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   0,    0,    0   ],
            [0,   1/Ix, 0,    0   ],
            [0,   0,    1/Iy, 0   ],
            [0,   0,    0,    1/Iz]
        ])

        # Discrete-time system matrices
        self.Ad = np.eye(12) + self.dt * self.A
        self.Bd = self.dt * self.B
        
        # Discrete-time Riccati equation solver
        self.Qd = np.eye(12)
        self.Rd = np.eye(4)
        self.Kd = self.compute_lqr()

        self.x = np.zeros(12)
        self.u = np.zeros(4)

        # Initialize time display
        self.time_text = self.ax_mp.text(0.02, 0.02, 0.02, "", transform=self.ax_mp.transAxes)
    
    def compute_lqr(self):
        P = np.eye(12)  # Initialize P matrix
        max_iterations = 1000
        tolerance = 1e-6
        for _ in range(max_iterations):
            P_next = self.A.T @ P @ self.A - (self.A.T @ P @ self.B) @ np.linalg.inv(self.Rd + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A) + self.Qd
            if np.allclose(P_next, P, atol=tolerance):
                break
            P = P_next
        K = np.linalg.inv(self.Rd + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return K
    
    def calculate_control_input(self):
        self.u = -self.Kd.dot(self.x - np.concatenate((self.setpoint, np.zeros(9))))

    def check_collision(self, position):
        distance_to_obstacle = np.linalg.norm(position - self.obstacle_position)
        return distance_to_obstacle < self.obstacle_radius + SAFETY_DISTANCE

    def a_star(self, start, goal):
        # f(n) = g(n) + h(n)
        def heuristic(position, goal):
            return np.linalg.norm(position - goal)

        motions = [(dx, dy, dz) 
                   for dx in range(-1, 2) 
                   for dy in range(-1, 2) 
                   for dz in range(-1, 2)]

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
                break

            for dx, dy, dz in motions:
                next_position = (current[0] + dx, current[1] + dy, current[2] + dz)

                # Check if next position is not in obstacles
                if (0 <= next_position[0] < 10) and (0 <= next_position[1] < 10) and (0 <= next_position[2] < 10):
                    if self.check_collision(next_position):
                        continue
                    
                    new_cost = cost_so_far[current] + 1
                    if next_position not in cost_so_far or new_cost < cost_so_far[next_position]:
                        cost_so_far[next_position] = new_cost
                        priority = new_cost + heuristic(np.array(next_position), goal)
                        open_set.put((priority, next_position))
                        came_from[next_position] = current

        path = [goal]
        current = goal
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plan_path(self):
        start = (int(self.drone_position[0]), int(self.drone_position[1]), int(self.drone_position[2]))
        goal = (int(self.setpoint[0]), int(self.setpoint[1]), int(self.setpoint[2]))
        return self.a_star(start, goal)

    def avoid_obstacle(self, path):
        # Check if the drone is close to the obstacle
        for position in path:
            if np.linalg.norm(position - self.obstacle_position) < self.obstacle_radius + SAFETY_DISTANCE:
                # Calculate a new direction to avoid the obstacle
                obstacle_direction = (position - self.obstacle_position) / np.linalg.norm(position - self.obstacle_position)
                perpendicular_direction = np.array([-obstacle_direction[1], obstacle_direction[0], 0])  # Perpendicular to obstacle direction
                new_direction = np.cross(obstacle_direction, perpendicular_direction)
                new_position = position + new_direction * SAFETY_DISTANCE
                return new_position
        return None

    def update_drone_position(self, frame):
        # Calculate real-world time elapsed
        time_elapsed = frame * self.dt  # Time in seconds
        
        # Calculate the time elapsed in real-world time
        real_time_elapsed = time.time() - self.start_time
        path = self.plan_path()

        if len(path) > 1:
            new_position = np.array(path[1])
            direction = (new_position - self.drone_position) / np.linalg.norm(new_position - self.drone_position)
            self.drone_position += 0.2 * direction

        # Calculate control input
        self.calculate_control_input()

        # Apply control input to drone's position
        self.drone_position += self.u[:3]

        self.flight_path.append(self.drone_position)
        
        epsilon = 0.1
        if np.linalg.norm(self.drone_position - self.setpoint) < epsilon:
            self.stop_animation()

        # Plot to Visualize
        self.ax_mp.clear()
        self.ax_mp.set_xlabel('X [m]')
        self.ax_mp.set_ylabel('Y [m]')
        self.ax_mp.set_zlabel('Z [m]')
        self.ax_mp.set_title('Motion Planning Trajectory')

        # Plot drone position
        self.ax_mp.scatter(self.drone_position[0], self.drone_position[1], self.drone_position[2], c='b', marker='o',
                        label='Drone')
        self.ax_mp.text(self.drone_position[0], self.drone_position[1], self.drone_position[2], "Drone", color='b')

        # Plot setpoint
        self.ax_mp.scatter(self.setpoint[0], self.setpoint[1], self.setpoint[2], c='g', marker='*', label='Setpoint')
        self.ax_mp.text(self.setpoint[0], self.setpoint[1], self.setpoint[2], "Setpoint", color='g')

        # Plot obstacle
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.obstacle_radius * np.outer(np.cos(u), np.sin(v)) + self.obstacle_position[0]
        y = self.obstacle_radius * np.outer(np.sin(u), np.sin(v)) + self.obstacle_position[1]
        z = self.obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.obstacle_position[2]
        self.ax_mp.plot_surface(x, y, z, color='r', alpha=0.5)
        self.ax_mp.text(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2], "Obstacle",
                        color='r')

        # Plot flight path
        if len(self.flight_path) > 1:
            flight_path = np.array(self.flight_path)
            self.ax_mp.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], c='y', label='Flight Path')

        # Check if the animation reached the maximum time or the drone reached the setpoint
        if time_elapsed >= MAX_STEPS * self.dt or np.linalg.norm(self.drone_position - self.setpoint) < 0.1:
            print("Drone reached the goal!")
            self.stop_animation()
            return

        # Annotate time elapsed
        self.ax_mp.text2D(0.02, 0.90, f'Real Time: {real_time_elapsed:.1f} s', transform=self.ax_mp.transAxes)

    def start_animation(self):
        self.start_time = time.time()
        self.animation = animation.FuncAnimation(self.fig_mp, self.update_drone_position, frames=np.arange(0, 1), interval=100)
        plt.show()

    def stop_animation(self):
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()

if __name__ == "__main__":
    motion_planning = MotionPlanning()
    motion_planning.start_animation()









































# import numpy as np
# import control as ct
# import matplotlib.pyplot as plt
# from scipy.linalg import solve_continuous_are
# import matplotlib.animation as animation
# from queue import PriorityQueue


# # Define Constants
# TARGET_ALTITUDE = 5
# SAFETY_DISTANCE = 0.01

# class MotionPlanning:
#     def __init__(self):
#         # Initialize drone, setpoint, obstacle
#         self.drone_position = np.array([4.0, 4.0, 5.0])
#         self.setpoint = np.random.uniform(low=[0.0, 0.0, 0.0], high=[10.0, 10.0, 10.0])
#         self.obstacle_position = np.random.uniform(low=[0.0, 0.0, 0.0], high=[10.0, 10.0, 10.0])
#         self.obstacle_radius = np.random.uniform(0.5, 2.0)

#         # Initialize plot for motion planning
#         self.fig_mp = plt.figure()
#         self.ax_mp = self.fig_mp.add_subplot(111, projection='3d')
#         self.ax_mp.set_xlabel('X [m]')
#         self.ax_mp.set_ylabel('Y [m]')
#         self.ax_mp.set_zlabel('Z [m]')
#         self.ax_mp.set_title('Motion Planning Trajectory')

#         # Store drone's flight path
#         self.flight_path = []
        
#         g = -9.81 # m/s^2
#         m = 0.57 # kg
        
#         Ix = 0.02898 # kgm^2
#         Iy = 0.04615 # kgm^2 
#         Iz = 0.00738 # kgm^2 
#         self.A = np.array([
#             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
#             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
#             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  
#             [0, 0, 0, 0, 0, 0, 0,-g, 0, 0, 0, 0], 
#             [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],  
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       
#         ])

#         self.B = np.array([
#             [0,   0,    0,    0   ],
#             [0,   0,    0,    0   ],
#             [0,   0,    0,    0   ],
#             [0,   0,    0,    0   ],
#             [1/m, 0,    0,    0   ],
#             [0,   0,    0,    0   ],
#             [0,   0,    0,    0   ],
#             [0,   0,    0,    0   ],
#             [0,   0,    0,    0   ],
#             [0,   1/Ix, 0,    0   ],
#             [0,   0,    1/Iy, 0   ],
#             [0,   0,    0,    1/Iz]
#         ])
        
#         # self.C = np.array([
#         #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
#         # ])
        
#         # self.D = np.array([
#         #     [0, 0, 0, 0],
#         #     [0, 0, 0, 0],
#         #     [0, 0, 0, 0],
#         #     [0, 0, 0, 0],
#         #     [0, 0, 0, 0],
#         #     [0, 0, 0, 0]
#         # ])
        
#         # Discrete-time system matrices
#         self.dt = 0.1  # Time step
#         self.Ad = np.eye(12) + self.dt * self.A
#         self.Bd = self.dt * self.B
        
#         # Discrete-time Riccati equation solver
#         self.Qd = np.eye(12)
#         self.Rd = np.eye(4)
#         self.Kd = np.linalg.inv(self.Rd + self.Bd.T @ self.Qd @ self.Bd) @ self.Bd.T @ self.Qd @ self.Ad

#         self.x = np.zeros(12)
#         self.u = np.zeros(4)

#     def calculate_control_input(self):
#         self.u = -self.Kd.dot(self.x - np.concatenate((self.setpoint, np.zeros(9))))

#     def visualize_trajectory(self):
#         self.ax_mp.scatter(self.drone_position[0], self.drone_position[1], self.drone_position[2], color='red', label='Drone')
#         self.ax_mp.scatter(self.setpoint[0], self.setpoint[1], self.setpoint[2], color='green', label='Setpoint')
#         self.ax_mp.scatter(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2], color='blue', label='Obstacle')

#         if self.flight_path:
#             flight_path = np.array(self.flight_path)
#             self.ax_mp.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], color='black', label='Flight Path')

#         self.ax_mp.legend()
#         plt.show()
        
# # if __name__ == "__main__":
# #     motion_planning = MotionPlanning()
    
#     def check_collision(self, position):
#         distance_to_obstacle = np.linalg.norm(position - self.obstacle_position)
#         return distance_to_obstacle < self.obstacle_radius + SAFETY_DISTANCE

#     def a_star(self, start, goal):
#         # f(n) = g(n) + h(n)
#         def heuristic(position, goal):
#             return np.linalg.norm(position - goal)

#         motions = [(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]

#         open_set = PriorityQueue()
#         open_set.put((0, start))
#         came_from = {}
#         cost_so_far = {start: 0}

#         while not open_set.empty():
#             _, current = open_set.get()

#             if current == goal:
#                 break

#             for dx, dy, dz in motions:
#                 next_position = (current[0] + dx, current[1] + dy, current[2] + dz)

#                 # Check if next position is not in obstacles
#                 if (0 <= next_position[0] < 10) and (0 <= next_position[1] < 10) and (0 <= next_position[2] < 10):
#                     if self.check_collision(next_position):
#                         continue
                    
#                     new_cost = cost_so_far[current] + 1
#                     if next_position not in cost_so_far or new_cost < cost_so_far[next_position]:
#                         cost_so_far[next_position] = new_cost
#                         priority = new_cost + heuristic(np.array(next_position), goal)
#                         open_set.put((priority, next_position))
#                         came_from[next_position] = current

#         path = [goal]
#         current = goal
#         while current != start:
#             current = came_from[current]
#             path.append(current)
#         path.reverse()
#         return path

#     def plan_path(self):
#         start = (int(self.drone_position[0]), int(self.drone_position[1]), int(self.drone_position[2]))
#         goal = (int(self.setpoint[0]), int(self.setpoint[1]), int(self.setpoint[2]))
#         return self.a_star(start, goal)

#     def avoid_obstacle(self, path):
#         # Check if the drone is close to the obstacle
#         for position in path:
#             if np.linalg.norm(position - self.obstacle_position) < self.obstacle_radius + SAFETY_DISTANCE:
#                 # Calculate a new direction to avoid the obstacle
#                 obstacle_direction = (position - self.obstacle_position) / np.linalg.norm(position - self.obstacle_position)
#                 perpendicular_direction = np.array([-obstacle_direction[1], obstacle_direction[0], 0])  # Perpendicular to obstacle direction
#                 new_direction = np.cross(obstacle_direction, perpendicular_direction)
#                 new_position = position + new_direction * SAFETY_DISTANCE
#                 return new_position
#         return None

#     # Visualize as Continuous plot
#     def update_drone_position(self, frame):
#         path = self.plan_path()

#         if len(path) > 1:
#             new_position = np.array(path[1])
#             direction = (new_position - self.drone_position) / np.linalg.norm(new_position - self.drone_position)
#             self.drone_position += 0.2 * direction

#         # Calculate control input
#         self.calculate_control_input()

#         # Apply control input to drone's position
#         self.drone_position += self.u[:3]

#         self.flight_path.append(self.drone_position)

#         epsilon = 0.1
#         if np.linalg.norm(self.drone_position - self.setpoint) < epsilon:
#             self.stop_animation()

#         # Plot to Visualize
#         self.ax_mp.clear()
#         self.ax_mp.set_xlabel('X [m]')
#         self.ax_mp.set_ylabel('Y [m]')
#         self.ax_mp.set_zlabel('Z [m]')
#         self.ax_mp.set_title('Motion Planning Trajectory')

#         self.ax_mp.scatter(self.drone_position[0], self.drone_position[1], self.drone_position[2], c='b', marker='o', label='Drone')
#         self.ax_mp.text(self.drone_position[0], self.drone_position[1], self.drone_position[2], "Drone", color='b')

#         self.ax_mp.scatter(self.setpoint[0], self.setpoint[1], self.setpoint[2], c='g', marker='*', label='Setpoint')
#         self.ax_mp.text(self.setpoint[0], self.setpoint[1], self.setpoint[2], "Setpoint", color='g')

#         u = np.linspace(0, 2 * np.pi, 100)
#         v = np.linspace(0, np.pi, 100)
#         x = self.obstacle_radius * np.outer(np.cos(u), np.sin(v)) + self.obstacle_position[0]
#         y = self.obstacle_radius * np.outer(np.sin(u), np.sin(v)) + self.obstacle_position[1]
#         z = self.obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.obstacle_position[2]
#         self.ax_mp.plot_surface(x, y, z, color='r', alpha=0.5)
#         self.ax_mp.text(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2], "Obstacle", color='r')

#         if len(self.flight_path) > 1:
#             flight_path = np.array(self.flight_path)
#             self.ax_mp.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], c='y', label='Flight Path')
#             print(self.flight_path)


#     def start_animation(self):
#         self.animation = animation.FuncAnimation(self.fig_mp, self.update_drone_position, frames=np.arange(0, 100), interval=100)
#         plt.show()

#     def stop_animation(self):
#         if hasattr(self, 'animation'):
#             self.animation.event_source.stop()

# if __name__ == "__main__":
#     motion_planning = MotionPlanning()
#     motion_planning.start_animation()
    
    




    
    
    
    
# # # import numpy as np
# # # import control as ct
# # # import matplotlib.pyplot as plt
# # # import matplotlib.animation as animation

# # # # Define Constants
# # # TARGET_ALTITUDE = 5
# # # SAFETY_DISTANCE = 0.2

# # # class MotionPlanning:
# # #     def __init__(self):
# # #         # Initialize drone, setpoint, obstacle
# # #         self.drone_position = np.array([4.0, 4.0, 1.0])
# # #         self.setpoint = np.array([8.0, 1.0, 5.0])
# # #         self.obstacle_position = np.array([6.0, 2.0, 5.0])
# # #         self.obstacle_radius = 1.0

# # #         # Initialize plot for motion planning
# # #         self.fig_mp = plt.figure()
# # #         self.ax_mp = self.fig_mp.add_subplot(111, projection='3d')
# # #         self.ax_mp.set_xlabel('X [m]')
# # #         self.ax_mp.set_ylabel('Y [m]')
# # #         self.ax_mp.set_zlabel('Z [m]')
# # #         self.ax_mp.set_title('Motion Planning Trajectory')

# # #         # Store drone's flight path
# # #         self.flight_path = []

# # #         # Define RL parameters
# # #         self.Q_table = np.zeros((10, 10, 10, 27))  # Q-table: state space (10x10x10) x action space (27)
# # #         self.learning_rate = 0.1
# # #         self.discount_factor = 0.9
# # #         self.epsilon = 0.1
# # #         g = 9.81 # m/s^2
# # #         m = 0.57 # kg
# # #         Ix = 0.02898 # kgm^2
# # #         Iy = 0.04615 # kgm^2 
# # #         Iz = 0.00738 # kgm^2 
        
# # #         self.A = np.array([
# # #             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
# # #             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  
# # #             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  
# # #             [0, 0, 0, 0, 0, 0, 0,-g, 0, 0, 0], 
# # #             [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],  
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       
# # #         ])

# # #         self.B = np.array([
# # #             [0,   0,    0,    0   ],
# # #             [0,   0,    0,    0   ],
# # #             [0,   0,    0,    0   ],
# # #             [0,   0,    0,    0   ],
# # #             [1/m, 0,    0,    0   ],
# # #             [0,   0,    0,    0   ],
# # #             [0,   0,    0,    0   ],
# # #             [0,   0,    0,    0   ],
# # #             [0,   0,    0,    0   ],
# # #             [0,   1/Ix, 0,    0   ],
# # #             [0,   0,    1/Iy, 0   ],
# # #             [0,   0,    0,    1/Iz]
# # #         ])
        
# # #         self.C = np.array([
# # #             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# # #             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# # #             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# # #             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
# # #             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
# # #             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# # #         ])
        
# # #         self.D = np.array([
# # #             [0, 0, 0, 0],
# # #             [0, 0, 0, 0],
# # #             [0, 0, 0, 0],
# # #             [0, 0, 0, 0],
# # #             [0, 0, 0, 0],
# # #             [0, 0, 0, 0]
# # #         ])

# # #         sysStateSpace=ct.ss(self.A,self.B,self.C,self.D)
        
# # #         # Define Cost Matrices Q and R for LQR controller
# # #         self.Q = np.eye(6)  # Identity matrix for state variables
# # #         self.R = np.eye(3)  # Identity matrix for control inputs

# # #         K, S, E = ct.lqr(sysStateSpace, self.Q, self.R)
        
# # #         Acl = self.A-np.matmul(self.B,K)

# # #         np.linalg.eig(Acl)[0]
        
# # #         self.compute_lqr_gain_matrix()

# # #     def compute_lqr_gain_matrix(self):
# # #         # Solve the discrete-time Algebraic Riccati Equation (DARE) to compute the LQR gain matrix K
# # #         A_d = np.eye(self.A.shape[0]) + self.A
# # #         B_d = self.B
# # #         P = np.eye(self.A.shape[0])
# # #         for _ in range(100):  # Iteratively solve DARE
# # #             P_new = self.Q + A_d.T @ P @ A_d - A_d.T @ P @ B_d @ np.linalg.inv(self.R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
# # #             if np.allclose(P, P_new):  # Check for convergence
# # #                 break
# # #             P = P_new
# # #         self.K = np.linalg.inv(self.R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d

# # #     def compute_control_input(self, state):
# # #         # Compute control input u = -K * x
# # #         control_input = -self.K @ state
# # #         return control_input

# # #     def discretize_state(self, position):
# # #         return tuple(np.floor(position).astype(int))

# # #     def discretize_action(self, action):
# # #         # Convert continuous action to discrete action
# # #         return tuple(np.round(action).astype(int))

# # #     def get_reward(self, state):
# # #         # Define reward function
# # #         distance_to_setpoint = np.linalg.norm(self.setpoint - state)
# # #         return -distance_to_setpoint  # Negative distance as closer is better

# # #     def update_Q_table(self, state, action, next_state, reward):
# # #         # Ensure state and action are tuples
# # #         state = tuple(state)
# # #         action = (action,)  # Convert action to a tuple
        
# # #         # Q-learning update rule
# # #         current_Q = self.Q_table[state + action]  # Concatenate action tuple with state tuple
# # #         next_max_Q = np.max(self.Q_table[next_state])
# # #         new_Q = (1 - self.learning_rate) * current_Q + self.learning_rate * (reward + self.discount_factor * next_max_Q)
# # #         self.Q_table[state + action] = new_Q  # Concatenate action tuple with state tuple

# # #     def select_action(self, state):
# # #         # Epsilon-greedy policy for action selection
# # #         if np.random.rand() < self.epsilon:
# # #             # Explore: Choose a random action
# # #             return tuple(np.random.randint(-1, 2, size=3))  # Random action in range [-1, 1] for each dimension
# # #         else:
# # #             # Exploit: Choose the action with maximum Q-value
# # #             return tuple(np.unravel_index(np.argmax(self.Q_table[state]), (3, 3, 3)))  # Convert action index to tuple

# # #     def update_drone_position(self, frame):
# # #         # Discretize current state
# # #         state = self.discretize_state(self.drone_position)
# # #         print("State dimension:", len(state))

# # #         # Select action using epsilon-greedy policy
# # #         action = self.select_action(state)

# # #         # Update drone position using LQR controller
# # #         control_input = self.compute_control_input(self.drone_position)
# # #         print("Control input dimension:", len(control_input))  # Add this line
# # #         new_position = self.drone_position + control_input * 0.2  # Move drone based on control input
# # #         self.drone_position = np.clip(new_position, 0, 9.9)  # Clip drone position within boundaries

# # #         # Discretize next state
# # #         next_state = self.discretize_state(self.drone_position)
# # #         print("Next state dimension:", len(next_state))

# # #         # Get reward for the new state
# # #         reward = self.get_reward(next_state)

# # #         # Update Q-table
# # #         self.update_Q_table(state, action, next_state, reward)

# # #         # Plot to Visualize
# # #         self.ax_mp.clear()
# # #         self.ax_mp.set_xlabel('X [m]')
# # #         self.ax_mp.set_ylabel('Y [m]')
# # #         self.ax_mp.set_zlabel('Z [m]')
# # #         self.ax_mp.set_title('Motion Planning Trajectory')

# # #         self.ax_mp.scatter(self.drone_position[0], self.drone_position[1], self.drone_position[2], c='b', marker='o', label='Drone')
# # #         self.ax_mp.text(self.drone_position[0], self.drone_position[1], self.drone_position[2], "Drone", color='b')

# # #         self.ax_mp.scatter(self.setpoint[0], self.setpoint[1], self.setpoint[2], c='g', marker='*', label='Setpoint')
# # #         self.ax_mp.text(self.setpoint[0], self.setpoint[1], self.setpoint[2], "Setpoint", color='g')

# # #         # Plot obstacle
# # #         u = np.linspace(0, 2 * np.pi, 100)
# # #         v = np.linspace(0, np.pi, 100)
# # #         x = self.obstacle_radius * np.outer(np.cos(u), np.sin(v)) + self.obstacle_position[0]
# # #         y = self.obstacle_radius * np.outer(np.sin(u), np.sin(v)) + self.obstacle_position[1]
# # #         z = self.obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.obstacle_position[2]
# # #         self.ax_mp.plot_surface(x, y, z, color='r', alpha=0.5)
# # #         self.ax_mp.text(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2], "Obstacle", color='r')

# # #         # Plot flight path
# # #         if len(self.flight_path) > 1:
# # #             flight_path = np.array(self.flight_path)
# # #             self.ax_mp.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], c='y', label='Flight Path')

# # #         # Update flight path
# # #         self.flight_path.append(self.drone_position)

# # #         # Check if reached setpoint
# # #         epsilon = 0.1
# # #         if np.linalg.norm(self.drone_position - self.setpoint) < epsilon:
# # #             self.stop_animation()

# # #     def start_animation(self):
# # #         self.animation = animation.FuncAnimation(self.fig_mp, self.update_drone_position, frames=np.arange(0, 100), interval=100)
# # #         plt.show()

# # #     def stop_animation(self):
# # #         if hasattr(self, 'animation'):
# # #             self.animation.event_source.stop()

# # # if __name__ == "__main__":
# # #     motion_planning = MotionPlanning()
# # #     motion_planning.start_animation()


# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import matplotlib.animation as animation
# # # from queue import PriorityQueue

# # # # Define Constants
# # # TARGET_ALTITUDE = 5
# # # SAFETY_DISTANCE = 0.2

# # # class MotionPlanning:
# # #     def __init__(self):
# # #         # Initialize drone, setpoint, obstacle
# # #         self.drone_position = np.array([4.0, 4.0, 1.0])
# # #         self.setpoint = np.array([8.0, 1.0, 5.0])
# # #         self.obstacle_position = np.array([6.0, 2.0, 5.0])
# # #         self.obstacle_radius = 1.0

# # #         # Initialize plot for motion planning
# # #         self.fig_mp = plt.figure()
# # #         self.ax_mp = self.fig_mp.add_subplot(111, projection='3d')
# # #         self.ax_mp.set_xlabel('X [m]')
# # #         self.ax_mp.set_ylabel('Y [m]')
# # #         self.ax_mp.set_zlabel('Z [m]')
# # #         self.ax_mp.set_title('Motion Planning Trajectory')

# # #         # Store drone's flight path
# # #         self.flight_path = []

# # #         # Define RL parameters
# # #         self.Q_table = np.zeros((10, 10, 10, 27))  # Q-table: state space (10x10x10) x action space (27)
# # #         self.learning_rate = 0.1
# # #         self.discount_factor = 0.9
# # #         self.epsilon = 0.1

# # #     def discretize_state(self, position):
# # #         return tuple(np.floor(position).astype(int))

# # #     def discretize_action(self, action):
# # #         # Convert continuous action to discrete action
# # #         return tuple(np.round(action).astype(int))

# # #     def get_reward(self, state):
# # #         # Define reward function
# # #         distance_to_setpoint = np.linalg.norm(self.setpoint - state)
# # #         return -distance_to_setpoint  # Negative distance as closer is better

# # #     def update_Q_table(self, state, action, next_state, reward):
# # #         # Ensure state and action are tuples
# # #         state = tuple(state)
# # #         action = (action,)  # Convert action to a tuple
        
# # #         # Q-learning update rule
# # #         current_Q = self.Q_table[state + action]  # Concatenate action tuple with state tuple
# # #         next_max_Q = np.max(self.Q_table[next_state])
# # #         new_Q = (1 - self.learning_rate) * current_Q + self.learning_rate * (reward + self.discount_factor * next_max_Q)
# # #         self.Q_table[state + action] = new_Q  # Concatenate action tuple with state tuple



# # #     def select_action(self, state):
# # #         # Epsilon-greedy policy for action selection
# # #         if np.random.rand() < self.epsilon:
# # #             # Explore: Choose a random action
# # #             return tuple(np.random.randint(-1, 2, size=3))  # Random action in range [-1, 1] for each dimension
# # #         else:
# # #             # Exploit: Choose the action with maximum Q-value
# # #             return tuple(np.unravel_index(np.argmax(self.Q_table[state]), (3, 3, 3)))  # Convert action index to tuple

# # #     def update_drone_position(self, frame):
# # #         # Discretize current state
# # #         state = self.discretize_state(self.drone_position)

# # #         # Select action using epsilon-greedy policy
# # #         action = self.select_action(state)

# # #         # Update drone position
# # #         new_position = self.drone_position + np.array(action) * 0.2  # Move drone based on action
# # #         self.drone_position = np.clip(new_position, 0, 9.9)  # Clip drone position within boundaries

# # #         # Discretize next state
# # #         next_state = self.discretize_state(self.drone_position)

# # #         # Get reward for the new state
# # #         reward = self.get_reward(next_state)

# # #         # Update Q-table
# # #         self.update_Q_table(state, action, next_state, reward)

# # #         # Plot to Visualize
# # #         self.ax_mp.clear()
# # #         self.ax_mp.set_xlabel('X [m]')
# # #         self.ax_mp.set_ylabel('Y [m]')
# # #         self.ax_mp.set_zlabel('Z [m]')
# # #         self.ax_mp.set_title('Motion Planning Trajectory')

# # #         self.ax_mp.scatter(self.drone_position[0], self.drone_position[1], self.drone_position[2], c='b', marker='o', label='Drone')
# # #         self.ax_mp.text(self.drone_position[0], self.drone_position[1], self.drone_position[2], "Drone", color='b')

# # #         self.ax_mp.scatter(self.setpoint[0], self.setpoint[1], self.setpoint[2], c='g', marker='*', label='Setpoint')
# # #         self.ax_mp.text(self.setpoint[0], self.setpoint[1], self.setpoint[2], "Setpoint", color='g')

# # #         u = np.linspace(0, 2 * np.pi, 100)
# # #         v = np.linspace(0, np.pi, 100)
# # #         x = self.obstacle_radius * np.outer(np.cos(u), np.sin(v)) + self.obstacle_position[0]
# # #         y = self.obstacle_radius * np.outer(np.sin(u), np.sin(v)) + self.obstacle_position[1]
# # #         z = self.obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.obstacle_position[2]
# # #         self.ax_mp.plot_surface(x, y, z, color='r', alpha=0.5)
# # #         self.ax_mp.text(self.obstacle_position[0], self.obstacle_position[1], self.obstacle_position[2], "Obstacle", color='r')

# # #         if len(self.flight_path) > 1:
# # #             flight_path = np.array(self.flight_path)
# # #             self.ax_mp.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], c='y', label='Flight Path')

# # #         # Update flight path
# # #         self.flight_path.append(self.drone_position)

# # #         # Check if reached setpoint
# # #         epsilon = 0.1
# # #         if np.linalg.norm(self.drone_position - self.setpoint) < epsilon:
# # #             self.stop_animation()

# # #     def start_animation(self):
# # #         self.animation = animation.FuncAnimation(self.fig_mp, self.update_drone_position, frames=np.arange(0, 100), interval=100)
# # #         plt.show()

# # #     def stop_animation(self):
# # #         if hasattr(self, 'animation'):
# # #             self.animation.event_source.stop()

# # # if __name__ == "__main__":
# # #     motion_planning = MotionPlanning()
# # #     motion_planning.start_animation()
