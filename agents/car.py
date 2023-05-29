import numpy as np

def vehicle_dyn(veh_state, actions_num=None, actions_str=[],
                if_error=False, r_seed=False, frequency=0.01, length=2):
    """
    Model the vehicle dynamics and constraints.
    actions_num: [v_long_dot (in m/s/s), wheel_angle (in radius [-pi,pi])]
    frequency: /s
    length: in s
    """

    # Define the vehicle parameters.
    (x, y, theta, v_long, v_lat, v_long_dot, omega_r, wheel_anlge) = veh_state
    (l_1, l_2, m, I_z, h, r_wheel) = (1.421, 1.434, 2270, 4600, 0.647, 0.351)
    (C_alhpa, F_x_max, F_y_1_max, F_y_2_max, mu_max, T_max, F_x_Tmax) = (
        100000, 20000, 10400, 10600, 0.88, 3000, 3000 / 0.351)

    if actions_num is not None:
        v_long_dot_target, wheel_anlge_target = actions_num

    # Get maximum acceleration or deceleration under the decision u.
    if 'dec-all' in actions_str:
        v_long_dot_target = -8
    elif 'dec-half' in actions_str:
        v_long_dot_target = -4
    elif 'cons' in actions_str:
        v_long_dot_target = 0
    elif 'acc-half' in actions_str:
        v_long_dot_target = 4
    elif 'acc-all' in actions_str:
        v_long_dot_target = 8

    # Get maximum steering angle under the decision u.
    wheel_anlge_max_ = np.max([(25 - v_long) / 25 * 15 + 5, 5])
    if 'left-all' in actions_str:
        wheel_anlge_target = np.deg2rad(wheel_anlge_max_)
    elif 'left-half' in actions_str:
        wheel_anlge_target = np.deg2rad(wheel_anlge_max_ / 2)
    elif 'straight' in actions_str:
        wheel_anlge_target = 0
    elif 'right-half' in actions_str:
        wheel_anlge_target = np.deg2rad(- wheel_anlge_max_ / 2)
    elif 'right-all' in actions_str:
        wheel_anlge_target = np.deg2rad(- wheel_anlge_max_)
    else:
        wheel_anlge_target = np.clip(wheel_anlge_target, -wheel_anlge_max_, wheel_anlge_max_)

    x_list, y_list, theta_list, v_long_list, v_lat_list, v_long_dot_list, omega_r_list, wheel_anlge_list = [], [], [], [], [], [], [], []
    # x_list.append(x)
    # y_list.append(y)
    # theta_list.append(theta)
    # v_long_list.append(v_long)
    # v_lat_list.append(v_lat)
    # v_long_dot_list.append(v_long_dot)
    # omega_r_list.append(omega_r)
    # wheel_anlge_list.append(wheel_anlge)

    # Define the random seed.
    np.random.seed(r_seed)

    # Update vehicle dynamics using the defined plane bicycle model.
    time_step = frequency
    for time_i in range(int(length/time_step)):
        if v_long_dot < 0.1 and v_long < 0.02 and v_long_dot_target < 0.01:
            x_list.append(x)
            y_list.append(y)
            theta_list.append(theta)
            v_long_list.append(v_long)
            v_lat_list.append(v_lat)
            v_long_dot_list.append(v_long_dot)
            omega_r_list.append(omega_r)
            wheel_anlge_list.append(wheel_anlge)
            continue

        beta = np.arctan(v_lat / (v_long + 0.001))

        alpha_1 = - (beta + l_1 * omega_r / (v_long + 0.001) - wheel_anlge)
        alpha_2 = - (beta - l_1 * omega_r / (v_long + 0.001))

        # Define the simplified linear vehicle tire model with saturation.
        F_y_1 = np.min([C_alhpa * np.abs(alpha_1), C_alhpa * np.deg2rad(8)]) * np.sign(alpha_1)
        F_y_2 = np.min([C_alhpa * np.abs(alpha_2), C_alhpa * np.deg2rad(8)]) * np.sign(alpha_2)

        omega_r_dot = (l_1 * F_y_1 * np.cos(wheel_anlge) - l_2 * F_y_2) / I_z
        v_lat_dot = (F_y_1 * np.cos(wheel_anlge) + F_y_2) / m - v_long * wheel_anlge
        F_x = m * (v_long_dot - v_lat * wheel_anlge)

        omega_r += omega_r_dot * time_step
        v_lat += v_lat_dot * time_step

        # Define control errors in the vehicle longitudinal dynamics.
        Cont_error_long = np.random.normal(0, 2, 1)[0] if if_error else 0
        v_long_dot_dot = 20 + Cont_error_long
        if 0 <= v_long_dot_target - v_long_dot <= v_long_dot_dot * time_step or 0 >= v_long_dot_target - v_long_dot >= v_long_dot_dot * time_step:
            v_long_dot = v_long_dot_target
        else:
            v_long_dot += v_long_dot_dot * time_step if v_long_dot < v_long_dot_target else -(
                    v_long_dot_dot + 5) * time_step
        v_long += v_long_dot * time_step

        # Define control errors in the vehicle lateral dynamics.
        Cont_error_lat = np.random.normal(0, 2, 1)[0] if if_error else 0
        wheel_anlge_dot = np.deg2rad(15 + Cont_error_lat)
        if 0 <= wheel_anlge_target - wheel_anlge <= wheel_anlge_dot * time_step or 0 >= wheel_anlge_target - wheel_anlge >= wheel_anlge_dot * time_step:
            wheel_anlge = wheel_anlge_target
        else:
            wheel_anlge += wheel_anlge_dot * time_step if wheel_anlge < wheel_anlge_target else -wheel_anlge_dot * time_step

        theta += omega_r * time_step
        x += (v_long * np.cos(theta) - v_lat * np.sin(theta)) * time_step
        y += (v_lat * np.cos(theta) + v_long * np.sin(theta)) * time_step

        x_list.append(x)
        y_list.append(y)
        theta_list.append(theta)
        v_long_list.append(v_long)
        v_lat_list.append(v_lat)
        v_long_dot_list.append(v_long_dot)
        omega_r_list.append(omega_r)
        wheel_anlge_list.append(wheel_anlge)

        # Define vehicle tire force constraints based on the laws of physics (friction ellipse).
        mu_1 = (F_x / F_x_max) ** 2 + (F_y_1 / F_y_1_max / (1 - F_x * h / (m * 9.8) / l_1)) ** 2
        mu_2 = (F_x / F_x_max) ** 2 + (F_y_2 / F_y_2_max / (1 + F_x * h / (m * 9.8) / l_2)) ** 2
        if mu_1 > mu_max or mu_2 > mu_max or F_x > F_x_Tmax:
            if v_long_dot > 0:
                v_long_dot -= 0.5
            else:
                v_long_dot += 0.5

    return x_list, y_list, theta_list, v_long_list, v_lat_list, v_long_dot_list, omega_r_list, wheel_anlge_list

class Agent:
    def __init__(self,
                 # init location, angle, velocity
                 x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, length=4.726, width=1.842, agent_id=None, color=None):
        self.x = x  # px
        self.y = y
        self.yaw = self.yaw_changer(yaw)
        self.vx = vx  # px/frame
        self.vy = vy
        self.length = max(1, length)
        self.width = max(1, width)
        self.agent_polys = []
        self.crashed = False
        self.agent_id = agent_id
        self.color = color

    def yaw_changer(self, yaw):
        return -yaw
