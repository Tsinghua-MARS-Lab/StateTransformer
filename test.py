import numpy as np


def compute_yaw_angles_with_interval(x, y, interval=3, movement_threshold=0.01):
    """Compute yaw angles relative to the previous point using specified intervals, filtering out noise when there's negligible movement."""

    # Ensure the interval is at least 2
    interval = max(2, interval)

    # Initialize list for yaw angles
    yaw_angles = [0]  # start with 0 for the first angle

    for i in range(0, len(x) - interval + 1, interval - 1):
        dx = x[i + interval - 1] - x[i]
        dy = y[i + interval - 1] - y[i]

        # Compute the distance moved over the interval
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Compute yaw angle using arctan2, but only where there's sufficient movement
        yaw_angle = np.arctan2(dy, dx) if distance > movement_threshold else 0
        yaw_angles.append(yaw_angle)

    # Get yaw angles relative to the previous interval
    relative_yaw_angles = np.diff(yaw_angles, prepend=yaw_angles[0])

    return relative_yaw_angles


# Example data with high frequency
x = np.array([0, 1, 2, 2.1, 2, 2.1, 2, 1, 1.1, 1])
y = np.array([0, 2, 1, 1.1, 1, 1.1, 1, 0, -0.1, -1])

dx = x[4::5] - x[:-4:5]
dy = y[4::5] - y[:-4:5]
print(dx, dy)
distances = np.sqrt(dx ** 2 + dy ** 2)
yaw_angles = np.where(distances > 0.05, np.arctan2(dy, dx), 0)
# accumulate yaw angle
relative_yaw_angles = yaw_angles.cumsum()
t = np.repeat(relative_yaw_angles, 5, axis=0)
print(yaw_angles, relative_yaw_angles, len(x), relative_yaw_angles.shape, t, t.shape)
exit()
relative_yaw_angles = compute_yaw_angles_with_interval(x, y)

# Display the relative yaw angle for each interval
for i, angle in enumerate(relative_yaw_angles):
    print(f"Interval {i} to {i + 1}: Relative Yaw Angle (in radians) = {angle:.2f}, (in degrees) = {np.degrees(angle):.2f}")

