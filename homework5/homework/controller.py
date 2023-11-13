import pystk
from itertools import product
from homework.utils import PyTux
import numpy as np

def control(aim_point, current_velocity):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_velocity: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift, nitro)
    """
    # Constants
    base_target_velocity = 40
    steering_scale = 3.0
    max_steering_angle = 1
    drift_threshold = 0.90
    nitro_threshold = 0.25
    brake_threshold = 0.90

    # Instantiate the action object
    action = pystk.Action()

    # Adaptive Target Velocity based on aim point (assuming sharper turn means closer aim point)
    target_velocity = base_target_velocity * (1 - abs(aim_point[0]))

    # Dynamic Steering Adjustment based on current velocity
    steering_scale = steering_scale * (base_target_velocity / (current_velocity + 1))

    # Calculate the difference between the current velocity and the target velocity
    velocity_difference = target_velocity - current_velocity

    # Set acceleration to 1 if below target velocity, else 0
    action.acceleration = 2.0 if velocity_difference > 0 else 0

    # Calculate steering angle based on the aim point
    steering_angle = aim_point[0] * steering_scale
    steering_angle = max(min(steering_angle, max_steering_angle), -max_steering_angle)
    action.steer = steering_angle

    # Advanced Drifting Logic
    action.drift = abs(steering_angle) > drift_threshold and current_velocity > target_velocity

    # Smart Braking Strategy
    action.brake = abs(steering_angle) > brake_threshold and current_velocity > target_velocity

    # Use nitro if going straight or almost straight
    action.nitro = abs(steering_angle) < nitro_threshold

    return action

if __name__ == '__main__':
    from argparse import ArgumentParser

    def test_controller(args):
        pytux = PyTux()
        for track_name in args.track:
            steps, distance_traveled = pytux.rollout(track_name, control, max_frames=1000, verbose=args.verbose)
            print(f'Track: {track_name}, Steps: {steps}, Distance: {distance_traveled}')
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
