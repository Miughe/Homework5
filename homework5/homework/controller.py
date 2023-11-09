import pystk
import numpy as np

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    # Target a constant velocity
    target_velocity = 40  # You can adjust this value as needed

    # Scale the aim_point to get normalized steering
    steering_scale = 2.0  # You can tune this factor
    target_steer = aim_point[0] * steering_scale
    action.steer = np.clip(target_steer, -1, 1)  # Clip the steer angle to -1..1

    # Check for skidding
    skid_threshold = 0.4  # Adjust this threshold as needed
    if abs(target_steer) > skid_threshold:
        action.drift = True  # Skid if the steering angle is too large
    else:
        action.drift = False

    # Adjust acceleration and braking
    if current_vel < target_velocity:
        action.acceleration = 1  # Accelerate
        action.brake = False
    else:
        action.acceleration = 0  # Maintain velocity
        action.brake = True  # You may set this to True for sharper turns if needed

    # Use nitro for fast acceleration (optional)
    use_nitro_threshold = 3.0  # Adjust this threshold as needed
    if abs(target_steer) < use_nitro_threshold and current_vel > target_velocity * 0.6:
        action.nitro = True
    else:
        action.nitro = False

    # Drift logic
    if abs(target_steer) > 0.8:  # You can adjust this threshold as needed
        action.drift = True  # Drift during sharp turns
    else:
        action.drift = False

    return action

if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
