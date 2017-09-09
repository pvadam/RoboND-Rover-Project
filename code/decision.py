import numpy as np
import time
from supporting_functions import get_angle

# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating
    # autonomously!

    # Distance required to fully stop (very rough estimate)
    # Rover.stop_forward = (Rover.vel * 10 / Rover.brake_set)**2

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        
        returned_from_approach_mode = False
    
        # Reset pick up command if pick up done
        if Rover.picking_up and not Rover.near_sample and Rover.vel == 0:
            Rover.send_pickup = False
            Rover.picking_up = False
            Rover.samples_collected = Rover.samples_collected + 1
        
        if Rover.mode == 'approach':
            # Check the terrain right in front of the camera first
            if Rover.close_obstacle or len(Rover.nav_dists) == 0:
                print ('exit from approach')
                print (len(Rover.nav_dists))
                # exit from approach mode
                Rover.mode = 'forward'
                returned_from_approach_mode = True
            else:
                mean_angle = np.mean(Rover.nav_angles * 180 / np.pi)
                if Rover.near_sample and not Rover.picking_up:
                # if  min(Rover.nav_dists) <= 10 and abs(mean_angle % 360) < 5:
                    print('picking up')
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
                    Rover.send_pickup = True
                    # Check the extent of navigable terrain
                else:
                    approach_speed_limit = Rover.max_vel
                    if min(Rover.nav_dists) < 50:
                        approach_speed_limit = 1.0
                    # Approach slowly
                    if Rover.vel < approach_speed_limit:
                        # Set throttle value to throttle setting
                        Rover.throttle = Rover.throttle_set
                        Rover.brake = 0
                    else: # Else too fast approach
                        Rover.throttle = 0
                        Rover.brake = Rover.brake_set
                    # Set steering to average angle clipped to the range +/- 15
                    Rover.steer = np.clip(mean_angle, -15, 15)
            
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the terrain right in front of the camera first
            if Rover.close_obstacle:
                if Rover.vel > 0.5:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                # If we're not moving (vel < 0.5) then do something else
                elif Rover.vel <= 0.5:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line
                    # will induce 4-wheel turning
                    # Set steering to average angle clipped to the range +/- 15
                    if len(Rover.nav_angles) > 0:
                        Rover.steer = np.clip(
                            np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
                    else:
                        Rover.steer = -15
            else:
                # Check the extent of navigable terrain
                if len(Rover.nav_angles) >= Rover.stop_forward:
                    # If mode is forward, navigable terrain looks good
                    # and velocity is below max, then throttle
                    if Rover.vel < Rover.max_vel:
                        # Set throttle value to throttle setting
                        Rover.throttle = Rover.throttle_set
                    else: # Else coast
                        Rover.throttle = 0
                    Rover.brake = 0
                    # Set steering to average angle clipped to the range +/- 15
                    Rover.steer = np.clip(
                        np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
                # If there's a lack of navigable terrain pixels then go to 'stop'
                # mode
                elif len(Rover.nav_angles) < Rover.stop_forward and not returned_from_approach_mode:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
            
            if (get_angle(Rover.pitch) >= Rover.pitch_threshold \
                or get_angle(Rover.roll) >= Rover.roll_threshold) \
                and Rover.throttle != 0:
                    # Slow down a bit
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    # Rover.brake = Rover.brake_set / 6
                    print('safety break')
                

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a
                # path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line
                    # will induce 4-wheel turning
                    if len(Rover.nav_angles) > 0:
                        best_way_dir = np.mean(Rover.nav_angles * 180 / np.pi)
                        Rover.steer = (best_way_dir / abs(best_way_dir)) * 15
                    else:
                        Rover.steer = -15
                # If we're stopped but see sufficient navigable terrain in
                # front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(
                        np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
                    Rover.mode = 'forward'

        elif Rover.mode == 'stuck':
            # No throttle
            Rover.throttle = 0
            # Release the brake to allow turning
            if Rover.vel > 0.2:
                Rover.brake = Rover.brake_set
            else:
                Rover.brake = 0
            if abs((Rover.yaw - Rover.last_yaw) % 360) < 5:
                # Turn range is 90 degrees
                # will induce 4-wheel turning
                Rover.steer = -90  # Keep turning the same way

    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover
