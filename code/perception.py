import numpy as np
import time
import cv2
from supporting_functions import arrays_to_image, get_angle

# Position recording frequency in seconds
pos_record_frequency = 10
# Distance range to leave meaning not getting stuck since last position was recorded
same_pos_range = 5
# Yaw range to leave meaning not getting stuck since last position was recorded (+/-)
same_yaw_range = 8

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160), rgb_max=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2]) \
                & (img[:,:,0] <= rgb_max[0]) \
                & (img[:,:,1] <= rgb_max[1]) \
                & (img[:,:,2] <= rgb_max[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def rotate_pix_reverse(xrot, yrot, yaw):
    yaw_rad = yaw * np.pi / 180
    xpix = xrot * np.cos(yaw_rad) + yrot * np.sin(yaw_rad)
    ypix = -xrot * np.sin(yaw_rad) + yrot * np.cos(yaw_rad)
    return xpix, ypix

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(np.round(xpix_tran)), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(np.round(ypix_tran)), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

def extend_point(xpix, ypix, extend=(0, 0)):
    x_extend = np.int_(np.round(extend[0]))
    y_extend = np.int_(np.round(extend[1]))
    xres = xpix[:]
    yres = ypix[:]
    for i in range(-x_extend, x_extend + 1):
        for j in range(-y_extend, y_extend + 1):
            if i != 0 or j != 0:
                xres = np.concatenate((xres, xpix + i))
                yres = np.concatenate((yres, ypix + j))

    return xres, yres

def sector_mask(shape, centre, radius, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    Credits: https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    """

    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx, y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask * anglemask

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def world_to_rover(w_map, xpos, ypos, yaw, world_size, scale, sensitivity_threshold=0):
    
    range = 25
    x_min = max(0, np.int_(round(xpos, 0)) - range)
    x_max = min(world_size, np.int_(round(xpos, 0)) + range)
    y_min = max(0, np.int_(round(ypos, 0)) - range)
    y_max = min(world_size, np.int_(round(ypos, 0)) + range)

    wm_filtered = np.zeros_like(w_map[:, :, 0])
    wm_filtered[x_min:x_max, y_min:y_max] = w_map[x_min:x_max, y_min:y_max, 2]

    w_map_mask = wm_filtered[:, :] > sensitivity_threshold
    wm_filtered[w_map_mask] = 1
    xpix_discovered, ypix_discovered = wm_filtered.nonzero()
    
    # Apply translation
    xpix_tran = (ypix_discovered - xpos) * scale
    ypix_tran = (xpix_discovered - ypos) * scale
    
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix_reverse(xpix_tran, ypix_tran, yaw)
    
    # Round and convert to int
    x_pix_rov = np.int_(np.around(xpix_rot))
    y_pix_rov = np.int_(np.around(ypix_rot))

    # Enlarge points
    x_pix_ext, y_pix_ext = extend_point(x_pix_rov, y_pix_rov, (scale / 2, scale / 2))
    # x_pix_ext, y_pix_ext = extend_point(x_pix_rov, y_pix_rov, (scale, scale))

    # Return the result
    return x_pix_ext, y_pix_ext

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0])) # mask for the range the camera can see
    
    return warped, mask

def get_rover_dim_on_world_map(pos, yaw, source, destination, world_size):
    extended_full_map_temp = np.zeros((world_size, world_size))
    extended_mask = sector_mask(
        (world_size, world_size),
        pos,
        2, # Half of the rover path width on the world map
        (yaw + 90, yaw + 270)
    )
    extended_full_map_temp[extended_mask] = 255

    return extended_full_map_temp.nonzero()

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])
    world_size = Rover.worldmap.shape[0]
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    scale = dst_size * 2
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    # The max. distance (in pixels) of the range when searching for obstacles right in front of the rover
    distance_treshold = 25
    # The max. angle (+/-) to search for close obstacles
    angle_threshold = 15
    angle_threshold_rad = angle_threshold * np.pi / 180
    # The minimum percentage of the obstacle free fields compared to the whole range
    # to ignore redirections due to close obstacles
    close_obstacle_treshold = 85
    
    # Perform perception steps to update Rover()
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed_navigable = color_thresh(warped, (160, 160, 160))
    threshed_obstacles = cv2.bitwise_not(threshed_navigable) * mask
    threshed_rocks = color_thresh(warped, (140, 110, -1), (210, 180, 80))
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    # Rover.vision_image[:,:,0] = threshed_obstacles
    # Rover.vision_image[:,:,1] = threshed_rocks
    # Rover.vision_image[:,:,2] = threshed_navigable

    Rover.vision_image = threshed_navigable * 255

    # 5) Convert map image and explored map pixel values to rover-centric coords
    xpix_navigable, ypix_navigable = rover_coords(threshed_navigable)
    xpix_rocks, ypix_rocks = rover_coords(threshed_rocks)    
    xpix_obstacles, ypix_obstacles = rover_coords(threshed_obstacles)


    # 6) Convert rover-centric pixel values to world coordinates
    x_pix_world_rocks, y_pix_world_rocks = \
        pix_to_world(xpix_rocks, ypix_rocks, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    if len(x_pix_world_rocks) > 20 and Rover.mode != 'stuck':
        Rover.mode = 'approach'

    # Only if roll and pitch is under threshold)
    if get_angle(Rover.pitch) < Rover.pitch_threshold and get_angle(Rover.roll) < Rover.roll_threshold:
    
        x_pix_world_navigable, y_pix_world_navigable = \
            pix_to_world(xpix_navigable, ypix_navigable, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        x_pix_world_obstacles, y_pix_world_obstacles = \
            pix_to_world(xpix_obstacles, ypix_obstacles, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

        # 7) Update Rover worldmap (to be displayed on right side of screen)
        Rover.worldmap[y_pix_world_obstacles, x_pix_world_obstacles, 0] += 1
        Rover.worldmap[y_pix_world_rocks, x_pix_world_rocks, 1] += 1
        Rover.worldmap[y_pix_world_navigable, x_pix_world_navigable, 2] += 1

        # Save the current position (and around) to the explored map
        x_pix_extended, y_pix_extended = \
            get_rover_dim_on_world_map(Rover.pos, Rover.yaw, source, destination, world_size)

        Rover.worldmap_explored[y_pix_extended, x_pix_extended, 2] += 1
    # else:
    #     print('Above roll/pitch threshold')

    # 8) Convert rover-centric pixel positions to polar coordinates
    rover_centric_pixel_distances, rover_centric_angles = \
        to_polar_coords(xpix_navigable, ypix_navigable)
    # Update Rover navigable pixel distances and angles
    Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_angles = rover_centric_angles

    # 9) Search for obstacles right in front of the rover
    # Create a matrix of the angles and distances
    mx_angles = np.column_stack((rover_centric_angles, rover_centric_pixel_distances))
    # Setup the filter by angle
    angle_filter = abs(mx_angles[:, 0]) < angle_threshold_rad
    # Setup the filter by distance
    distance_filter = abs(mx_angles[:, 1]) < distance_treshold
    # Filter the matrix to the really close range
    mx_filtered = mx_angles[angle_filter & distance_filter]
    # The navigable terrain right in front of the rover
    close_navigable_pixels = len(mx_filtered)
    # The size of the terrain right in front of the rover
    max_range = (distance_treshold)**2 * np.pi / 360 * angle_threshold * 2
    # The offset size of the terrain right in front of the rover
    offset_pixels = bottom_offset**2 * np.pi / 360 * angle_threshold * 2
    Rover.close_obstacle = \
        close_navigable_pixels / (max_range - offset_pixels) * 100 < close_obstacle_treshold

    # 10) Consider explored fields
    # Create a map based on the explored navigable terrain
    diff = time.time()
    x_pix_explored, y_pix_explored = world_to_rover(Rover.worldmap_explored, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale, 0)
    print('Time: ' + str(time.time() - diff))

    # Extract the explored from navigable
    if x_pix_explored.shape[0] != 0:
        if xpix_navigable.shape[0] != 0:
            x_max = np.int_(max(max(xpix_navigable), max(x_pix_explored)))
            x_min = np.int_(min(min(xpix_navigable), min(x_pix_explored)))
            y_max = np.int_(max(max(ypix_navigable), max(y_pix_explored)))
            y_min = np.int_(min(min(ypix_navigable), min(y_pix_explored)))
        else:
            x_min = np.int_(min(x_pix_explored))
            x_max = np.int_(max(x_pix_explored))
            y_min = np.int_(min(y_pix_explored))
            y_max = np.int_(max(y_pix_explored))
    else:
        if xpix_navigable.shape[0] != 0:
            x_min = np.int_(min(xpix_navigable))
            x_max = np.int_(max(xpix_navigable))
            y_min = np.int_(min(ypix_navigable))
            y_max = np.int_(max(ypix_navigable))
        else:
            x_min = 0
            x_max = 0
            y_min = 0
            y_max = 0

    mx = np.zeros((x_max - x_min + 1, y_max - y_min + 1))

    mx[np.int_(xpix_navigable) - x_min, np.int_(ypix_navigable) - y_min] = 1
    mx[np.int_(x_pix_explored) - x_min, np.int_(y_pix_explored) - y_min] = 0
    
    x_pix_free, y_pix_free = mx.nonzero()

    x_pix_free = x_pix_free + x_min
    y_pix_free = y_pix_free + y_min

    # Convert explored coordinates to polar coords
    rover_centric_pixel_distances_free, rover_centric_angles_free = \
        to_polar_coords(x_pix_free, y_pix_free)

    # Sets the minimum number of required free pixels to navigate based
    # on the extraction of the explored fields from the navigable terrain
    free_threshold = 100

    if len(rover_centric_pixel_distances_free) > free_threshold:
        # Update Rover explored pixel distances and angles
        Rover.nav_dists = rover_centric_pixel_distances_free
        Rover.nav_angles = rover_centric_angles_free

        # Display explored rover nav view on the third nav map
        Rover.vision_image = arrays_to_image(
            (threshed_navigable.shape[1], threshed_navigable.shape[0]),
            x_pix_free, # x pix
            y_pix_free, # y pix
            (0, 160 - 1), # x range
            (-160 + 1, 160 - 1), # y range
            255 # displayed value
        )
    else:
        # Display rover nav view on the third nav map
        Rover.vision_image = arrays_to_image(
            (threshed_navigable.shape[1], threshed_navigable.shape[0]),
            xpix_navigable, # x pix
            ypix_navigable, # y pix
            (0, 160 - 1), # x range
            (-160 + 1, 160 - 1), # y range
            255 # displayed value
        )
    
    if Rover.mode == 'approach':
        # Re-calibrate `close` term => could also be very close
        Rover.close_obstacle = False

        # The max. distance (in pixels) of the range when searching for obstacles right in front of the rover
        distance_treshold = 5

        # The minimum percentage of the obstacle free fields compared to the whole range
        # to ignore redirections due to close obstacles
        close_obstacle_treshold = 65

        # Setup the filter by distance
        distance_filter = abs(mx_angles[:, 1]) < distance_treshold
        # Filter the matrix to the really close range
        mx_filtered = mx_angles[angle_filter & distance_filter]
        # The navigable terrain right in front of the rover
        close_navigable_pixels = len(mx_filtered)
        # The size of the terrain right in front of the rover
        max_range = (distance_treshold)**2 * np.pi / 360 * angle_threshold * 2
        # The offset size of the terrain right in front of the rover
        offset_pixels = bottom_offset**2 * np.pi / 360 * angle_threshold * 2
        Rover.very_close_obstacle = \
            close_navigable_pixels / (max_range - offset_pixels) * 100 < close_obstacle_treshold

        rover_centric_pixel_distances_rocks, rover_centric_angles_rocks = \
            to_polar_coords(xpix_rocks, ypix_rocks)
        # Update Rover navigable pixel distances and angles
        Rover.nav_dists = rover_centric_pixel_distances_rocks
        Rover.nav_angles = rover_centric_angles_rocks

        # Display explored rover nav view on the third nav map
        Rover.vision_image = arrays_to_image(
            (threshed_navigable.shape[1], threshed_navigable.shape[0]),
            xpix_rocks, # x pix
            ypix_rocks, # y pix
            (0, 160 - 1), # x range
            (-160 + 1, 160 - 1), # y range
            255 # displayed value
        )
    
    # Check if got stuck
    if time.time() - Rover.last_rec_time > pos_record_frequency:
        
        if Rover.mode == 'stuck':
            Rover.mode = 'forward'
            # Check if the position and yaw changed over time
        elif not Rover.picking_up:
            if (abs(Rover.last_pos[0] - Rover.pos[0]) < same_pos_range \
                    and abs(Rover.last_pos[1] - Rover.pos[1]) < same_pos_range \
                    and Rover.vel <= 0.1 \
                    and abs((Rover.last_yaw - Rover.yaw) % 360) < same_yaw_range) \
                or (Rover.vel == 0 and Rover.last_vel == 0):
                # or (Rover.last_steer == Rover.steer and abs(Rover.steer) == 15):

                if (abs(Rover.last_pos[0] - Rover.pos[0]) < same_pos_range \
                    and abs(Rover.last_pos[1] - Rover.pos[1]) < same_pos_range \
                    and abs((Rover.last_yaw - Rover.yaw) % 360) < same_yaw_range):
                    print('stuck 1')
                if (Rover.vel == 0 and Rover.last_vel == 0):
                    print('stuck 2')
                # if (Rover.last_steer == Rover.steer and abs(Rover.steer) == 15):
                #     print('stuck 3')

                Rover.mode = 'stuck'

        Rover.last_pos = Rover.pos
        Rover.last_yaw = Rover.yaw
        Rover.last_steer = Rover.steer
        Rover.last_vel = Rover.vel
        Rover.last_rec_time = time.time()
    else: # Reset the last values if they have changed
        if Rover.vel > 0.2 and Rover.last_vel == 0:
            Rover.last_vel = -1
        if abs(Rover.yaw) != 15 and abs(Rover.last_yaw) == 15 and Rover.mode != 'stuck':
            Rover.last_yaw = -1

    return Rover
