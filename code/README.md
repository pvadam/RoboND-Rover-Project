# Solution of Search and Sample Return by Peter ADAM

Screen resolution: 1280x800
Graphics quality: good
FPS: 14-35 (better at the beginning)

## Recognizing obstacles and rock samples

* Obstacles

Identifying obstacles was easy by simply negating the navigable terrain. A mask filter was also added to the selection to ignore the fields on the image outside the camera view angle.

```py
threshed_obstacles = cv2.bitwise_not(threshed_navigable) * mask
```

* Rocks

I was trying to find a threshold to recognize rocks, but I could only do so by setting up a minimum and maximum range, therefore I have extended the `color_thresh` function with an optional maximum color range parameter (`rgb_max`) which defaults to white color:

```py
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
```

## Customizing the `process_image()` function

1. Setup
First step was setting up the relation between the rover camera and the world map and loading the current position in addition to reading the current position from the rover:

```py
world_size = data.worldmap.shape[0]
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

xpos = data.xpos[data.count]
ypos = data.ypos[data.count]
yaw = data.yaw[data.count]

# 1) Define source and destination points for perspective transform 
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                ])
```

2. Applying perspective transform

A mask was used (copied from the solution video) to ignore the range in the camera image which is outside the view.

```py
warped, mask = perspect_transform(img, source, destination)
```

3. Separate the parts of the image by color ranges

Using the updated function described above:

```py
threshed_navigable = color_thresh(warped, (160, 160, 160))
threshed_rocks = color_thresh(warped, (140, 110, -1), (210, 180, 80))
threshed_obstacles = cv2.bitwise_not(threshed_navigable)
```

4. Convert thresholded image pixel values to rover-centric coords

The same functions which were introduced in the lessons for finding non-zero pixels and converting the pixel positions with reference to the center bottom of the image.

```py
# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

xpix_navigable, ypix_navigable = rover_coords(threshed_navigable)
xpix_rocks, ypix_rocks = rover_coords(threshed_rocks)
xpix_obstacles, ypix_obstacles = rover_coords(threshed_obstacles)
```

5. Convert rover-centric pixel values to world coords

The navigable fields, obstacles and rock samples are converted to world coordinates separately by the same functions which were introduced in the lessons for scaling, translation and rotation.

```py
# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

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
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

x_pix_world_navigable, y_pix_world_navigable = pix_to_world(xpix_navigable, ypix_navigable, xpos, ypos, yaw, world_size, scale)
x_pix_world_rocks, y_pix_world_rocks = pix_to_world(xpix_rocks, ypix_rocks, xpos, ypos, yaw, world_size, scale)
x_pix_world_obstacles, y_pix_world_obstacles = pix_to_world(xpix_obstacles, ypix_obstacles, xpos, ypos, yaw, world_size, scale)
```

5. Update worldmap (to be displayed on right side of screen)

Set the identified pixels in the right dimensions.

```py
data.worldmap[y_pix_world_obstacles, x_pix_world_obstacles, 0] += 1
data.worldmap[y_pix_world_rocks, x_pix_world_rocks, 1] += 10
data.worldmap[y_pix_world_navigable, x_pix_world_navigable, 2] += 1
```

6. Video output

See [here](../output/test_mapping.mp4).

## Autonomous Navigation and Mapping

### Perception

The key changes beside the ones explained above

1. A constant was set up to record the data from rover to the world map only if the pith and roll angles are below this particular threshold.

```py
# Only if roll and pitch is under threshold)
    if get_angle(Rover.pitch) < Rover.pitch_threshold and get_angle(Rover.roll) < Rover.roll_threshold:
        ...
```

2. New modes:

* approach
When a rock sample found and trying to approach it for a pickup. This mode is activated if the number of pixels identified as rock samples are above a certain limit.

```py
# 6) Convert rover-centric pixel values to world coordinates
x_pix_world_rocks, y_pix_world_rocks = \
    pix_to_world(xpix_rocks, ypix_rocks, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

if len(x_pix_world_rocks) > 20 and Rover.mode != 'stuck':
    Rover.mode = 'approach
```

* stuck
Activated when the rover got stuck. It is checked at a certain frequency (parameter, currently 10 seconds) if the rover is still in the same position (with a little threshold) or its orientation was not changed. If its velocity or orientation significaly changed during this period, the values are reset.

```py
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
    if abs(Rover.yaw) != 15 and abs(Rover.last_yaw) == 15:
        Rover.last_yaw = -1
```

3. Storing the path already taken

To avoid taking the same path on the same part of a navigable route, I am recording the current path on a map with the same size as the world map to avoid saving too many pixels for this purpose. I am only recording the backside of the rover. For this purpose I create a circle around the current position and cut the area before the rover. Otherwise the rover would think on the next image that it has already been there.

```py
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

x_pix_extended, y_pix_extended = \
    get_rover_dim_on_world_map(Rover.pos, Rover.yaw, source, destination, world_size)

Rover.worldmap_explored[y_pix_extended, x_pix_extended, 2] += 1
```

When using the explored world map, the path is enlarged and translated back to the rover view to be able to compare with the processed image from the rover camera.

```py
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
```

4. Close obstacles

The algorithm keeps analysing the terrain right in front of the rover. The angles and distances are merged into a matrix for easier filtering on both values at the same time. The pixels close to the rover (in a certain view angle) is filtered first, than they are analized weather most part of it is navigable or not. If the result is below the threshold, the rover's `close_obstacle` property is set.

The distance threshold is lowered while in `approach` mode.

```py
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
```

5. The polar coordinates and the third image in the video

The decision is mostly made based on the polar coordinates (see it later), but the source of the polar coordinates are keep changing. Three sources are possible to get it from (in the order of higher importance):

* Rock samples
If resonable number of pixels are identified as rock samples on the image from the rover's camera, the base of the navigation is coming from the polar coordinates of the rock sample pixels.

* Non-explored navigable
If no rock samples are visible, the navigable terrain is returned which was not yet taken by the rover.

* Navigable
If no rock samples visible and the non-explored navigable map returns too few useful pixels, the navigable coordinates are used.

### Decision

The currently selected image for the decision making can always be seen in the third map from the rover point of view. The possible options:
* rock sample ahead
* navigable, but not yet taken terrain
* navigable terrain

1. Forward mode
In case an obstacle gets very close to the front of the rover, it stops and steers left or right (to the better choice). Another change I have made is a safety breaking (actually not accellerating anymore) if the pitch or roll values are above the threshold where the image data is not recording to the world map.

2. Stop mode
My only change in this mode was to turn to the right direction instead of the fixed -15 degree.

3. Approach mode
If a rock sample is in the view of the camera, the rover will try to go closer slowly. If there is an obstacle very close, it switches to forward mode to be able to avoid it.

Below a certain distance, it slows down even more. If it looses the rock sample from sight, it switches to forward mode again.

If the rover got close enough to the rock sample it picks it up and returns to forward mode.

4. Stuck mode
The rover is stopped and tries to turn until turning -90 degrees.
