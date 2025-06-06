import carla

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
m = world.get_map()

world = client.load_world("Town01")

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Generate a batch of random points on sidewalks
spawn_points = [world.get_random_location_from_navigation() for _ in range(300)]

# Sort points by point.x and point.y
spawn_points = sorted(spawn_points, key=lambda point: (-point.x, -point.y))

# Print and draw the sorted points
for i, spawn_point in enumerate(spawn_points):
    print(i, spawn_point)
    world.debug.draw_string(spawn_point, str(i), life_time=100)

# Initialize spectator
spectator = world.get_spectator()

spectator.set_transform(carla.Transform(carla.Location(z=300, x=50, y= 50), carla.Rotation(yaw=0, pitch=-90)))

# Main loop
while True:
    try:
        # Get user input
        user_input = input("Enter the viewpoint number (0-299), or 'q' to quit: ")
        
        # If user inputs 'q', exit the program
        if user_input.lower() == 'q':
            print("Exiting the program.")
            break
        
        # Convert input to integer
        spectator_point = int(user_input)
        
        # Check if the input is within the valid range
        if 0 <= spectator_point < len(spawn_points):
            # Update spectator's location
            spectator_location = carla.Location(z=30, x=spawn_points[spectator_point].x, y=spawn_points[spectator_point].y)
            bv_transform = carla.Transform(spectator_location, carla.Rotation(yaw=0, pitch=-90))
            spectator.set_transform(bv_transform)
            print(f"Switched to viewpoint {spectator_point}.")
        else:
            print("Invalid number. Please enter a number between 0 and 299.")
    
    except ValueError:
        print("Invalid input. Please enter a number or 'q' to quit.")
    
    # Wait for the next frame
    world.wait_for_tick()