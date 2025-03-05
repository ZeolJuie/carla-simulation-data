import carla

# 连接到CARLA服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()


# We need to save the settings to be able to recover them at the end
# of the script to leave the server in the same state that we found it.
original_settings = world.get_settings()
settings = world.get_settings()

# We set CARLA syncronous mode
settings.fixed_delta_seconds = 0.05
settings.synchronous_mode = True
world.apply_settings(settings)

# 获取所有行人
walkers = world.get_actors().filter('walker.*')

# 打印所有行人的ID
for walker in walkers:
    print(f"Walker ID: {walker.id}, Type: {walker.type_id}")


world.apply_settings(original_settings)
