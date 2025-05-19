import carla
import sys

carla_root = "/home/zmh/carla/PythonAPI/carla/"
sys.path.append(carla_root)

print(sys.path)


from agents.navigation.global_route_planner import GlobalRoutePlanner


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
m = world.get_map()

spawn_points = [world.get_random_location_from_navigation() for _ in range(300)]

origin = carla.Location(spawn_points[294])
destination = carla.Location(spawn_points[295])  

distance = 2
grp = GlobalRoutePlanner(m, distance)
route = grp.trace_route(origin, destination)

T = 100
for pi, pj in zip(route[:-1], route[1:]):
    pi_location = pi[0].transform.location
    pj_location = pj[0].transform.location 
    pi_location.z = 0.5
    pj_location.z = 0.5
    world.debug.draw_line(pi_location, pj_location, thickness=0.2, life_time=T, color=carla.Color(r=255))
    pi_location.z = 0.6
    world.debug.draw_point(pi_location, color=carla.Color(r=255), life_time=T)

