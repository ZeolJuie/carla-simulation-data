# actors/walker.py

import carla

import random

class Walker:
    def __init__(self, world, blueprint_library, spawn_point):
        self.world = world
        self.walker = self._setup_walker(blueprint_library, spawn_point)
        self.controller = self._setup_controller()

    def _setup_walker(self, blueprint_library, spawn_point):
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        walker = self.world.spawn_actor(walker_bp, spawn_point)
        return walker

    def _setup_controller(self):
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        controller = self.world.spawn_actor(controller_bp, carla.Transform(), self.walker)
        controller.start()
        return controller

    def set_target_location(self, target_location):

        self.controller.go_to_location(target_location)

    def get_location(self):
        return self.walker.get_location()

    def get_transform(self):
        return self.walker.get_transform()

    def destroy(self):
        if self.controller:
            self.controller.stop()
            self.controller.destroy()
        if self.walker:
            self.walker.destroy()