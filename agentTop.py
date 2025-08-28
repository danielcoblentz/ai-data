# agentTop.py - Top Layer
# AIFCA Python code Version 0.9.17 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable 
from agentMiddle import Rob_middle_layer
from agents import Agent, Environment
from agentEnv import Rob_body, World
import math

class Rob_top_layer(Agent, Environment):
    """Original top controller (visits locations in order of y-coordinate)."""
    def __init__(self, lower, world, timeout=200 ):
        self.lower = lower
        self.world = world
        self.timeout = timeout  

    def do(self,plan):
        """carry out actions.
        actions is of the form {'visit':list_of_locations}
        It visits the locations in turn (sorted by y-coordinate).
        """
        to_do = sorted(plan['visit'], key=lambda x: self.world.locations[x][1])
        for loc in to_do:
            position = self.world.locations[loc]
            arrived = self.lower.do({'go_to':position, 'timeout':self.timeout})
            self.display(1,"Goal",loc,arrived)


class Opportunistic_top_layer(Rob_top_layer):
    """Opportunistic controller (part a): 
    always go to the nearest unvisited location next.
    """
    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def do(self, plan):
        unvisited = plan['visit'][:]
        current_position = self.world.agent_pos  # get agent’s current position from world

        while unvisited:
            # choose nearest location
            closest = min(
                unvisited,
                key=lambda loc: self.distance(current_position, self.world.locations[loc])
            )
            unvisited.remove(closest)

            # move there
            position = self.world.locations[closest]
            arrived = self.lower.do({'go_to': position, 'timeout': self.timeout})
            self.display(1, "Goal", closest, arrived)

            # update current position
            current_position = self.world.agent_pos


def rob_ex(use_opportunistic=False):
    """Example setup function.
    Pass use_opportunistic=True to test opportunistic controller.
    """
    global world, body, middle, top
    world = World(
        walls = {((20,0),(30,20)), ((70,-5),(70,25))},
        locations = {
            'mail': (-5,10),
            'o103': (50,10),
            'o109': (100,10),
            'storage': (101,51)
        }
    )
    body = Rob_body(world)
    middle = Rob_middle_layer(body)
    if use_opportunistic:
        top = Opportunistic_top_layer(middle, world)
    else:
        top = Rob_top_layer(middle, world)

    top.do({'visit':['o109','storage','o109','o103']})


if __name__ == "__main__":
    rob_ex()
    print("Try: rob_ex(use_opportunistic=True) for opportunistic behavior")
    print("Or: top.do({'visit':['o109','storage','o109','o103']})")


# Robot Trap for which the current controller cannot escape:
def robot_trap():
    global trap_world, trap_body, trap_middle, trap_top
    trap_world = World(
        {((10, 51), (60, 51)), ((30, 10), (30, 20)),
         ((10, -1), (10, 20)), ((10, 30), (10, 51)),
         ((30, 30), (30, 40)), ((10, -1), (60, -1)),
         ((10, 30), (30, 30)), ((10, 20), (30, 20)),
         ((60, -1), (60, 51))},
        locations={'goal':(90,25)}
    )
    trap_body = Rob_body(trap_world,init_pos=(0,25), init_dir=90)
    trap_middle = Rob_middle_layer(trap_body)
    trap_top = Rob_top_layer(trap_middle, trap_world)
