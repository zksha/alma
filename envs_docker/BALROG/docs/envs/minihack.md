# MiniHack

MiniHack is a powerful sandbox framework built
on top of the NLE that enables researchers to
easily design rich and diverse environments for RL. It provides a
flexible platform for creating custom RL tasks ranging from simple
grid-world navigation to complex, procedurally generated worlds with
intricate game mechanics. The framework allows users to define
environments using a human-readable description language or a simple
Python interface, giving fine-grained control over environment elements
such as terrain, objects, monsters, and traps. MiniHack offers a diverse
array of tasks, which can be broadly categorized into three main groups:
Navigation Tasks, Skill Acquisition Tasks, and Ported Tasks. To enable
interaction with language models, we use the NetHack Language Wrapper.

From the MiniHack Navigation Tasks, we picked Maze 9x9, Maze 15x15, Corridor and
CorridorBattle, which challenge the agent to reach the goal position by
overcoming various difficulties on their way, such as fighting monsters
in corridors and navigating through complex or procedurally generated
mazes. These tasks feature a relatively small action space, i.e.,
movement towards 8 compass directions, and based on the environment,
search, kick, open, and eat actions.

![Examples of MiniHack Corridor
task.](../imgs/minihack_corridor.png)

![Example of MiniHack CorridorBattle
task.](../imgs/minihack_corridor_battle.png)

From the MiniHack Skill Acquisition Tasks, we picked Quest (with two
different difficulty levels, Easy, Medium), which challenges
the agent to use objects found in the environment to cross a lava river
(these objects can provide levitation or freezing abilities), fight
monsters.

We additionally test the agents on MiniHack Boxoban. This family of
environments is an adaptation of the Boxoban puzzle game, which itself
is inspired by the classic Sokoban. These environments present a
challenging puzzle-solving task within the MiniHack framework,
leveraging the NetHack game mechanics. The primary goal in MiniHack
Boxoban is to push four boulders (MiniHack's equivalent of boxes) onto
four designated goal locations, which are represented by fountains. This
task requires strategic thinking and planning, as the agent must
carefully maneuver the boulders through the environment without getting
them stuck in corners or against walls.

![Example of MiniHack Boxoban Hard
task.](../imgs/minihack_boxoban.png)
