TextWorld
=========


TextWorld is a text-based game environment
developed by Microsoft Research that allows for the creation and
customization of interactive fiction games. In our experiments, we
utilize three specific games from the TextWorld domain: "Treasure
Hunter\", "The Cooking Game\", and "Coin Collector\". Each task can be
generated with different levels of difficulty by changing number of
rooms, enabling obstacles and including distractor rooms.

## Tasks

### Treasure Hunter

In Treasure Hunter, we create a challenging maze-like environment with
20 rooms. The game is set to the maximum difficulty level of 30,
introducing locked doors and containers that must be manipulated to
locate the target object. To increase complexity, we remove the solution
description and filter out tasks that can be solved optimally in 20
steps or fewer. This setup requires the agent to navigate a complex
space, interact with various objects, and devise strategies to overcome
obstacles in its quest to find the treasure.

### The Cooking Game

The Cooking Game presents a culinary challenge set across 13 rooms. We
maximize the complexity by including up to 5 ingredients and enabling
all additional challenging options. The agent must navigate through
doors, process food items using tools like knives, and cook ingredients
using various methods such as grilling, frying, and roasting. This game
tests the agent's ability to plan and execute multi-step processes in a
dynamic environment, simulating the complexities of real-world cooking
tasks.

### Coin Collector

Coin Collector features an expansive environment with 40 rooms,
including potential distractor rooms to increase navigation difficulty.
Similar to Treasure Hunter, we remove the solution description to
enhance the challenge. The optimal path from the agent's starting point
to the target is set to 20 steps, requiring efficient exploration and
decision-making. This game tests the agent's ability to navigate large
spaces, avoid distractions, and efficiently reach its goal in a complex,
maze-like structure.

![TextWorld interface along with
visualization.](../imgs/textworld_map.png)