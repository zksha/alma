# Baba Is AI

Baba Is AI is a benchmark environment based on the puzzle game \"Baba Is
You\". In this gridworld game, players interact with various objects and
textual rule blocks to achieve specific goals. The unique aspect of Baba
Is AI is that the rules of the game can be manipulated and rearranged by
the player, creating a dynamic environment where agents must identify
relevant objects and rules and then manipulate them to change or create
new rules to succeed. This benchmark allows researchers to explore a
broader notion of generalization compared to current benchmarks, as it
requires agents to not only learn and follow the rules but also to
combine previously seen rules in novel ways. Agents are tested on 40
different puzzle levels.

![One of the Baba Is AI puzzles, where the agent has to break the "wall
is stop\" rule, create new rule "door is win\" and go to green door to
solve the task.](../imgs/babaisai_map.png)


## Baba Is AI Language Wrapper

To enable interaction with language models, we made a custom language
wrapper for Baba Is AI. It constructs language observation from active
rules and creates a description by formatting object positions relative
to the player. We don't provide the solution for the agent and don't
specify grid boundaries in the text-only experiments.