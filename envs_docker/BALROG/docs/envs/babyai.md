# Baby AI

BabyAI is a research platform designed to study
grounded language learning and instruction following in artificial
agents. It consists of a suite of 2D grid world environments with
increasing levels of complexity. In these environments, an agent
navigates through rooms and interacts with various objects like doors,
keys, balls, and boxes of different colors. The agent receives natural
language instructions, called "missions\", which describe tasks it needs
to complete, such as picking up specific objects or navigating to
certain locations. Many existing works on decision-making have studied
model performance on this environment.
We use it as a historically relevant environment that we expect to be
relatively easy to solve.

## BabyAI-Text

We evaluate the agents on 5 tasks introduced in
BabyAI-Text, which provides a description of each
observation instead of a symbolic representation. A textual description
consists of a list of template descriptions with the following
structure:

-   "You see a `<object>` `<location>`" if the object is a key, a ball,
    a box or a wall.

-   "You see a(n) open/closed door `<location>`" , if the agent sees a
    door.

-   "You carry a `<object>`", if the agent carries an object.