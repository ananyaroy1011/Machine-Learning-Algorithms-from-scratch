# MENACE - Matchbox Educable Noughts and Crosses Engine (Reinforcement Learning)

This is a Python implementation of MENACE - A collection of matchboxes that learn how to play tic-tac-toe


[Read More about MENACE here](https://en.wikipedia.org/wiki/Matchbox_Educable_Noughts_and_Crosses_Engine)

## Training
* Step I: Train MENACE
  * To train MENACE against a Random opponent or another MENACE:
```
game_manager.py --mode train --trainer random --episodes 10000
game_manager.py --mode train --trainer menace --episodes 10000
```

* Step II: Play against the Trained MENACE:
```
game_manager.py --mode play
```

The trained weights/matchboxes are stored as a JSON file in the *weights* folder

*(Remember to train MENACE before you play!)*
