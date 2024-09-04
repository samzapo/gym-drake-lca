# Initial Setup
## Setup PyEnv

https://github.com/pyenv/pyenv


```
pyenv install 3.10.4
pyenv global 3.10.4
```

## Setup Drake

```
https://drake.mit.edu/pip.html
```

check that the installation worked

## Setup Blender (optional, for mesh conversion)

```
snap install blender --classic
pip install bpy
```

## Setup Project

```
source env/bin/activate
pip install -e .
```

# Tests

## Just Drake
```
python3 gym_drake_lca/lca_passive_simulation.py
```

## DrakeGymEnv
```
python3 gym_drake_lca/envs/test_lift_cube_env.py
```

## DrakeGymEnv + Gym
```
python3 gym_drake_lca/envs/test_env.py
```