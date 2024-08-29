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

# Running

```
source env/bin/activate
python3 gym_drake_lca/lca_passive_simulation.py
```