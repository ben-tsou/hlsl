import roboschool, gym
import os
print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))

curpath = os.path.abspath(os.curdir)
print("Current path is: %s" % (curpath))