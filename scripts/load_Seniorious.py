# import extension script corresponding to sd webui version
# since it is difficult for sampler scheduler extension for v1.6.x to be backward compatible.

from modules import launch_utils

version = launch_utils.git_tag()

if version.startswith('1.'):
    v = int(version[2])
elif version.startswith('v1.'):
    v = int(version[3])
else:
    raise RuntimeError(f"WebUI Version cannot be found")

if v == 6:
    from Seniorious_16 import *
elif v <= 5:
    from Seniorious import *
else:
    raise RuntimeError(f"Unsupported WebUI Version")