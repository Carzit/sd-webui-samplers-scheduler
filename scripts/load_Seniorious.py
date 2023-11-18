# use extension corresponding to sd webui version
# since it is difficult for sampler scheduler extension for v1.6.x to be backward compatible.

import os

filename = 'CHANGELOG.md'

# I make a trick here. I use the CHANGELOG.md to get the webui version and if it is deleted, error will occur.

if os.path.exists(filename):
    with open(filename, 'r') as md:
        first_line = next(md)
        version = first_line.replace('## 1.', '')[0]
    
else:
    print(f"{filename} does not exist...")

if int(version) >= 6:
    from Seniorious_16 import *
else:
    from Seniorious import *
