# use extension corresponding to sd webui version
# since it is difficult for sampler scheduler extension for v1.6.x to be backward compatible.

with open('CHANGELOG.md', 'r') as md:
    first_line = next(md)

version = first_line.replace('## 1.', '')[0]

if int(version) >= 6:
    from Seniorious_16 import *
else:
    from Seniorious import *