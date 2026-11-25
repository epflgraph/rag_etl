"""The one marker staff write into names and pages: `[NO_BOT]`, to exclude.

A module, folder, file or page carrying it is not for the bots; everything
else is extracted. What the material is, the judge and the overrides decide.
"""

import re

NO_BOT = re.compile(r"\[NO_BOT\]")
