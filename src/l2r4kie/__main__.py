"""Allow ``python -m l2r4kie <command>`` as an alias of the ``l2r4kie`` script."""

import sys

from .cli import main

sys.exit(main())
