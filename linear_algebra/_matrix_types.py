from collections.abc import Callable
from typing import Any

Row = list[Any]
Rows = list[Row]
ScalarParser = Callable[[str], Any]
