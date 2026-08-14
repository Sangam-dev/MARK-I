from __future__ import annotations

# ── Failure classes ──────────────────────────────────────────────────────
NOT_FOUND = "not_found"                # target file / app / object does not exist
INVALID_ARGUMENT = "invalid_argument"  # bad arguments, invalid path, unknown action
UNSUPPORTED = "unsupported"            # operation or dependency not available
PERMISSION = "permission"              # access denied / approval required

# Classes that can never succeed by retrying the same call. The Scheduler
# skips retries for these and terminates instead of looping.
TERMINAL_ERROR_TYPES = frozenset({NOT_FOUND, INVALID_ARGUMENT, UNSUPPORTED, PERMISSION})

# Hard cap on replacement plans generated for one delegation. Belt-and-braces
# behind the identical-plan guard: even a sequence of *different* still-failing
# plans must terminate instead of looping forever.
MAX_REPLANS_PER_DELEGATION = 5

