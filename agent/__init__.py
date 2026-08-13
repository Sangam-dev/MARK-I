"""OpenCode execution layer — delegation for coding, research and
multi-step work that the assistant's own tools cannot do.

The assistant's action system handles bounded, single-shot operations:
open an app, read an email, set an alarm. Some requests are not that
shape — "build a REST API for a library management system", "research
current RAG techniques and compare them", "find the performance problems
in this project, fix them and run the tests". Those need a working
directory, many steps, and the ability to revise earlier work.

This package delegates that class of task to OpenCode. Four files, one
concern each::

    config.py   every tunable, sourced from the environment
    client.py   the HTTP transport and the server's lifecycle
    progress.py what a run is doing right now, folded from its events
    tool.py     the structured action surface + session tracking

Delegated work runs in the background. ``delegate`` returns as soon as
the agent is under way, so the user can keep talking while a build takes
its several minutes, ask "how's it going" and get real counts, and be
told when it finishes. See :mod:`agent.tool` for how the run task and
the event pump fit together.

The layering rule is the one the rest of the project already uses: the
LLM talks to ``tool.py``, ``tool.py`` talks to ``client.py``, and
``client.py`` talks to OpenCode. Nothing in this package imports from
``actions/`` — OpenCode is isolated from the assistant's tools, and the
assistant does not become OpenCode's front end. The single seam is the
``agent_task`` entry in :mod:`tasks.registry`, which is the only way
either LLM learns the capability exists.
"""

from agent.client import (
    OpenCodeClient,
    OpenCodeResult,
    close_shared_opencode,
    get_shared_opencode_client,
    set_shared_opencode_client,
)
from agent.config import OpenCodeConfig
from agent.progress import RunProgress, ToolStep
from agent.tool import (
    ACTIONS,
    AgentToolResult,
    DelegatedSession,
    OpenCodeTool,
    describe_actions,
    get_shared_opencode_tool,
    set_shared_opencode_tool,
)

__all__ = [
    "ACTIONS",
    "AgentToolResult",
    "DelegatedSession",
    "OpenCodeClient",
    "OpenCodeConfig",
    "OpenCodeResult",
    "OpenCodeTool",
    "RunProgress",
    "ToolStep",
    "close_shared_opencode",
    "describe_actions",
    "get_shared_opencode_client",
    "get_shared_opencode_tool",
    "set_shared_opencode_client",
    "set_shared_opencode_tool",
]
