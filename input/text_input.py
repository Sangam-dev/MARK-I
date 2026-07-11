from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.bus import EventBus
from core.events import ShutdownRequested, TextInputReceived


@dataclass(slots=True)
class TextInputHandler:
	bus: EventBus
	session_id: str = "default"

	async def run(self) -> None:
		try:
			while True:
				line = await self._read_line()
				if line is None:
					await self.bus.emit_and_wait(ShutdownRequested(reason="EOF"))
					return

				text = line.strip()
				if not text:
					continue

				if text.lower() in {"exit", "quit"}:
					await self.bus.emit_and_wait(ShutdownRequested(reason="user requested"))
					return

				await self.bus.emit_and_wait(TextInputReceived(text=text, session_id=self.session_id))
		except KeyboardInterrupt:
			await self.bus.emit_and_wait(ShutdownRequested(reason="keyboard interrupt"))

	async def _read_line(self) -> str | None:
		loop = asyncio.get_running_loop()
		try:
			return await loop.run_in_executor(None, input, "> ")
		except EOFError:
			return None
