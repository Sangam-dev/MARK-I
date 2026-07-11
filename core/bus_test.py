import asyncio
from .bus import EventBus
from .events import TranscriptReady, TextInputReceived
from core import bus

async def main():
    bus = EventBus()
    received = []

    # Test 1: basic pub/sub
    async def on_transcript(e: TranscriptReady):
        received.append(e.text)

    bus.subscribe(TranscriptReady, on_transcript)
    bus.emit(TranscriptReady(text='hello'))
    await asyncio.sleep(0.05)
    assert received == ['hello'], f'Expected hello, got {received}'
    print('pub/sub ok')

    # Test 2: multiple handlers
    called = []
    async def handler_a(e): called.append('a')
    async def handler_b(e): called.append('b')

    bus.subscribe(TextInputReceived, handler_a)
    bus.subscribe(TextInputReceived, handler_b)
    bus.emit(TextInputReceived(text='test'))
    await asyncio.sleep(0.05)
    assert sorted(called) == ['a', 'b']
    print('multiple handlers ok')

    # Test 3: crash isolation
    crash_log = []
    bus2 = EventBus()

    async def bad(e): raise RuntimeError('boom')
    async def good(e): crash_log.append('good ran')

    bus2.subscribe(TextInputReceived, bad)
    bus2.subscribe(TextInputReceived, good)
    bus2.emit(TextInputReceived(text='trigger'))
    await asyncio.sleep(0.1)
    assert 'good ran' in crash_log
    print('crash isolation ok')

    # Test 4: no handlers — no crash
    bus3 = EventBus()
    bus3.emit(TranscriptReady(text='nobody listening'))
    print('no handlers ok')

    print('bus.py verified')

    def sync_handler(e): pass
    try:
        bus.subscribe(TranscriptReady, sync_handler)
        assert False, "should have raised"
    except TypeError:
        print("sync handler rejected ok")
    
    bus4 = EventBus()
    async def quick(e): pass
    bus4.subscribe(TranscriptReady, quick)
    bus4.emit(TranscriptReady(text="test"))
    await bus4.drain()
    assert len(bus4._tasks) == 0
    print("task cleanup ok")

asyncio.run(main())