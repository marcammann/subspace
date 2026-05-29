from conftest import function_call_backend_events, text_backend_events

from subspace.middleware.utils import StreamTracker
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDoneEvent,
)


class TestStreamTracker:
    def test_close_text_message(self):
        tracker = StreamTracker()
        events = text_backend_events()
        for e in events[:6]:
            tracker.track(e)

        close_events = tracker.close_open_items()
        types = [type(e) for e in close_events]
        assert ResponseOutputTextDoneEvent in types
        assert ResponseContentPartDoneEvent in types
        assert ResponseOutputItemDoneEvent in types

    def test_close_function_call(self):
        tracker = StreamTracker()
        events = function_call_backend_events()
        for e in events[:5]:
            tracker.track(e)

        close_events = tracker.close_open_items()
        types = [type(e) for e in close_events]
        assert ResponseFunctionCallArgsDoneEvent in types
        assert ResponseOutputItemDoneEvent in types

    def test_close_when_idle(self):
        tracker = StreamTracker()
        assert tracker.close_open_items() == []

    def test_close_after_item_done(self):
        tracker = StreamTracker()
        for e in text_backend_events():
            tracker.track(e)
        assert tracker.close_open_items() == []

    def test_completed_items_tracked(self):
        tracker = StreamTracker()
        for e in text_backend_events():
            if isinstance(e, ResponseCompletedEvent):
                break
            tracker.track(e)
        assert len(tracker.completed_items) == 1

    def test_response_tracked(self):
        tracker = StreamTracker()
        events = text_backend_events()
        tracker.track(events[0])
        assert tracker.response is not None
