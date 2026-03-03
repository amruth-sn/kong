"""Tests for the work queue with call-graph ordering."""

from __future__ import annotations

from kong.agent.queue import WorkItem, WorkQueue
from kong.ghidra.types import FunctionClassification, FunctionInfo


def _func(addr: int, name: str, size: int = 100, classification: FunctionClassification = FunctionClassification.MEDIUM) -> FunctionInfo:
    return FunctionInfo(address=addr, name=name, size=size, classification=classification)


class TestWorkQueueBasic:
    def test_empty_queue(self):
        q = WorkQueue()
        assert len(q) == 0
        assert not q
        assert q.next() is None

    def test_build_simple(self):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b"), _func(0x3000, "c")]
        q = WorkQueue()
        q.build(funcs, callers_map={}, callees_map={})
        assert q.total == 3
        assert len(q) == 3

    def test_next_consumes(self):
        funcs = [_func(0x1000, "a")]
        q = WorkQueue()
        q.build(funcs, callers_map={}, callees_map={})

        item = q.next()
        assert item is not None
        assert item.address == 0x1000
        assert len(q) == 0
        assert q.next() is None

    def test_skips_imported_and_thunks(self):
        funcs = [
            _func(0x1000, "a", classification=FunctionClassification.IMPORTED),
            _func(0x2000, "b", classification=FunctionClassification.THUNK),
            _func(0x3000, "c", classification=FunctionClassification.MEDIUM),
        ]
        q = WorkQueue()
        q.build(funcs, callers_map={}, callees_map={})
        assert q.total == 1
        item = q.next()
        assert item.address == 0x3000

    def test_get_by_address(self):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b")]
        q = WorkQueue()
        q.build(funcs, callers_map={}, callees_map={})

        item = q.get_by_address(0x2000)
        assert item is not None
        assert item.function.name == "b"
        assert q.get_by_address(0x9999) is None


class TestWorkQueueOrdering:
    def test_leaves_before_callers(self):
        """Leaf functions (no callees) should come before their callers."""
        # a calls b, b calls c. c is leaf -> c first, then b, then a.
        funcs = [
            _func(0x1000, "a", size=100),
            _func(0x2000, "b", size=100),
            _func(0x3000, "c", size=100),
        ]
        callees_map = {
            0x1000: [0x2000],  # a -> b
            0x2000: [0x3000],  # b -> c
            0x3000: [],        # c is leaf
        }
        callers_map = {
            0x1000: [],
            0x2000: [0x1000],
            0x3000: [0x2000],
        }

        q = WorkQueue()
        q.build(funcs, callers_map, callees_map)

        items = []
        while q:
            items.append(q.next())

        names = [i.function.name for i in items]
        assert names == ["c", "b", "a"]

    def test_same_depth_smaller_first(self):
        """At the same call graph depth, smaller functions come first."""
        funcs = [
            _func(0x1000, "big", size=500),
            _func(0x2000, "small", size=50),
            _func(0x3000, "medium", size=200),
        ]
        q = WorkQueue()
        q.build(funcs, callers_map={}, callees_map={})

        items = []
        while q:
            items.append(q.next())

        names = [i.function.name for i in items]
        assert names == ["small", "medium", "big"]

    def test_diamond_call_graph(self):
        """Diamond: a -> b, a -> c, b -> d, c -> d. d is deepest leaf."""
        funcs = [
            _func(0x1000, "a", size=100),
            _func(0x2000, "b", size=100),
            _func(0x3000, "c", size=100),
            _func(0x4000, "d", size=100),
        ]
        callees_map = {
            0x1000: [0x2000, 0x3000],
            0x2000: [0x4000],
            0x3000: [0x4000],
            0x4000: [],
        }
        callers_map = {
            0x1000: [],
            0x2000: [0x1000],
            0x3000: [0x1000],
            0x4000: [0x2000, 0x3000],
        }

        q = WorkQueue()
        q.build(funcs, callers_map, callees_map)

        items = []
        while q:
            items.append(q.next())

        # d (depth 0) -> b, c (depth 1) -> a (depth 2)
        assert items[0].function.name == "d"
        assert items[-1].function.name == "a"

    def test_work_item_stores_callers_callees(self):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b")]
        callees_map = {0x1000: [0x2000], 0x2000: []}
        callers_map = {0x1000: [], 0x2000: [0x1000]}

        q = WorkQueue()
        q.build(funcs, callers_map, callees_map)

        b_item = q.get_by_address(0x2000)
        assert 0x1000 in b_item.callers

        a_item = q.get_by_address(0x1000)
        assert 0x2000 in a_item.callees


class TestWorkQueueProgress:
    def test_completed_tracking(self):
        funcs = [_func(0x1000, "a"), _func(0x2000, "b"), _func(0x3000, "c")]
        q = WorkQueue()
        q.build(funcs, callers_map={}, callees_map={})

        assert q.completed == 0
        assert q.remaining == 3
        assert q.total == 3

        q.next()
        assert q.completed == 1
        assert q.remaining == 2

        q.next()
        q.next()
        assert q.completed == 3
        assert q.remaining == 0

    def test_peek_doesnt_consume(self):
        funcs = [_func(0x1000, "a")]
        q = WorkQueue()
        q.build(funcs, callers_map={}, callees_map={})

        item = q.peek()
        assert item is not None
        assert q.remaining == 1

        item2 = q.peek()
        assert item2.address == item.address
        assert q.remaining == 1
