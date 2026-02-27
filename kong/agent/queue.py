"""Work queue with call-graph-aware ordering for bottom-up analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from kong.ghidra.types import FunctionClassification, FunctionInfo


@dataclass
class WorkItem:
    function: FunctionInfo
    depth: int = 0  # call graph depth (0 = leaf)
    callers: list[int] = field(default_factory=list)
    callees: list[int] = field(default_factory=list)
    priority: int = 0  # lower = processed first

    @property
    def address(self) -> int:
        return self.function.address


class WorkQueue:
    """Priority queue ordered bottom-up by call graph depth.

    Leaf functions (no callees) are processed first, then their callers,
    etc. Within the same depth, smaller functions come first.
    """

    def __init__(self) -> None:
        self._items: list[WorkItem] = []
        self._index: int = 0
        self._by_address: dict[int, WorkItem] = {}

    def build(
        self,
        functions: list[FunctionInfo],
        callers_map: dict[int, list[int]],
        callees_map: dict[int, list[int]],
        skip_classifications: set[FunctionClassification] | None = None,
    ) -> None:
        """Build the work queue from functions and call graph edges.

        Args:
            functions: All functions in the binary.
            callers_map: addr -> list of caller addrs.
            callees_map: addr -> list of callee addrs.
            skip_classifications: Function types to exclude (e.g., IMPORTED, THUNK).
        """
        if skip_classifications is None:
            skip_classifications = {
                FunctionClassification.IMPORTED,
                FunctionClassification.THUNK,
            }

        analyzable = {
            f.address: f
            for f in functions
            if f.classification not in skip_classifications
        }

        depths = self._compute_depths(analyzable, callees_map)

        items = []
        for addr, func in analyzable.items():
            depth = depths.get(addr, 0)
            item = WorkItem(
                function=func,
                depth=depth,
                callers=callers_map.get(addr, []),
                callees=callees_map.get(addr, []),
                # Priority: depth first (bottom-up), then size (smaller first)
                priority=depth * 1_000_000 + func.size,
            )
            items.append(item)
            self._by_address[addr] = item

        # lowest first = leaves + small functions first
        items.sort(key=lambda w: w.priority)
        self._items = items
        self._index = 0

    def _compute_depths(
        self,
        analyzable: dict[int, FunctionInfo],
        callees_map: dict[int, list[int]],
    ) -> dict[int, int]:
        """Compute call graph depth for each function via BFS from leaves."""
        addr_set = set(analyzable.keys())

        # Build adjacency: caller -> callees (only within analyzable set)
        children: dict[int, list[int]] = defaultdict(list)
        parent_count: dict[int, int] = defaultdict(int)

        for addr in addr_set:
            for callee in callees_map.get(addr, []):
                if callee in addr_set:
                    children[addr].append(callee)
                    parent_count[callee] += 1

        leaves = deque(addr for addr in addr_set if not children[addr])

        depths: dict[int, int] = {}
        for leaf in leaves:
            depths[leaf] = 0

        # BFS upward: callers get depth = max(callee depths) + 1
        # Build reverse adjacency for upward traversal
        reverse: dict[int, list[int]] = defaultdict(list)
        for addr in addr_set:
            for callee in children[addr]:
                reverse[callee].append(addr)

        visited = set(leaves)
        queue = deque(leaves)

        while queue:
            addr = queue.popleft()
            for caller in reverse.get(addr, []):
                new_depth = depths[addr] + 1
                if caller not in depths or new_depth > depths[caller]:
                    depths[caller] = new_depth
                if caller not in visited:
                    visited.add(caller)
                    queue.append(caller)

        for addr in addr_set:
            if addr not in depths:
                depths[addr] = 0

        return depths

    def __len__(self) -> int:
        return len(self._items) - self._index

    def __bool__(self) -> bool:
        return self._index < len(self._items)

    @property
    def total(self) -> int:
        return len(self._items)

    @property
    def completed(self) -> int:
        return self._index

    @property
    def remaining(self) -> int:
        return len(self)

    def next(self) -> WorkItem | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        return item

    def peek(self) -> WorkItem | None:
        if self._index >= len(self._items):
            return None
        return self._items[self._index]

    def get_by_address(self, addr: int) -> WorkItem | None:
        return self._by_address.get(addr)

    def all_items(self) -> list[WorkItem]:
        return list(self._items)
