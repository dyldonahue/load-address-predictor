# ================================================
# Author: Dylan Donahue
# 11.15.2025
# Basic implementation inspired by Berti: https://ieeexplore.ieee.org/document/9923806, https://github.com/agusnt/Berti-Artifact
# ================================================

# List of simplifications made from orginal:
# no timestamps, use instruction counter instead
# no cache simulator, just a predict history that expires after some time
# (since no cache sim) No L1 vs L2 distinction - either prefetch or don't

import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple


@dataclass(frozen=True)
class HistoryEntry:
    """Single access in history table"""

    tag: int
    address: int
    counter: int


@dataclass
class DeltaEntry:
    """Single delta pattern"""

    delta: int
    count: int = 0
    total: int = 0

    @property
    def coverage(self) -> float:
        if self.total == 0:
            return 0.0
        return self.count / self.total

    def should_prefetch(self, thresh: float = 0.35) -> bool:
        return self.coverage >= thresh


class BaseAddressPredictor:
    def __init__(
        self,
        num_sets=8,
        num_ways=16,
        delta_size=16,
        timely_length=4,
        max_deltas_per_pc=16,
        prediction_window=100,
        cache_line_size=64,
    ):
        """
        Args:
            num_sets: Number of sets in history table
            num_ways: Ways per set (associativity) in history table
            max_pcs: Maximum PCs tracked in delta table
            max_deltas_per_pc: Maximum deltas per PC
            timely_length: Minimum instruction gap for "timely" delta
            prediction_window: Instructions to wait before marking prediction useless
        """
        self.num_sets = num_sets
        self.num_ways = num_ways
        self.timely_length = timely_length
        self.max_deltas_per_pc = max_deltas_per_pc
        self.prediction_window = prediction_window
        self.cache_line_size = cache_line_size

        self.learning_events_per_pc = {}

        # creates num_set indices with num_ways assoiciativity
        # Berti uses FIFO, and I dont expect any variiation to change this, so using deque as this will implicitly handle FIFO eviction
        self.history: List[deque] = [deque(maxlen=num_ways) for _ in range(num_sets)]

        # Delta table is fully associative and FIFO, need to store PC with the entry so each item will be a tuple of pc,  DeltaEntry
        self.deltas = deque(maxlen=delta_size)

        self.instr_count = 0
        self.total_accesses = 0
        self.total_predictions = 0
        self.pending_predictions: Dict[int, int] = {}  # address -> when_predicted
        self.useful_predictions = 0
        self.useless_predictions = 0

    def get_hash(self, pc):
        """Hash PC to history set index"""
        return pc % self.num_sets

    def is_timely(self, history_entry, current_counter):
        """Check if history entry is timely enough for learning"""
        gap = current_counter - history_entry.counter
        return gap >= self.timely_length

    def find_delta_entry(self, pc: int) -> Optional[Tuple[int, List[DeltaEntry]]]:
        """Find delta list for this PC"""
        # Berti only tags this by lower 10 bits
        tag = pc & 1023
        for entry in self.deltas:
            if entry[0] == tag:
                return entry
        return None

    def _get_or_create_delta_list(self, pc: int) -> List[DeltaEntry]:
        """Get or create delta list for PC"""
        entry = self.find_delta_entry(pc)

        if entry is not None:
            return entry[1]

        # Create new entry (FIFO eviction automatic with maxlen)
        new_list = []
        tag = pc & 1023
        self.deltas.append((tag, new_list))
        return new_list

    def _add_delta(self, delta_list: List[DeltaEntry], delta: int) -> DeltaEntry:
        """Add or get delta in list"""
        # Check if exists
        for entry in delta_list:
            if entry.delta == delta:
                return entry

        # Add new
        if len(delta_list) >= self.max_deltas_per_pc:
            self._evict_worst_delta(delta_list)
        
        new_delta = DeltaEntry(delta=delta)
        delta_list.append(new_delta)
        return new_delta

    def _evict_worst_delta(self, delta_list: List[DeltaEntry]):
        """Evict delta with worst coverage"""
        if not delta_list:
            return
        worst = min(delta_list, key=lambda d: d.coverage)
        delta_list.remove(worst)

    def _find_timely_deltas(self, pc: int, cache_line: int) -> List[int]:
        """Search history for timely deltas"""
        set_idx = self.get_hash(pc)
        timely_deltas = []

        for entry in self.history[set_idx]:
            if entry.counter == self.instr_count:
                continue
            # Berti history records 7 LSB after those used for hashing, so bits 3:9
            tag = pc & 1016

            if entry.tag == tag and self.is_timely(entry, self.instr_count):
                delta = cache_line - entry.address
                timely_deltas.append(delta)

        return timely_deltas

    def _to_cache_line(self, byte_address: int) -> int:
        """Convert byte address to cache line number"""
        return byte_address // self.cache_line_size

    # expects input trace line of pc,load_address
    def predict(self, pc: int, address: int) -> List[int]:
        """Generate prefetch predictions"""
        delta_entry = self.find_delta_entry(pc)

        if delta_entry is None:
            return []

        predictions = []
        delta_list = delta_entry[1]

        for delta in delta_list:
            if delta.should_prefetch():
                pred_line = address + delta.delta
                predictions.append(pred_line)
                self.pending_predictions[pred_line] = self.instr_count
                self.total_predictions += 1

        return predictions

    def on_fill(self, pc: int, address: int):
        """Learn deltas when data fills cache"""
        # Find timely deltas
        timely_deltas = self._find_timely_deltas(pc, address)
        self.learning_events_per_pc[pc] = self.learning_events_per_pc.get(pc, 0) + 1

        if not timely_deltas:
            return

        # Update delta table
        delta_list = self._get_or_create_delta_list(pc)
        unique_timely_deltas = set(timely_deltas)

        # Update counts for timely deltas
        for delta in unique_timely_deltas:
            delta_entry = self._add_delta(delta_list, delta)
            delta_entry.count += 1

        # Increment totals for all deltas
        for delta_entry in delta_list:
            delta_entry.total += 1


    def _check_prediction_accuracy(self, address: int):
        """Check if this access was predicted"""
        if address in self.pending_predictions:
            pred_time = self.pending_predictions[address]
            window = self.instr_count - pred_time

            if window <= self.prediction_window:
                # Prediction was useful!
                self.useful_predictions += 1
            else:
                # Predicted but too late
                self.useless_predictions += 1

            del self.pending_predictions[address]

    def _expire_old_predictions(self):
        """Remove predictions that are too old (never accessed)"""
        expired = []
        for addr, pred_time in self.pending_predictions.items():
            if self.instr_count - pred_time > self.prediction_window:
                expired.append(addr)

        for addr in expired:
            self.useless_predictions += 1
            del self.pending_predictions[addr]

    def finalize(self):
        """Call at end of trace to clean up pending predictions"""
        self._expire_old_predictions()

    def access(self, pc: int, address: int) -> List[int]:
        """
        Process a memory access.
        Returns list of addresses to prefetch.
        """
        self.instr_count += 1
        self.total_accesses += 1

        cache_line_full = self._to_cache_line(address)

        # truncate to 24 bit as done in BERTI
        cache_line = cache_line_full & ((1 << 24) - 1)

        # Do some accuracy work
        was_prefetch_hit = cache_line in self.pending_predictions
        self._check_prediction_accuracy(cache_line)

        # Clean up old predictions periodically
        if self.instr_count % 100 == 0:
            self._expire_old_predictions()

        

        # Generate predictions
        predictions = self.predict(pc, cache_line)

        # Update history periodically (simulating demand misses) or on prefetch hit
        if was_prefetch_hit or self.instr_count % 10 == 0:

            # Record in history
            tag = pc & 1016
            entry = HistoryEntry(tag=tag, address=cache_line, counter=self.instr_count)
            set_idx = self.get_hash(pc)
            self.history[set_idx].append(entry)

            # learn
            self.on_fill(pc, cache_line)

        return predictions

    def process_trace_line(self, line: str) -> List[int]:
        """
        Process a trace line: "pc address"
        Returns predictions.
        """
        parts = line.strip().split()
        if len(parts) != 2:
            return []

        try:
            pc = int(parts[0], 16) if parts[0].startswith("0x") else int(parts[0])
            address = int(parts[1], 16) if parts[1].startswith("0x") else int(parts[1])
        except ValueError:
            return []

        return self.access(pc, address)

    def get_stats(self) -> Dict:
        """Get statistics with accuracy"""
        accuracy = 0.0
        if self.total_predictions > 0:
            accuracy = self.useful_predictions / self.total_predictions

        return {
            "total_accesses": self.total_accesses,
            "total_predictions": self.total_predictions,
            "useful_predictions": self.useful_predictions,
            "useless_predictions": self.useless_predictions,
            "accuracy": accuracy,
            "tracked_pcs": len(self.deltas),
            "pending_predictions": len(self.pending_predictions),
        }

    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()
        print("\n=== Predictor Statistics ===")
        print(f"Total Accesses: {stats['total_accesses']}")
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"  Useful:   {stats['useful_predictions']}")
        print(f"  Useless:  {stats['useless_predictions']}")
        print(f"  Pending:  {stats['pending_predictions']}")
        print(f"Accuracy: {stats['accuracy']:.1%}")
        print(f"Tracked PCs: {stats['tracked_pcs']}")
        if stats["total_accesses"] > 0:
            print(
                f"Predictions/Access: {stats['total_predictions']/stats['total_accesses']:.2f}"
            )


