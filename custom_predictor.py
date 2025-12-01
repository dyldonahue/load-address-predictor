# ================================================
# Author: Dylan Donahue
# 11.24.2025
# ================================================

# List of changes made from orginal:
# - Replaced FIFO delta eviction with LFU delta eviction
# - Implemented Cost Benefit Analysis, with switching mechanism between Berti-style and simple Stride prediction


import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple, OrderedDict
from base_predictor import BaseAddressPredictor, DeltaEntry

class SimpleStridePredictor:
    """Simple per-PC stride predictor"""
    
    def __init__(self, max_pcs: int = 16):
        self.strides = {}  # pc -> (last_addr, stride, confidence)
        self.MAX_PCS = max_pcs
        self.total_predictions = 0
        self.lfu_freq = {}
    
    def predict(self, pc: int, cache_line: int) -> List[int]:
        """Predict using stride"""

        self.lfu_freq[pc] = self.lfu_freq.get(pc, 0) + 1
        
        if pc not in self.strides:
            # Evict if needed
            if len(self.strides) >= self.MAX_PCS:
                # LFU: Evict least frequent
                lfu_pc = min(self.lfu_freq.keys(), key=lambda p: self.lfu_freq[p])
                del self.strides[lfu_pc]
                del self.lfu_freq[lfu_pc]

            # First access - no prediction
            self.strides[pc] = (cache_line, 0, 0)
            
            return []
        
        # Get stride info
        last_addr, stride, confidence = self.strides[pc]
        current_stride = cache_line - last_addr
        
        # Update stride
        if current_stride == stride:
            # Consistent stride, increase confidence
            confidence = min(confidence + 1, 7)
        else:
            # Stride changed - reset
            stride = current_stride
            confidence = 1
        
        self.strides[pc] = (cache_line, stride, confidence)
        
        # Predict if confident
        if confidence >= 3:
            prediction = cache_line + stride
            self.total_predictions += 1
            return [prediction]
        
        return []
    
    def get_prediction_rate(self) -> float:
        """For switching logic"""
        return self.total_predictions / 100000 if self.total_predictions > 0 else 0.0

class CustomAddressPredictor(BaseAddressPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Replace deque with OrderedDict for LFU
        self.deltas = {} # tag -> List[DeltaEntry]
        self.delta_access_freq = {}  # tag -> access count
        self.MAX_PCS = kwargs.get('delta_size', 16)
        
        # Switching mechanism state
        self.using_stride = False
        self.stride_predictor = SimpleStridePredictor()
        self.check_interval = 100000
        self.last_useful = 0
        self.last_total = 0

    def find_delta_entry(self, pc: int):
        """Find and update LFU"""
        tag = pc & 1023
        
        if tag in self.deltas:
            # LFU: Increment access frequency
            self.delta_access_freq[tag] += 1
            return (tag, self.deltas[tag])
        
        return None
    
    def _get_or_create_delta_list(self, pc: int) -> List[DeltaEntry]:
        """Get or create with LFU eviction"""
        tag = pc & 1023
        
        if tag in self.deltas:
            self.delta_access_freq[tag] += 1
            return self.deltas[tag]
        
        # Need to add new entry
        if len(self.deltas) >= self.MAX_PCS:
            # LFU: Evict least frequently used
            lfu_tag = min(self.delta_access_freq.keys(), 
                         key=lambda t: self.delta_access_freq[t])
            
            del self.deltas[lfu_tag]
            del self.delta_access_freq[lfu_tag]
        
        self.deltas[tag] = []
        self.delta_access_freq[tag] = 1
        return self.deltas[tag]
    
    def access(self, pc: int, address: int) -> List[int]:
        """Access with periodic switching check"""
        
        # Check if should switch predictor
        if self.instr_count % self.check_interval == 0 and self.instr_count > 0:
            self._check_switch()
        
        # Route to active predictor
        if self.using_stride:
            return self._access_stride(pc, address)
        else:
            return super().access(pc, address)
        
    def _check_switch(self):
        """Cost-benefit switching logic"""
        
        recent_useful = self.useful_predictions - self.last_useful
        recent_total = self.total_predictions - self.last_total
        
        if recent_total < 100:
            return  # Not enough data
        
        recent_accuracy = recent_useful / recent_total
        recent_prediction_rate = recent_total / self.check_interval
        
        # Switch to stride if efficiency is poor
        if not self.using_stride:
            # Cost-benefit: low accuracy and high prediction rate = wasteful
            if recent_accuracy < 0.40:
                print(f"\n[{self.instr_count:,}] SWITCH → STRIDE (accuracy={recent_accuracy:.1%})")
                self.using_stride = True
            elif recent_accuracy < 0.55 and recent_prediction_rate > 0.20:
                print(f"\n[{self.instr_count:,}] SWITCH → STRIDE (efficiency low)")
                self.using_stride = True
        
        # Switch back to Berti if stride isn't helping
        else:
            # If we're not predicting much with stride either, go back
            stride_rate = self.stride_predictor.get_prediction_rate()
            if stride_rate < 0.05:  # Stride also struggling
                print(f"\n[{self.instr_count:,}] SWITCH → BERTI (stride not helping)")
                self.using_stride = False
        
        self.last_useful = self.useful_predictions
        self.last_total = self.total_predictions

    def _access_stride(self, pc: int, address: int) -> List[int]:
        """Access using stride predictor"""
        self.instr_count += 1
        self.total_accesses += 1
        
        cache_line = self._to_cache_line(address)
        cache_line = cache_line & ((1 << 24) - 1)
        
        self._check_prediction_accuracy(cache_line)

        if self.instr_count % 100 == 0:
            self._expire_old_predictions()
        
        # Use stride predictor
        predictions = self.stride_predictor.predict(pc, cache_line)
        
        # Track predictions
        for pred in predictions:
            self.pending_predictions[pred] = self.instr_count
            self.total_predictions += 1
        
        return predictions
    
    def get_stats(self):
        """Enhanced stats with switching info"""
        stats = super().get_stats()
        stats['currently_using'] = 'stride' if self.using_stride else 'berti'
        return stats

        

    
