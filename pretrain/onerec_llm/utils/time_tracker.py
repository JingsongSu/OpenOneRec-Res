"""Time tracking utilities for performance profiling."""

import os
import time
from typing import Dict, List, Literal, Optional


class TimeTracker:
    """Track time intervals between tick calls and compute rolling averages.
    
    This class records time intervals for named events and maintains a rolling
    average of the last N intervals for each event. Supports both absolute
    wall-clock time and CPU time tracking.
    
    Args:
        n: Number of recent intervals to average (default: 1)
        time_types: List of time types to track. Options: "absolute" (wall-clock)
            or "cpu" (CPU time). Default: ["absolute"]
    
    Example:
        >>> tracker = TimeTracker(n=10)
        >>> tracker.tick("start")
        >>> # ... do some work ...
        >>> tracker.tick("end")
        >>> stats = tracker.stat()  # Returns average intervals
    """
    
    def __init__(
        self, 
        n: int = 1, 
        time_types: Optional[List[Literal["absolute", "cpu"]]] = None
    ):
        if time_types is None:
            time_types = ["absolute"]
        
        self.n = n
        self.time_types = time_types
        self.last_times: Dict[str, float] = {
            "absolute": time.perf_counter(),
            "cpu": os.times().user
        }
        self.interval_records: Dict[str, List[float]] = {}

    def tick(self, name: str) -> None:
        """Record time interval for a named event.
        
        Records the time elapsed since the last tick() call for each configured
        time type. Maintains a rolling window of the last N intervals.
        
        Args:
            name: Name of the event to track
        """
        for time_type in self.time_types:
            # Get current time based on type
            if time_type == "absolute":
                current_time = time.perf_counter()
            elif time_type == "cpu":
                current_time = os.times().user
            else:
                raise ValueError(
                    f"Invalid time_type '{time_type}'. "
                    "Allowed values are 'absolute' or 'cpu'."
                )
            
            # Calculate interval
            last_time = self.last_times[time_type]
            interval = current_time - last_time
            self.last_times[time_type] = current_time
            
            # Store interval in rolling window
            key = f"{time_type}@{name}"
            if key not in self.interval_records:
                self.interval_records[key] = []
            
            intervals = self.interval_records[key]
            intervals.append(interval)
            
            # Maintain rolling window of size n
            if len(intervals) > self.n:
                intervals.pop(0)

    def stat(self) -> Dict[str, float]:
        """Get average time intervals for all tracked events.
        
        Returns:
            Dictionary mapping event keys (format: "{time_type}@{name}") to
            average interval values. Only includes events with recorded intervals.
        """
        result: Dict[str, float] = {}
        for key, intervals in self.interval_records.items():
            if intervals:
                result[key] = sum(intervals) / len(intervals)
        return result