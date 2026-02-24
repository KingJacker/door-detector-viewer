import threading
import collections
import numpy as np
from pathlib import Path

class AsyncFrameLoader:
    def __init__(self, buffer_size_ahead=60, buffer_size_behind=20):
        self.frame_paths = []
        self.cache = {}  # index -> frame data
        self.buffer_size_ahead = buffer_size_ahead
        self.buffer_size_behind = buffer_size_behind
        self.current_index = -1
        
        self._lock = threading.RLock()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition(self._lock)
        
    def set_session(self, frames_dir):
        self.stop()
        with self._lock:
            self.frame_paths = sorted(list(Path(frames_dir).glob("frame_*.npz")))
            self.cache.clear()
            self.current_index = -1
            self._stop_event.clear()
            if self.frame_paths:
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()
        return len(self.frame_paths)

    def get_frame(self, index):
        if not (0 <= index < len(self.frame_paths)):
            return None
            
        with self._lock:
            self.current_index = index
            self._condition.notify_all()
            
            if index in self.cache:
                return self.cache[index]
        
        # Cache miss (should be rare with pre-fetching)
        try:
            frame = np.load(self.frame_paths[index])
            with self._lock:
                self.cache[index] = frame
            return frame
        except Exception:
            return None

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self._condition.notify_all()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.1)
        self._worker_thread = None

    def _worker_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                if self.current_index == -1:
                    self._condition.wait(0.1)
                    continue
                
                curr = self.current_index
                
                # Identify indices to keep
                to_keep = set(range(max(0, curr - self.buffer_size_behind), 
                                   min(len(self.frame_paths), curr + self.buffer_size_ahead + 1)))
                
                # Remove stale entries
                for idx in list(self.cache.keys()):
                    if idx not in to_keep:
                        del self.cache[idx]
                
                # Find next missing frame in priority order (ahead first)
                next_to_load = None
                for i in range(curr, min(len(self.frame_paths), curr + self.buffer_size_ahead + 1)):
                    if i not in self.cache:
                        next_to_load = i
                        break
                
                if next_to_load is None:
                    for i in range(max(0, curr - self.buffer_size_behind), curr):
                        if i not in self.cache:
                            next_to_load = i
                            break
                
                if next_to_load is None:
                    self._condition.wait(0.1)
                    continue

            # Perform load outside the lock
            try:
                frame = np.load(self.frame_paths[next_to_load])
                with self._lock:
                    # Double check we still want this frame
                    if next_to_load in to_keep:
                        self.cache[next_to_load] = frame
            except Exception:
                pass
