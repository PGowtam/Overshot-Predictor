import numpy as np
from collections import deque

class WelfordZScore:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, val):
        if len(self.values) == self.window_size:
            old_val = self.values[0]
            self.values.append(val)
            old_mean = self.mean
            self.mean += (val - old_val) / self.window_size
            self.m2 += (val - old_val) * (val - self.mean + old_val - old_mean)
        else:
            self.values.append(val)
            n = len(self.values)
            delta = val - self.mean
            self.mean += delta / n
            self.m2 += delta * (val - self.mean)

    def get_z_score(self, val):
        if len(self.values) < 30:
            return 0.0
        var = self.m2 / (len(self.values) - 1)
        if var < 1e-12:
            return 0.0
        return (val - self.mean) / np.sqrt(var)

class LiveFeatureEngine:
    def __init__(self):
        self.z_ofi = WelfordZScore(1000)
        self.z_depth = WelfordZScore(1000)
        self.z_susc = WelfordZScore(1000)
        self.z_vel = WelfordZScore(1000)
        self.z_spread = WelfordZScore(1000)

        self.prev_bid = None
        self.prev_ask = None
        self.prev_time = None

        self.brick_open = None
        self.brick_size = 1.0
        self.prev_brick_open = None
        self.prev_brick_size = 1.0

    def on_new_brick(self, brick):
        self.prev_brick_open = self.brick_open
        self.prev_brick_size = self.brick_size

        self.brick_open = brick["close"] - (brick["dir"] * self.brick_size)

    def compute_vector(self, tick):
        bid = tick["bid"]
        ask = tick["ask"]
        bid_vol = tick["bid_vol"]
        ask_vol = tick["ask_vol"]
        time_msc = tick["time_msc"]

        mid = (bid + ask) / 2
        spread = ask - bid
        depth = bid_vol + ask_vol

        # Velocity
        if self.prev_time is not None:
            dt = (time_msc - self.prev_time) / 1000.0
        else:
            dt = 0.250
        self.prev_time = time_msc
        vel = 1.0 / (dt + 1e-3)

        # OFI logic
        ofi = 0.0
        if self.prev_bid is not None and self.prev_ask is not None:
            if bid > self.prev_bid:
                ofi += bid_vol
            elif bid == self.prev_bid:
                ofi += (bid_vol - 0) # Simplification for pure synthetic
            else:
                ofi -= bid_vol

            if ask < self.prev_ask:
                ofi -= ask_vol
            elif ask == self.prev_ask:
                ofi -= (ask_vol - 0)
            else:
                ofi += ask_vol

        self.prev_bid = bid
        self.prev_ask = ask

        # Susceptibility
        susc = ofi / (depth + 1e-8)

        # Z-scores
        self.z_ofi.update(ofi)
        self.z_depth.update(depth)
        self.z_susc.update(susc)
        self.z_vel.update(vel)
        self.z_spread.update(spread)

        v_ofi = self.z_ofi.get_z_score(ofi)
        v_depth = self.z_depth.get_z_score(depth)
        v_susc = self.z_susc.get_z_score(susc)
        v_vel = self.z_vel.get_z_score(vel)
        v_spread = self.z_spread.get_z_score(spread)

        # Progress & Zones
        if self.brick_open is None:
            self.brick_open = mid

        progress = (mid - self.brick_open) / self.brick_size
        flag_curr = 1.0

        flag_zone = 0.0
        if self.prev_brick_open is not None:
            if abs(mid - self.prev_brick_open) >= self.prev_brick_size:
                flag_zone = 1.0

        decay = 0.0

        return [v_ofi, v_depth, v_susc, v_vel, v_spread, progress, flag_curr, flag_zone, decay]

class InferenceBuffer:
    def __init__(self):
        self.micro = deque(maxlen=100)
        self.snapshots = deque(maxlen=10)
        self.macro = deque(maxlen=10)
        self.current_brick_id = 0
        self.brick_sizes = deque(maxlen=50)
        self.last_brick_ts = None

    def push_tick(self, vec):
        self.micro.append((vec, self.current_brick_id))

    def on_brick_close(self, brick):
        if len(self.micro) == 0:
            return None

        # Snapshot
        snapshot = np.zeros((100, 9))
        for i, (v, b_id) in enumerate(self.micro):
            vec = list(v)
            vec[6] = 1.0 if b_id == self.current_brick_id else 0.0
            vec[8] = (self.current_brick_id - b_id) / 100.0
            snapshot[100 - len(self.micro) + i] = vec

        self.snapshots.append(snapshot)

        # Macro
        dur = 0.0
        if self.last_brick_ts is not None:
            dur = (brick["ts"] - self.last_brick_ts) / 1000.0
        self.last_brick_ts = brick["ts"]

        self.brick_sizes.append(1.0)
        z_size = 0.0
        if len(self.brick_sizes) > 1:
            z_size = (1.0 - np.mean(self.brick_sizes)) / (np.std(self.brick_sizes) + 1e-8)

        self.macro.append([np.log(dur + 1.0), float(brick["dir"]), z_size])
        self.current_brick_id += 1

        if len(self.snapshots) == 10:
            micro_tensor = np.array(self.snapshots)[np.newaxis, ...]
            macro_tensor = np.array(self.macro)[np.newaxis, ...]
            return micro_tensor, macro_tensor
        return None

if __name__ == '__main__':
    print("Feature Engine defined.")
