import math


class LowPassFilter:
    def __init__(self):
        self.initialized = False
        self.last_value = None

    def filter(self, value, alpha):
        if not self.initialized:
            self.last_value = value
            self.initialized = True
            return value

        filtered = alpha * value + (1 - alpha) * self.last_value
        self.last_value = filtered

        return filtered


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff=1.0,
        beta=0.007,
        d_cutoff=1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

        self.last_time = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e

        return r / (r + 1)

    def filter(self, value, timestamp):

        # First frame
        if self.last_time is None:
            self.last_time = timestamp

            self.x_filter.filter(value, 1.0)
            self.dx_filter.filter(0.0, 1.0)

            return value

        # Time difference
        dt = timestamp - self.last_time

        if dt <= 0:
            dt = 1e-6

        self.last_time = timestamp

        # Estimate derivative
        previous_value = self.x_filter.last_value

        dx = (value - previous_value) / dt

        # Smooth derivative
        alpha_d = self.smoothing_factor(
            dt,
            self.d_cutoff
        )

        filtered_dx = self.dx_filter.filter(
            dx,
            alpha_d
        )

        # Adaptive cutoff
        cutoff = (
            self.min_cutoff
            + self.beta * abs(filtered_dx)
        )

        # Calculate smoothing factor
        alpha = self.smoothing_factor(
            dt,
            cutoff
        )

        # Smooth signal
        filtered_value = self.x_filter.filter(
            value,
            alpha
        )

        return filtered_value