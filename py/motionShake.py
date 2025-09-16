import numpy as np

try:
    from bezier import Curve
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False


class MotionShakeNode:
    """
    Effector that produces per-frame shake specs for layer image.
    Modes:
    - horizontal: left-right jitter
    - vertical: up-down jitter
    - signal: 2D jitter（no colored edges）
    """

    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Shake effector for layer image animation. Supports horizontal/vertical/signal styles."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["horizontal", "vertical", "signal"], {"default": "horizontal"}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "motion_curve": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
                "inverse": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": False}),

                # Shake parameters
                "amplitude_px_default": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 512.0, "step": 0.1}),
                "amplitude_px_target": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 512.0, "step": 0.1}),
                "frequency_hz_default": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "frequency_hz_target": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "randomness": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                # No edge color settings (removed)
            },
        }

    RETURN_TYPES = ("S4_SHAKE_EFFECTOR",)
    RETURN_NAMES = ("Shake",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def _motion_curve_func(self, curve, t):
        if curve == "linear":
            return t
        if BEZIER_AVAILABLE:
            bezier_curves = {
                "ease_in_out": [(0.0, 0.0), (0.42, 0.0), (0.58, 1.0), (1.0, 1.0)],
                "ease_in": [(0.0, 0.0), (0.42, 0.0), (1.0, 1.0), (1.0, 1.0)],
                "ease_out": [(0.0, 0.0), (0.0, 0.0), (0.58, 1.0), (1.0, 1.0)],
            }
            if curve in bezier_curves:
                try:
                    points = bezier_curves[curve]
                    nodes = np.array(points).T
                    bezier_curve = Curve(nodes, degree=len(points) - 1)
                    s_vals = np.linspace(0.0, 1.0, 512)
                    curve_points = bezier_curve.evaluate_multi(s_vals)
                    x_vals = curve_points[0]
                    y_vals = curve_points[1]
                    result = np.interp(t, x_vals, y_vals)
                    return float(np.clip(result, 0.0, 1.0))
                except Exception:
                    pass
        if curve == "ease_in_out":
            return t * t * (3 - 2 * t)
        if curve == "ease_in":
            return t * t * t
        if curve == "ease_out":
            return 1 - (1 - t) ** 3
        return t

    def _lerp(self, a, b, t):
        return float(a + (b - a) * t)

    def process(
        self,
        mode="horizontal",
        duration=1.0,
        delay=0.0,
        motion_curve="linear",
        inverse=False,
        loop=False,
        amplitude_px_default=5.0,
        amplitude_px_target=5.0,
        frequency_hz_default=4.0,
        frequency_hz_target=8.0,
        randomness=0.15,
        # edge color settings removed
        **kwargs,
    ):
        def shake_effector(total_frames, total_time):
            fps = total_frames / total_time if total_time > 0 else 1
            delay_frames = int(delay * fps)
            motion_frames = max(1, int(duration * fps))

            rng = np.random.default_rng(12345)

            def one_pass():
                specs = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    tt = self._motion_curve_func(motion_curve, t)

                    amp = self._lerp(amplitude_px_default, amplitude_px_target, tt)
                    freq = self._lerp(frequency_hz_default, frequency_hz_target, tt)
                    time_s = (i / fps)
                    base = amp * np.sin(2.0 * np.pi * freq * time_s)
                    jitter = amp * randomness * (rng.random() * 2.0 - 1.0)

                    if mode == "horizontal":
                        offset_x = float(base + jitter)
                        offset_y = 0.0
                    elif mode == "vertical":
                        offset_x = 0.0
                        offset_y = float(base + jitter)
                    else:
                        # signal mode: 2D jitter without colored edges
                        offset_x = float(base + jitter * 0.5)
                        offset_y = float(-base + jitter * 0.5)

                    spec = {
                        "mode": mode,
                        "params": {
                            "offset_x": offset_x,
                            "offset_y": offset_y,
                        },
                    }
                    specs.append(spec)
                return specs

            if inverse:
                forward = one_pass()
                backward = list(reversed(forward[:-1])) if len(forward) > 1 else []
                motion_specs_once = forward + backward
                motion_frames_effective = len(motion_specs_once)
            else:
                motion_specs_once = one_pass()
                motion_frames_effective = len(motion_specs_once)

            if loop:
                repeat = (total_frames - delay_frames + motion_frames_effective - 1) // motion_frames_effective
                motion_specs = (motion_specs_once * repeat)[: max(0, total_frames - delay_frames)]
            else:
                if len(motion_specs_once) < total_frames - delay_frames:
                    motion_specs = motion_specs_once + [motion_specs_once[-1]] * (total_frames - delay_frames - len(motion_specs_once))
                else:
                    motion_specs = motion_specs_once[: max(0, total_frames - delay_frames)]

            start_spec = motion_specs_once[0] if len(motion_specs_once) > 0 else {"mode": mode, "params": {"offset_x": 0, "offset_y": 0}}
            specs = [start_spec] * delay_frames + motion_specs
            if len(specs) < total_frames:
                specs += [specs[-1]] * (total_frames - len(specs))

            return specs

        shake_effector.effector_type = "shake"
        return (shake_effector,)


