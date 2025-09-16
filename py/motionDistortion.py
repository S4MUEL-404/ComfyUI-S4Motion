import numpy as np

try:
    from bezier import Curve
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False
    print("[MotionDistortion] Warning: bezier library not found. Install with: pip install bezier")
    print("[MotionDistortion] Falling back to basic easing functions.")


class MotionDistortionNode:
    """
    Node for controlling distortion effect of the layer image.
    Produces per-frame distortion specs consumed by Motion Config.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Distortion effector for layer image animation. Supports wave, vortex and radial with motion curve."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Core timeline controls
                "mode": (["wave", "vortex", "radial"], {"default": "wave"}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "motion_curve": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
                "inverse": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": False}),

                # Wave parameters
                "wave_amplitude_default": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "wave_amplitude_target": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "wave_wavelength_default": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 4096.0, "step": 0.1}),
                "wave_wavelength_target": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 4096.0, "step": 0.1}),
                "wave_phase_default": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "wave_phase_target": ("FLOAT", {"default": 360.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "wave_direction": (["x", "y", "xy"], {"default": "x"}),

                # Vortex parameters (center-relative swirl with seamless phase)
                "vortex_strength_deg_default": ("FLOAT", {"default": 20.0, "min": -720.0, "max": 720.0, "step": 1.0}),
                "vortex_strength_deg_target": ("FLOAT", {"default": 20.0, "min": -720.0, "max": 720.0, "step": 1.0}),
                "vortex_radius_px_default": ("FLOAT", {"default": 256.0, "min": 1.0, "max": 8192.0, "step": 1.0}),
                "vortex_radius_px_target": ("FLOAT", {"default": 256.0, "min": 1.0, "max": 8192.0, "step": 1.0}),
                "vortex_phase_deg_default": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 1.0}),
                "vortex_phase_deg_target": ("FLOAT", {"default": 360.0, "min": -3600.0, "max": 3600.0, "step": 1.0}),

                # Radial (barrel/pincushion) parameters
                "radial_k_default": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001}),
                "radial_k_target": ("FLOAT", {"default": 0.2, "min": -2.0, "max": 2.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("S4_DISTORTION_EFFECTOR",)
    RETURN_NAMES = ("Distortion",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def _motion_curve_func(self, curve, t):
        # Easing function with optional Bezier
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
                    s_vals = np.linspace(0.0, 1.0, 1000)
                    curve_points = bezier_curve.evaluate_multi(s_vals)
                    x_vals = curve_points[0]
                    y_vals = curve_points[1]
                    result = np.interp(t, x_vals, y_vals)
                    return float(np.clip(result, 0.0, 1.0))
                except Exception as e:
                    print(f"[MotionDistortion] Bezier curve error for {curve}: {e}, falling back to basic")
        if curve == "ease_in_out":
            return t * t * (3 - 2 * t)
        elif curve == "ease_in":
            return t * t * t
        elif curve == "ease_out":
            return 1 - (1 - t) ** 3
        else:
            return t

    def _lerp(self, a, b, t):
        return float(a + (b - a) * t)

    def process(
        self,
        mode="wave",
        duration=1.0,
        delay=0.0,
        motion_curve="linear",
        inverse=False,
        loop=False,
        # wave
        wave_amplitude_default=0.0,
        wave_amplitude_target=10.0,
        wave_wavelength_default=100.0,
        wave_wavelength_target=100.0,
        wave_phase_default=0.0,
        wave_phase_target=360.0,
        wave_direction="x",
        # vortex
        vortex_strength_deg_default=20.0,
        vortex_strength_deg_target=20.0,
        vortex_radius_px_default=256.0,
        vortex_radius_px_target=256.0,
        vortex_phase_deg_default=0.0,
        vortex_phase_deg_target=360.0,
        # radial
        radial_k_default=0.0,
        radial_k_target=0.2,
        **kwargs,
    ):
        def distortion_effector(total_frames, total_time):
            fps = total_frames / total_time if total_time > 0 else 1
            delay_frames = int(delay * fps)
            motion_frames = max(1, int(duration * fps))

            def one_pass_specs():
                specs = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)

                    if mode == "wave":
                        spec = {
                            "mode": "wave",
                            "params": {
                                "amplitude_px": self._lerp(wave_amplitude_default, wave_amplitude_target, curved_t),
                                "wavelength_px": self._lerp(wave_wavelength_default, wave_wavelength_target, curved_t),
                                "phase_deg": self._lerp(wave_phase_default, wave_phase_target, curved_t),
                                "direction": wave_direction,
                            },
                        }
                    elif mode == "vortex":
                        spec = {
                            "mode": "vortex",
                            "params": {
                                "strength_deg": self._lerp(vortex_strength_deg_default, vortex_strength_deg_target, curved_t),
                                "radius_px": self._lerp(vortex_radius_px_default, vortex_radius_px_target, curved_t),
                                "center_x": 0.5,
                                "center_y": 0.5,
                                "phase_deg": self._lerp(vortex_phase_deg_default, vortex_phase_deg_target, curved_t),
                            },
                        }
                    elif mode == "radial":
                        spec = {
                            "mode": "radial",
                            "params": {
                                "k": self._lerp(radial_k_default, radial_k_target, curved_t),
                                "center_x": 0.5,
                                "center_y": 0.5,
                            },
                        }
                    else:
                        spec = {"mode": "none", "params": {}}

                    specs.append(spec)
                return specs

            if inverse:
                forward = one_pass_specs()
                backward = list(reversed(forward[:-1])) if len(forward) > 1 else []
                motion_specs_once = forward + backward
                motion_frames_effective = len(motion_specs_once)
            else:
                motion_specs_once = one_pass_specs()
                motion_frames_effective = len(motion_specs_once)

            # Loop and delay handling
            if loop:
                repeat = (total_frames - delay_frames + motion_frames_effective - 1) // motion_frames_effective
                motion_specs = (motion_specs_once * repeat)[: max(0, total_frames - delay_frames)]
            else:
                if len(motion_specs_once) < total_frames - delay_frames:
                    motion_specs = motion_specs_once + [motion_specs_once[-1]] * (total_frames - delay_frames - len(motion_specs_once))
                else:
                    motion_specs = motion_specs_once[: max(0, total_frames - delay_frames)]

            # Prepend delay frames with first spec (no distortion during delay interpreted as first state)
            start_spec = motion_specs_once[0] if len(motion_specs_once) > 0 else {"mode": mode, "params": {}}
            specs = [start_spec] * delay_frames + motion_specs
            if len(specs) < total_frames:
                specs += [specs[-1]] * (total_frames - len(specs))

            # Debug
            if len(specs) > 0:
                print(f"[MotionDistortion] Mode: {mode}")
                print(f"[MotionDistortion] Frames: {len(specs)} (delay {delay_frames}, motion {motion_frames_effective})")

            return specs

        distortion_effector.effector_type = "distortion"
        return (distortion_effector,)

