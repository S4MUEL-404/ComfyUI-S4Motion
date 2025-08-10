import numpy as np
from PIL import Image

try:
    from bezier import Curve
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False
    print("[MotionMask] Warning: bezier library not found. Install with: pip install bezier")


class MotionMaskNode:
    """
    Node for providing animated mask specs and a base mask image for Motion Config.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Mask effector providing Photoshop-like clipping mask with transform over time."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"default": None}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "motion_curve": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
                "inverse": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": False}),

                # Transform defaults/targets
                "x_default": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "x_target": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "y_default": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "y_target": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "rotation_default": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "rotation_target": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "scale_default": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "scale_target": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "opacity_default": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "opacity_target": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("S4_MASK_EFFECTOR",)
    RETURN_NAMES = ("Mask",)

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
                    s_vals = np.linspace(0.0, 1.0, 1000)
                    curve_points = bezier_curve.evaluate_multi(s_vals)
                    x_vals = curve_points[0]
                    y_vals = curve_points[1]
                    result = np.interp(t, x_vals, y_vals)
                    return float(np.clip(result, 0.0, 1.0))
                except Exception as e:
                    print(f"[MotionMask] Bezier curve error for {curve}: {e}, falling back to basic")
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

    def _to_pil_mask(self, mask_input):
        """Convert ComfyUI MASK to PIL L image.

        Accepts torch/numpy/PIL; normalizes to 0..255 uint8 single channel.
        Handles shapes like (H,W), (1,H,W), (1,1,H,W).
        """
        if mask_input is None:
            return None
        if isinstance(mask_input, Image.Image):
            return mask_input.convert('L')
        try:
            import torch
            is_torch = isinstance(mask_input, torch.Tensor)
        except Exception:
            torch = None
            is_torch = False

        if is_torch:
            arr = mask_input.detach().cpu().numpy()
        elif isinstance(mask_input, np.ndarray):
            arr = mask_input
        else:
            # Fallback: try array conversion
            try:
                arr = np.array(mask_input)
            except Exception:
                raise Exception(f"Unsupported mask input type: {type(mask_input)}")

        # Squeeze leading singleton dims
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        while arr.ndim > 2 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        if arr.ndim != 2:
            # If still has channels, take first channel
            arr = arr[..., 0]

        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = arr.round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr, mode='L')

    def process(
        self,
        mask=None,
        duration=1.0,
        delay=0.0,
        motion_curve="linear",
        inverse=False,
        loop=False,
        x_default=0.0,
        x_target=0.0,
        y_default=0.0,
        y_target=0.0,
        rotation_default=0.0,
        rotation_target=0.0,
        scale_default=1.0,
        scale_target=1.0,
        opacity_default=1.0,
        opacity_target=1.0,
        **kwargs,
    ):
        base_mask = self._to_pil_mask(mask) if mask is not None else None

        def mask_effector(total_frames, total_time):
            fps = total_frames / total_time if total_time > 0 else 1
            delay_frames = int(delay * fps)
            motion_frames = max(1, int(duration * fps))

            def one_pass():
                specs = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    tt = self._motion_curve_func(motion_curve, t)
                    spec = {
                        "x": self._lerp(x_default, x_target, tt),
                        "y": self._lerp(y_default, y_target, tt),
                        "rotation": self._lerp(rotation_default, rotation_target, tt),
                        "scale": self._lerp(scale_default, scale_target, tt),
                        "opacity": self._lerp(opacity_default, opacity_target, tt),
                    }
                    specs.append(spec)
                return specs

            if inverse:
                forward = one_pass()
                backward = list(reversed(forward[:-1])) if len(forward) > 1 else []
                motion_specs_once = forward + backward
            else:
                motion_specs_once = one_pass()

            if loop:
                repeat = (total_frames + len(motion_specs_once) - 1) // len(motion_specs_once)
                motion_specs = (motion_specs_once * repeat)[: total_frames]
            else:
                if len(motion_specs_once) < total_frames:
                    motion_specs = motion_specs_once + [motion_specs_once[-1]] * (total_frames - len(motion_specs_once))
                else:
                    motion_specs = motion_specs_once[: total_frames]

            # apply delay
            start_spec = motion_specs_once[0] if len(motion_specs_once) > 0 else {"x": 0, "y": 0, "rotation": 0, "scale": 1, "opacity": 1}
            specs = [start_spec] * min(delay_frames, total_frames)
            remaining = total_frames - len(specs)
            specs.extend(motion_specs[: max(0, remaining)])

            return specs

        # tag effector type and carry the base mask image for use in Motion Config
        mask_effector.effector_type = "mask"
        mask_effector.mask_image = base_mask
        return (mask_effector,)

