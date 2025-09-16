import numpy as np

try:
    from bezier import Curve
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False
    print("[MotionScale] Warning: bezier library not found. Install with: pip install bezier")
    print("[MotionScale] Falling back to basic easing functions.")

class MotionScaleNode:
    """
    Node for controlling the scale animation of layer image.
    Provides scale values for each frame based on parameters.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Scale effector for layer image animation. Supports motion curve."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "default_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "target_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "motion_curve": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
                "inverse": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("S4_SCALE_EFFECTOR",)
    RETURN_NAMES = ("Scale",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def _motion_curve_func(self, curve, t):
        # Professional easing functions for smooth animation
        # t in [0, 1]
        if curve == "linear":
            return t
        
        # Use bezier curves for smoother animation if available
        if BEZIER_AVAILABLE:
            # Define bezier control points for different easing types
            bezier_curves = {
                "ease_in_out": [(0.0, 0.0), (0.42, 0.0), (0.58, 1.0), (1.0, 1.0)],
                "ease_in": [(0.0, 0.0), (0.42, 0.0), (1.0, 1.0), (1.0, 1.0)],
                "ease_out": [(0.0, 0.0), (0.0, 0.0), (0.58, 1.0), (1.0, 1.0)]
            }
            
            if curve in bezier_curves:
                try:
                    points = bezier_curves[curve]
                    nodes = np.array(points).T
                    bezier_curve = Curve(nodes, degree=len(points)-1)
                    # Find the y value for given t (x value)
                    # Use numerical method to find parameter that gives us t as x-coordinate
                    s_vals = np.linspace(0.0, 1.0, 1000)
                    curve_points = bezier_curve.evaluate_multi(s_vals)
                    x_vals = curve_points[0]
                    y_vals = curve_points[1]
                    
                    # Interpolate to find y for given t
                    result = np.interp(t, x_vals, y_vals)
                    return float(np.clip(result, 0.0, 1.0))
                except Exception as e:
                    print(f"[MotionScale] Bezier curve error for {curve}: {e}, falling back to basic")
        
        # Fallback to enhanced mathematical easing functions
        if curve == "ease_in_out":
            return t * t * (3 - 2 * t)  # Smoothstep
        elif curve == "ease_in":
            return t * t * t  # Cubic ease-in for smoother effect
        elif curve == "ease_out":
            return 1 - (1 - t) ** 3  # Cubic ease-out for smoother effect
        else:
            return t

    def process(self, default_scale=1.0, target_scale=1.0, duration=1.0, delay=0.0, motion_curve="linear", inverse=False, loop=False, **kwargs):
        def scale_effector(total_frames, total_time):
            fps = total_frames / total_time if total_time > 0 else 1
            delay_frames = int(delay * fps)
            motion_frames = max(1, int(duration * fps))
            
            # Handle inverse motion
            if inverse:
                # First half: default to target
                scales_once = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    scale = default_scale + (target_scale - default_scale) * curved_t
                    # Ensure scale is always positive and within valid range
                    scale = max(0.01, float(scale))  # Minimum scale to prevent zero/negative values
                    scales_once.append(scale)
                
                # Second half: target to default
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    scale = target_scale + (default_scale - target_scale) * curved_t
                    # Ensure scale is always positive and within valid range
                    scale = max(0.01, float(scale))  # Minimum scale to prevent zero/negative values
                    scales_once.append(scale)
                
                motion_frames *= 2
            else:
                # Single motion: default to target
                scales_once = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    scale = default_scale + (target_scale - default_scale) * curved_t
                    # Ensure scale is always positive and within valid range
                    scale = max(0.01, float(scale))  # Minimum scale to prevent zero/negative values
                    scales_once.append(scale)
            
            # Debug: print first and last scales to verify target is reached
            if len(scales_once) > 0:
                print(f"[MotionScale] Motion curve: {motion_curve}")
                print(f"[MotionScale] Start: {scales_once[0]:.2f}")
                print(f"[MotionScale] End: {scales_once[-1]:.2f}")
                print(f"[MotionScale] Target: {target_scale}")
            
            # Handle loop and delay
            if loop:
                repeat = (total_frames - delay_frames + motion_frames - 1) // motion_frames
                motion_scales = (scales_once * repeat)[:max(0, total_frames - delay_frames)]
            else:
                if len(scales_once) < total_frames - delay_frames:
                    motion_scales = scales_once + [scales_once[-1]] * (total_frames - delay_frames - len(scales_once))
                else:
                    motion_scales = scales_once[:max(0, total_frames - delay_frames)]
            
            # Add delay and pad to total_frames
            if len(scales_once) > 0:
                start_scale = scales_once[0]
            else:
                start_scale = default_scale
            scales = [start_scale] * delay_frames + motion_scales
            if len(scales) < total_frames:
                scales += [scales[-1]] * (total_frames - len(scales))
            
            return scales
        
        scale_effector.effector_type = "scale"
        return (scale_effector,)