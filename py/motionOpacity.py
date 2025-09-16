import numpy as np

try:
    from bezier import Curve
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False
    print("[MotionOpacity] Warning: bezier library not found. Install with: pip install bezier")
    print("[MotionOpacity] Falling back to basic easing functions.")

class MotionOpacityNode:
    """
    Node for controlling the opacity animation of layer image.
    Provides opacity values for each frame based on parameters.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Opacity effector for layer image animation. Supports motion curve."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "default_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "motion_curve": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
                "inverse": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("S4_OPACITY_EFFECTOR",)
    RETURN_NAMES = ("Opacity",)

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
                    print(f"[MotionOpacity] Bezier curve error for {curve}: {e}, falling back to basic")
        
        # Fallback to enhanced mathematical easing functions
        if curve == "ease_in_out":
            return t * t * (3 - 2 * t)  # Smoothstep
        elif curve == "ease_in":
            return t * t * t  # Cubic ease-in for smoother effect
        elif curve == "ease_out":
            return 1 - (1 - t) ** 3  # Cubic ease-out for smoother effect
        else:
            return t

    def process(self, default_opacity=1.0, target_opacity=1.0, duration=1.0, delay=0.0, motion_curve="ease_in_out", inverse=False, loop=False, **kwargs):
        def opacity_effector(total_frames, total_time):
            fps = total_frames / total_time if total_time > 0 else 1
            delay_frames = int(delay * fps)
            motion_frames = max(1, int(duration * fps))
            
            # Handle inverse motion
            if inverse:
                # First half: default to target
                opacities_once = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    opacity = default_opacity + (target_opacity - default_opacity) * curved_t
                    # Ensure opacity is always within valid range
                    opacity = max(0.0, min(1.0, float(opacity)))
                    opacities_once.append(opacity)
                
                # Second half: target to default
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    opacity = target_opacity + (default_opacity - target_opacity) * curved_t
                    # Ensure opacity is always within valid range
                    opacity = max(0.0, min(1.0, float(opacity)))
                    opacities_once.append(opacity)
                
                motion_frames *= 2
            else:
                # Single motion: default to target
                opacities_once = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    opacity = default_opacity + (target_opacity - default_opacity) * curved_t
                    # Ensure opacity is always within valid range
                    opacity = max(0.0, min(1.0, float(opacity)))
                    opacities_once.append(opacity)
            
            # Debug: print first and last opacities to verify target is reached
            if len(opacities_once) > 0:
                print(f"[MotionOpacity] Motion curve: {motion_curve}")
                print(f"[MotionOpacity] Start: {opacities_once[0]:.2f}")
                print(f"[MotionOpacity] End: {opacities_once[-1]:.2f}")
                print(f"[MotionOpacity] Target: {target_opacity}")
            
            # Handle loop and delay
            if loop:
                repeat = (total_frames - delay_frames + motion_frames - 1) // motion_frames
                motion_opacities = (opacities_once * repeat)[:max(0, total_frames - delay_frames)]
            else:
                if len(opacities_once) < total_frames - delay_frames:
                    motion_opacities = opacities_once + [opacities_once[-1]] * (total_frames - delay_frames - len(opacities_once))
                else:
                    motion_opacities = opacities_once[:max(0, total_frames - delay_frames)]
            
            # Add delay and pad to total_frames
            if len(opacities_once) > 0:
                start_opacity = opacities_once[0]
            else:
                start_opacity = default_opacity
            opacities = [start_opacity] * delay_frames + motion_opacities
            if len(opacities) < total_frames:
                opacities += [opacities[-1]] * (total_frames - len(opacities))
            
            return opacities
        
        opacity_effector.effector_type = "opacity"
        return (opacity_effector,)