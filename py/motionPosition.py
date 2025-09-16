import numpy as np

try:
    from bezier import Curve
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False
    print("[MotionPosition] Warning: bezier library not found. Install with: pip install bezier")
    print("[MotionPosition] Falling back to basic easing functions.")

class MotionPositionNode:
    """
    Node for controlling the position animation of layer image on background image.
    Provides position values for each frame based on parameters.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Position effector for layer image animation. Supports motion curve."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "default_x": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "default_y": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "target_x": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "target_y": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "motion_curve": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear"}),
                "inverse": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("S4_POSITION_EFFECTOR",)
    RETURN_NAMES = ("Position",)

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
                    print(f"[MotionPosition] Bezier curve error for {curve}: {e}, falling back to basic")
        
        # Fallback to enhanced mathematical easing functions
        if curve == "ease_in_out":
            return t * t * (3 - 2 * t)  # Smoothstep
        elif curve == "ease_in":
            return t * t * t  # Cubic ease-in for smoother effect
        elif curve == "ease_out":
            return 1 - (1 - t) ** 3  # Cubic ease-out for smoother effect
        else:
            return t
    


    def process(self, default_x=0.0, default_y=0.0, target_x=0.0, target_y=0.0, duration=1.0, delay=0.0, motion_curve="linear", inverse=False, loop=False, **kwargs):
        def position_effector(total_frames, total_time):
            fps = total_frames / total_time if total_time > 0 else 1
            delay_frames = int(delay * fps)
            motion_frames = max(1, int(duration * fps))
            
            # Handle inverse motion
            if inverse:
                # First half: default to target
                positions_once = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    x = default_x + (target_x - default_x) * curved_t
                    y = default_y + (target_y - default_y) * curved_t
                    positions_once.append((float(x), float(y)))
                
                # Second half: target to default
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    x = target_x + (default_x - target_x) * curved_t
                    y = target_y + (default_y - target_y) * curved_t
                    positions_once.append((float(x), float(y)))
                
                motion_frames *= 2
            else:
                # Single motion: default to target
                positions_once = []
                for i in range(motion_frames):
                    t = i / (motion_frames - 1) if motion_frames > 1 else 0
                    curved_t = self._motion_curve_func(motion_curve, t)
                    x = default_x + (target_x - default_x) * curved_t
                    y = default_y + (target_y - default_y) * curved_t
                    positions_once.append((float(x), float(y)))
                
                # Debug: print first and last positions to verify target is reached
                if len(positions_once) > 0:
                    print(f"[MotionPosition] Motion curve: {motion_curve}")
                    print(f"[MotionPosition] Start: ({positions_once[0][0]:.2f}, {positions_once[0][1]:.2f})")
                    print(f"[MotionPosition] End: ({positions_once[-1][0]:.2f}, {positions_once[-1][1]:.2f})")
                    print(f"[MotionPosition] Target: ({target_x}, {target_y})")
                
                # Handle loop and delay
            if loop:
                repeat = (total_frames - delay_frames + motion_frames - 1) // motion_frames
                motion_positions = (positions_once * repeat)[:max(0, total_frames - delay_frames)]
            else:
                if len(positions_once) < total_frames - delay_frames:
                    motion_positions = positions_once + [positions_once[-1]] * (total_frames - delay_frames - len(positions_once))
                else:
                    motion_positions = positions_once[:max(0, total_frames - delay_frames)]
            
            # Add delay and pad to total_frames
            if len(positions_once) > 0:
                start_pos = positions_once[0]
            else:
                start_pos = (default_x, default_y)
            positions = [start_pos] * delay_frames + motion_positions
            if len(positions) < total_frames:
                positions += [positions[-1]] * (total_frames - len(positions))
            
            return positions
        
        position_effector.is_path_mode = False
        position_effector.effector_type = "position"
        return (position_effector,)

