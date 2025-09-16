import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d, splprep, splev
from skimage import morphology, measure
import math

try:
    from bezier import Curve
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False
    print("[MotionPositionOnPath] Warning: bezier library not found. Install with: pip install bezier")
    print("[MotionPositionOnPath] Falling back to basic easing functions.")

class MotionPositionOnPathNode:
    """
    Node for controlling the position animation of layer image along a path defined by an image.
    Provides position values for each frame based on path tracing.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Position effector for layer image animation along a custom path. Supports motion curve."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_image": ("IMAGE", {"default": None}),
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

    def _to_pil_image(self, img):
        """Convert input image to PIL format"""
        if 'torch' in str(type(img)):
            import torch
            if isinstance(img, torch.Tensor):
                arr = img.detach().cpu().numpy()
            else:
                arr = np.array(img)
        elif isinstance(img, np.ndarray):
            arr = img
        elif isinstance(img, Image.Image):
            return img
        else:
            raise Exception("Unsupported image input type: {}".format(type(img)))
        
        # Handle various array shapes from ComfyUI nodes
        original_shape = arr.shape
        
        # Remove batch dimensions if present
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        
        # Handle single image case
        if arr.ndim == 3:
            # Expected shape: (height, width, channels)
            if arr.shape[2] in [1, 3, 4]:  # Grayscale, RGB, RGBA
                pass
            else:
                # Try to reshape if channels are in wrong position
                if arr.shape[0] in [1, 3, 4]:
                    arr = arr.transpose(1, 2, 0)
        elif arr.ndim == 2:
            # Grayscale image, add channel dimension
            arr = arr[..., None]
        else:
            raise Exception(f"Cannot handle array shape: {original_shape}")
        
        # Convert data type
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        
        return Image.fromarray(arr)

    # auto direction related utilities removed

    def _motion_curve_func(self, curve, t):
        """Professional easing functions for smooth animation"""
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
                    print(f"[MotionPositionOnPath] Bezier curve error for {curve}: {e}, falling back to basic")
        
        # Fallback to enhanced mathematical easing functions
        if curve == "ease_in_out":
            return t * t * (3 - 2 * t)  # Smoothstep
        elif curve == "ease_in":
            return t * t * t  # Cubic ease-in for smoother effect
        elif curve == "ease_out":
            return 1 - (1 - t) ** 3  # Cubic ease-out for smoother effect
        else:
            return t

    def _extract_path_from_image(self, path_image):
        """Extract path coordinates from the input image"""
        print(f"[MotionPositionOnPath] Extracting path from image...")
        
        # Convert PIL to numpy array
        img_array = np.array(path_image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 3:  # RGB
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            elif img_array.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            else:
                gray = img_array[:, :, 0]  # Use first channel
        else:
            gray = img_array
        
        # Normalize to 0-1 range
        gray = gray.astype(np.float32) / 255.0
        
        print(f"[MotionPositionOnPath] Image size: {gray.shape}")
        
        # Create binary mask based on fixed threshold
        # Assume dark pixels (below 0.5) represent the path
        threshold = 0.5
        binary = gray < threshold
        
        # Clean up the binary image
        # Remove small noise
        binary = morphology.remove_small_objects(binary, min_size=50)
        
        # Fill small holes
        binary = morphology.remove_small_holes(binary, area_threshold=100)
        
        # Skeletonize to get centerline
        skeleton = morphology.skeletonize(binary)
        
        print(f"[MotionPositionOnPath] Skeleton pixels: {np.sum(skeleton)}")
        
        if np.sum(skeleton) < 10:
            print("[MotionPositionOnPath] Warning: Very few skeleton pixels found. Trying alternative method...")
            # Try edge detection as fallback
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            skeleton = edges > 0
        
        # Find connected components and select the largest one
        labeled = measure.label(skeleton)
        if labeled.max() == 0:
            raise Exception("No path found in the image. Please check the path image and threshold value.")
        
        # Get the largest connected component
        props = measure.regionprops(labeled)
        largest_component = max(props, key=lambda x: x.area)
        main_path = labeled == largest_component.label
        
        # Extract coordinates of the path
        y_coords, x_coords = np.where(main_path)
        path_points = list(zip(x_coords, y_coords))
        
        print(f"[MotionPositionOnPath] Raw path points: {len(path_points)}")
        
        if len(path_points) < 2:
            raise Exception("Path too short. Please provide a longer path in the image.")
        
        # Sort points to create a continuous path
        sorted_path = self._sort_path_points(path_points)
        
        # Apply optimal smoothing for best quality
        if len(sorted_path) > 3:
            sorted_path = self._smooth_path(sorted_path, 0.3)  # Optimal smoothing value
        
        print(f"[MotionPositionOnPath] Final path points: {len(sorted_path)}")
        return sorted_path

    def _sort_path_points(self, points):
        """Order skeleton pixels into a continuous path without shortcuts.

        This traces along 8-connected neighbors from one endpoint to the other.
        If no endpoints exist (closed loop), it performs a full walk starting
        from the upper-leftmost pixel to cover the loop once.
        """
        if len(points) < 2:
            return points

        point_set = set(points)

        def neighbors(p):
            x, y = p
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    q = (x + dx, y + dy)
                    if q in point_set:
                        yield q

        # Find endpoints: pixels with exactly one neighbor
        endpoints = [p for p in points if sum(1 for _ in neighbors(p)) == 1]

        # Choose start
        if len(endpoints) >= 1:
            start = min(endpoints, key=lambda p: (p[1], p[0]))  # top-most, then left-most
        else:
            # Closed loop: pick the upper-leftmost point
            start = min(points, key=lambda p: (p[1], p[0]))

        ordered = [start]
        visited = {start}
        prev = None
        current = start

        while True:
            nbrs = [q for q in neighbors(current) if q != prev]
            # Prefer unvisited neighbor; if multiple, pick the one with smallest turn
            unvisited = [q for q in nbrs if q not in visited]
            if not unvisited:
                # If closed loop, we might come back to start with only prev as neighbor
                break
            # Choose the next neighbor
            next_pt = None
            if len(unvisited) == 1:
                next_pt = unvisited[0]
            else:
                # Heuristic: choose neighbor with smallest Euclidean distance to any endpoint (to keep along main branch)
                ref_points = endpoints if endpoints else [start]
                def score(q):
                    return min((q[0]-r[0])**2 + (q[1]-r[1])**2 for r in ref_points)
                next_pt = min(unvisited, key=score)

            prev = current
            current = next_pt
            ordered.append(current)
            visited.add(current)

        return ordered

    def _smooth_path(self, path_points, smoothing_factor):
        """Smooth the path using spline interpolation"""
        if len(path_points) < 4:
            return path_points
        
        try:
            # Convert to numpy arrays
            points = np.array(path_points)
            x = points[:, 0]
            y = points[:, 1]
            
            # Create parameter array
            t = np.linspace(0, 1, len(points))
            
            # Fit spline with smoothing
            s = smoothing_factor * len(points)  # Smoothing parameter
            tck, u = splprep([x, y], s=s, per=False)
            
            # Generate smooth path with more points
            u_new = np.linspace(0, 1, max(len(points), 100))
            smooth_x, smooth_y = splev(u_new, tck)
            
            # Convert back to list of tuples
            smooth_path = [(float(smooth_x[i]), float(smooth_y[i])) for i in range(len(smooth_x))]
            
            print(f"[MotionPositionOnPath] Path smoothed: {len(path_points)} -> {len(smooth_path)} points")
            return smooth_path
            
        except Exception as e:
            print(f"[MotionPositionOnPath] Smoothing failed: {e}, using original path")
            return path_points

    def _resample_path(self, path_points, num_frames):
        """Resample path to exact number of frames"""
        if len(path_points) < 2:
            return path_points * num_frames
        
        # Calculate cumulative distances along the path
        points = np.array(path_points)
        distances = [0]
        for i in range(1, len(points)):
            dist = math.sqrt((points[i][0] - points[i-1][0])**2 + 
                           (points[i][1] - points[i-1][1])**2)
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        if total_distance == 0:
            return path_points * num_frames
        
        # Normalize distances to [0, 1]
        normalized_distances = [d / total_distance for d in distances]
        
        # Create interpolation functions
        x_coords = [p[0] for p in path_points]
        y_coords = [p[1] for p in path_points]
        
        fx = interp1d(normalized_distances, x_coords, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        fy = interp1d(normalized_distances, y_coords, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        
        # Sample at regular intervals
        sample_positions = np.linspace(0, 1, num_frames)
        resampled_x = fx(sample_positions)
        resampled_y = fy(sample_positions)
        
        resampled_path = [(float(resampled_x[i]), float(resampled_y[i])) 
                         for i in range(len(resampled_x))]
        
        return resampled_path

    def _calculate_orientations(self, path_points):
        """Calculate orientation angles based on path direction"""
        if len(path_points) < 2:
            return [0.0] * len(path_points)
        
        orientations = []
        for i in range(len(path_points)):
            if i == 0:
                # Use direction to next point
                dx = path_points[i+1][0] - path_points[i][0]
                dy = path_points[i+1][1] - path_points[i][1]
            elif i == len(path_points) - 1:
                # Use direction from previous point
                dx = path_points[i][0] - path_points[i-1][0]
                dy = path_points[i][1] - path_points[i-1][1]
            else:
                # Use average direction from previous and to next point
                dx1 = path_points[i][0] - path_points[i-1][0]
                dy1 = path_points[i][1] - path_points[i-1][1]
                dx2 = path_points[i+1][0] - path_points[i][0]
                dy2 = path_points[i+1][1] - path_points[i][1]
                dx = (dx1 + dx2) / 2
                dy = (dy1 + dy2) / 2
            
            # Calculate angle in degrees
            angle = math.degrees(math.atan2(dy, dx))
            orientations.append(float(angle))
        
        return orientations

    def process(self, path_image, duration=1.0, delay=0.0, motion_curve="linear", 
                inverse=False, loop=False, **kwargs):
        
        def position_effector(total_frames, total_time):
            print(f"[MotionPositionOnPath] Processing path animation...")
            print(f"[MotionPositionOnPath] Duration: {duration}s, Delay: {delay}s")
            print(f"[MotionPositionOnPath] Inverse: {inverse}")
            print(f"[MotionPositionOnPath] Total frames: {total_frames}, Total time: {total_time}s")
            
            fps = total_frames / total_time if total_time > 0 else 1
            delay_frames = int(delay * fps)
            motion_frames = max(1, int(duration * fps))
            
            print(f"[MotionPositionOnPath] FPS: {fps}, Delay frames: {delay_frames}, Motion frames: {motion_frames}")
            print(f"[MotionPositionOnPath] Duration ratio: {duration}/{total_time} = {duration/total_time if total_time > 0 else 0:.2f}")
            
            # Convert path image to PIL
            pil_image = self._to_pil_image(path_image)
            
            # Extract path from image
            raw_path = self._extract_path_from_image(pil_image)
            
            # Get the starting position (first point of the path)
            if len(raw_path) > 0:
                start_x, start_y = raw_path[0]
                print(f"[MotionPositionOnPath] Path start position: ({start_x:.1f}, {start_y:.1f})")
            else:
                start_x, start_y = 0, 0
            
            # Resample path to motion frames first
            path_once = self._resample_path(raw_path, motion_frames)
            
            # Handle inverse motion
            if inverse:
                # Create return path by reversing (excluding the last point to avoid duplication)
                path_return = path_once[:-1][::-1]  # Reverse and exclude last point
                
                # Combine both directions: forward + reverse
                path_once = path_once + path_return
                motion_frames *= 2
            
            # Apply motion curve and convert to offset positions
            curved_path = []
            for i in range(motion_frames):
                t = i / (motion_frames - 1) if motion_frames > 1 else 0
                curved_t = self._motion_curve_func(motion_curve, t)
                
                # Interpolate along the path based on curved time
                path_index = curved_t * (len(path_once) - 1)
                index_low = int(path_index)
                index_high = min(index_low + 1, len(path_once) - 1)
                
                if index_low == index_high:
                    point = path_once[index_low]
                else:
                    # Linear interpolation between path points
                    alpha = path_index - index_low
                    point_low = path_once[index_low]
                    point_high = path_once[index_high]
                    point = (
                        point_low[0] + alpha * (point_high[0] - point_low[0]),
                        point_low[1] + alpha * (point_high[1] - point_low[1])
                    )
                
                # Keep absolute path coordinates - Motion Position also returns absolute coordinates
                # The motionConfig will handle the positioning correctly
                curved_path.append(point)
            
            # Auto direction removed
            
            # Handle loop and delay
            if loop:
                target_len = max(0, total_frames - delay_frames)
                if inverse:
                    # In inverse mode, curved_path already contains forward+backward once
                    repeat = (target_len + motion_frames - 1) // motion_frames
                    motion_positions = (curved_path * repeat)[:target_len]
                else:
                    # Forward-only looping: end -> instant jump to start
                    repeat = (target_len + motion_frames - 1) // motion_frames
                    motion_positions = (curved_path * repeat)[:target_len]
            else:
                if len(curved_path) < total_frames - delay_frames:
                    motion_positions = curved_path + [curved_path[-1]] * (total_frames - delay_frames - len(curved_path))
                else:
                    motion_positions = curved_path[:max(0, total_frames - delay_frames)]
            
            # Add delay and pad to total_frames
            if len(curved_path) > 0:
                start_pos = curved_path[0]  # First position on path
            else:
                start_pos = (0.0, 0.0)
            
            positions = [start_pos] * delay_frames + motion_positions
            if len(positions) < total_frames:
                positions += [positions[-1]] * (total_frames - len(positions))
            
            print(f"[MotionPositionOnPath] Path animation completed")
            print(f"[MotionPositionOnPath] Path length: {len(raw_path)} points")
            print(f"[MotionPositionOnPath] Total frames: {total_frames}, Motion frames: {motion_frames}")
            print(f"[MotionPositionOnPath] Positions array length: {len(positions)}")
            print(f"[MotionPositionOnPath] Start pos: ({positions[0][0]:.1f}, {positions[0][1]:.1f})")
            print(f"[MotionPositionOnPath] End pos: ({positions[-1][0]:.1f}, {positions[-1][1]:.1f})")
            if len(positions) > 10:
                print(f"[MotionPositionOnPath] Mid pos: ({positions[len(positions)//2][0]:.1f}, {positions[len(positions)//2][1]:.1f})")
            return positions
        
        position_effector.is_path_mode = True
        position_effector.effector_type = "position"
        return (position_effector,)