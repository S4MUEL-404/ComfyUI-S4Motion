import numpy as np
from PIL import Image
import torch

class VideoCropNode:
    """
    Node for cropping video or image sequences with adjustable position, size, and background.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Crop video or image sequences with custom position, size, and background settings."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "crop_y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "crop_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "crop_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "background_type": (["transparent", "solid_color"], {"default": "transparent"}),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Frame sequence",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def _load_video_frames(self, video_obj):
        """Load frames from ComfyUI VideoFromFile object"""
        from ..dependency_manager import require_dependency
        
        # Require OpenCV for video processing
        if not require_dependency('cv2', 'Video Processing', allow_fallback=False):
            raise Exception("[VideoCrop] OpenCV is required for video processing but not available")
        
        import cv2
        import tempfile
        import os
        
        print(f"[VideoCrop] Processing VideoFromFile object")
        
        try:
            # Try to get video stream source
            if hasattr(video_obj, 'get_stream_source'):
                stream_source = video_obj.get_stream_source()
                print(f"[VideoCrop] Stream source: {stream_source}")
                
                # If stream source is a file path, use it directly
                if isinstance(stream_source, str) and os.path.exists(stream_source):
                    video_path = stream_source
                    print(f"[VideoCrop] Using direct path: {video_path}")
                else:
                    # Save video to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                        print(f"[VideoCrop] Saving video to temporary file: {temp_path}")
                        video_obj.save_to(temp_path)
                        video_path = temp_path
            else:
                # Fallback: save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                    print(f"[VideoCrop] Saving video to temporary file: {temp_path}")
                    video_obj.save_to(temp_path)
                    video_path = temp_path
            
            print(f"[VideoCrop] Loading video from: {video_path}")
            
            # Load video using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"[VideoCrop] Cannot open video file: {video_path}")
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[VideoCrop] Total frames in video: {total_frames}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                
                frame_count += 1
                if total_frames > 0 and frame_count % max(1, total_frames // 10) == 0:
                    print(f"[VideoCrop] Loaded {frame_count}/{total_frames} frames")
            
            cap.release()
            
            # Clean up temporary file if we created one
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"[VideoCrop] Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"[VideoCrop] Warning: Could not clean up temporary file {temp_path}: {e}")
            
            print(f"[VideoCrop] Successfully loaded {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            print(f"[VideoCrop] Error processing video: {e}")
            
            # Try alternative approach: use get_components if available
            if hasattr(video_obj, 'get_components'):
                try:
                    print(f"[VideoCrop] Trying alternative approach with get_components")
                    components = video_obj.get_components()
                    print(f"[VideoCrop] Components type: {type(components)}")
                    
                    # If components is already a list of frames, convert them
                    if isinstance(components, (list, tuple)):
                        frames = []
                        for i, component in enumerate(components):
                            if hasattr(component, 'shape') or isinstance(component, np.ndarray):
                                # Convert numpy array to PIL
                                if isinstance(component, np.ndarray):
                                    if component.dtype == np.float32 or component.dtype == np.float64:
                                        component = (component * 255).astype(np.uint8)
                                    pil_frame = Image.fromarray(component)
                                    frames.append(pil_frame)
                            if i % 10 == 0:
                                print(f"[VideoCrop] Processed component {i + 1}/{len(components)}")
                        
                        print(f"[VideoCrop] Successfully loaded {len(frames)} frames from components")
                        return frames
                    
                except Exception as comp_e:
                    print(f"[VideoCrop] Components approach failed: {comp_e}")
            
            raise Exception(f"[VideoCrop] Failed to load video frames: {e}")

    def _hex_to_rgb(self, hex_color):
        """Convert hex color string to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except (ValueError, IndexError):
            print(f"[VideoCrop] Invalid hex color {hex_color}, using black")
            return (0, 0, 0)

    def _to_pil_images(self, img_input):
        """Convert input to list of PIL images, supporting both single images and image sequences"""
        # Handle ComfyUI VideoFromFile objects
        if hasattr(img_input, '__class__') and 'VideoFromFile' in str(img_input.__class__):
            print(f"[VideoCrop] Detected VideoFromFile object")
            print(f"[VideoCrop] Object type: {type(img_input)}")
            print(f"[VideoCrop] Object attributes: {[attr for attr in dir(img_input) if not attr.startswith('_')]}")
            
            # Try different ways to get the video data or path
            return self._load_video_frames(img_input)
        
        if isinstance(img_input, list):
            return [self._to_pil_image(img) for img in img_input]
        elif isinstance(img_input, np.ndarray) and img_input.ndim == 4:
            return [self._to_pil_image(img_input[i]) for i in range(img_input.shape[0])]
        elif 'torch' in str(type(img_input)) and hasattr(img_input, 'shape') and len(img_input.shape) == 4:
            import torch
            if isinstance(img_input, torch.Tensor):
                arr = img_input.detach().cpu().numpy()
                return [self._to_pil_image(arr[i]) for i in range(arr.shape[0])]
        else:
            return [self._to_pil_image(img_input)]

    def _to_pil_image(self, img):
        """Convert various image formats to PIL Image"""
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
        
        original_shape = arr.shape
        
        # Remove batch dimensions if present
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        
        # Handle single image case
        if arr.ndim == 3:
            if arr.shape[2] in [1, 3, 4]:  # Grayscale, RGB, RGBA
                pass
            else:
                if arr.shape[0] in [1, 3, 4]:
                    arr = arr.transpose(1, 2, 0)
        elif arr.ndim == 2:
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

    def _frames_to_comfyui_format(self, frames):
        """Convert PIL frames to ComfyUI format"""
        import torch
        
        frame_arrays = []
        for frame in frames:
            arr = np.array(frame)
            if arr.ndim == 3:
                arr = arr[None, ...]
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            frame_arrays.append(arr)
        
        if frame_arrays:
            stacked = np.concatenate(frame_arrays, axis=0)
            return torch.from_numpy(stacked)
        else:
            return torch.tensor([])

    def crop_frame(self, img, crop_x, crop_y, crop_width, crop_height, background_type, background_color_rgb):
        """Crop a single frame with background handling"""
        # Get original dimensions
        orig_width, orig_height = img.size
        
        # Determine final image mode
        if background_type == "transparent":
            final_mode = "RGBA"
            bg_color = (0, 0, 0, 0)  # Transparent
        else:
            # For solid color, preserve original alpha if present, otherwise use RGB
            final_mode = "RGBA" if img.mode in ["RGBA", "LA"] else "RGB"
            if final_mode == "RGBA":
                bg_color = background_color_rgb + (255,)  # Solid color with full alpha
            else:
                bg_color = background_color_rgb
        
        # Create output canvas
        output = Image.new(final_mode, (crop_width, crop_height), bg_color)
        
        # Convert input image to match output mode
        if img.mode != final_mode:
            img = img.convert(final_mode)
        
        # Calculate source and destination areas
        src_left = max(0, crop_x)
        src_top = max(0, crop_y)
        src_right = min(orig_width, crop_x + crop_width)
        src_bottom = min(orig_height, crop_y + crop_height)
        
        # Skip if no overlap
        if src_left >= src_right or src_top >= src_bottom:
            return output
        
        # Calculate destination position
        dst_left = max(0, -crop_x)
        dst_top = max(0, -crop_y)
        
        # Crop the source region
        cropped = img.crop((src_left, src_top, src_right, src_bottom))
        
        # Paste onto output canvas
        if final_mode == "RGBA" and cropped.mode == "RGBA":
            output.paste(cropped, (dst_left, dst_top), cropped)
        else:
            output.paste(cropped, (dst_left, dst_top))
        
        return output

    def process(self, crop_x=0, crop_y=0, crop_width=512, crop_height=512, background_type="transparent", background_color="#000000", video=None, image=None):
        print(f"[VideoCrop] Starting crop operation...")
        print(f"[VideoCrop] Crop region: ({crop_x}, {crop_y}) - {crop_width}x{crop_height}")
        print(f"[VideoCrop] Background: {background_type}")
        if background_type == "solid_color":
            print(f"[VideoCrop] Background color: {background_color}")
        
        # Determine input source
        if video is not None and image is not None:
            # Both inputs are connected - warn user and use video
            print(f"[VideoCrop] ⚠️  WARNING: Both VIDEO and IMAGE inputs are connected!")
            print(f"[VideoCrop] ⚠️  Using VIDEO input and ignoring IMAGE input")
            print(f"[VideoCrop] 💡 Please disconnect one of the inputs for cleaner workflow")
            input_media = video
        elif video is not None:
            input_media = video
            print(f"[VideoCrop] ✅ Using VIDEO input")
        elif image is not None:
            input_media = image
            print(f"[VideoCrop] ✅ Using IMAGE input")
        else:
            raise Exception("[VideoCrop] ❌ No input provided. Please connect either video or image input.")
        
        # Convert input to PIL images
        input_images = self._to_pil_images(input_media)
        print(f"[VideoCrop] Processing {len(input_images)} frame(s)")
        
        # Parse background color
        background_color_rgb = self._hex_to_rgb(background_color)
        
        # Process each frame
        cropped_frames = []
        for i, img in enumerate(input_images):
            if i % max(1, len(input_images) // 10) == 0 or i == len(input_images) - 1:
                progress = (i + 1) / len(input_images) * 100
                print(f"[VideoCrop] Progress: {progress:.1f}% ({i + 1}/{len(input_images)})")
            
            cropped_frame = self.crop_frame(img, crop_x, crop_y, crop_width, crop_height, background_type, background_color_rgb)
            cropped_frames.append(cropped_frame)
        
        # Convert to ComfyUI format
        print(f"[VideoCrop] Converting frames to output format...")
        frame_sequence = self._frames_to_comfyui_format(cropped_frames)
        print(f"[VideoCrop] Crop operation completed successfully!")
        
        return (frame_sequence,)