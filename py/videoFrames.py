import numpy as np
from PIL import Image
import torch

class VideoFramesNode:
    """
    Node for extracting specific frame ranges from video or image sequences.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Extract specific frames from video or image sequences with custom frame range and background settings."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "frame_count": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
                "background_type": (["transparent", "solid_color"], {"default": "transparent"}),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Frame",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def _load_video_frames(self, video_obj):
        """Load frames from ComfyUI VideoFromFile object"""
        from ..dependency_manager import require_dependency
        
        # Require OpenCV for video processing
        if not require_dependency('cv2', 'Video Processing', allow_fallback=False):
            raise Exception("[VideoFrames] OpenCV is required for video processing but not available")
        
        import cv2
        import tempfile
        import os
        
        print(f"[VideoFrames] Processing VideoFromFile object")
        
        try:
            # Try to get video stream source
            if hasattr(video_obj, 'get_stream_source'):
                stream_source = video_obj.get_stream_source()
                print(f"[VideoFrames] Stream source: {stream_source}")
                
                # If stream source is a file path, use it directly
                if isinstance(stream_source, str) and os.path.exists(stream_source):
                    video_path = stream_source
                    print(f"[VideoFrames] Using direct path: {video_path}")
                else:
                    # Save video to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                        print(f"[VideoFrames] Saving video to temporary file: {temp_path}")
                        video_obj.save_to(temp_path)
                        video_path = temp_path
            else:
                # Fallback: save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                    print(f"[VideoFrames] Saving video to temporary file: {temp_path}")
                    video_obj.save_to(temp_path)
                    video_path = temp_path
            
            print(f"[VideoFrames] Loading video from: {video_path}")
            
            # Load video using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"[VideoFrames] Cannot open video file: {video_path}")
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[VideoFrames] Total frames in video: {total_frames}")
            
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
                    print(f"[VideoFrames] Loaded {frame_count}/{total_frames} frames")
            
            cap.release()
            
            # Clean up temporary file if we created one
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"[VideoFrames] Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"[VideoFrames] Warning: Could not clean up temporary file {temp_path}: {e}")
            
            print(f"[VideoFrames] Successfully loaded {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            print(f"[VideoFrames] Error processing video: {e}")
            
            # Try alternative approach: use get_components if available
            if hasattr(video_obj, 'get_components'):
                try:
                    print(f"[VideoFrames] Trying alternative approach with get_components")
                    components = video_obj.get_components()
                    print(f"[VideoFrames] Components type: {type(components)}")
                    
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
                                print(f"[VideoFrames] Processed component {i + 1}/{len(components)}")
                        
                        print(f"[VideoFrames] Successfully loaded {len(frames)} frames from components")
                        return frames
                    
                except Exception as comp_e:
                    print(f"[VideoFrames] Components approach failed: {comp_e}")
            
            raise Exception(f"[VideoFrames] Failed to load video frames: {e}")

    def _hex_to_rgb(self, hex_color):
        """Convert hex color string to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except (ValueError, IndexError):
            print(f"[VideoFrames] Invalid hex color {hex_color}, using black")
            return (0, 0, 0)

    def _to_pil_images(self, img_input):
        """Convert input to list of PIL images, supporting both single images and image sequences"""
        # Handle ComfyUI VideoFromFile objects
        if hasattr(img_input, '__class__') and 'VideoFromFile' in str(img_input.__class__):
            print(f"[VideoFrames] Detected VideoFromFile object")
            print(f"[VideoFrames] Object type: {type(img_input)}")
            print(f"[VideoFrames] Object attributes: {[attr for attr in dir(img_input) if not attr.startswith('_')]}")
            
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

    def apply_background(self, img, background_type, background_color_rgb):
        """Apply background settings to frame"""
        if background_type == "transparent":
            # Ensure image has alpha channel
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            return img
        else:
            # Apply solid color background
            width, height = img.size
            
            # Determine output mode based on input
            if img.mode in ["RGBA", "LA"]:
                final_mode = "RGBA"
                bg_color = background_color_rgb + (255,)
            else:
                final_mode = "RGB"
                bg_color = background_color_rgb
            
            # Create background
            background = Image.new(final_mode, (width, height), bg_color)
            
            # Composite if input has alpha, otherwise just convert
            if img.mode in ["RGBA", "LA"]:
                img_converted = img.convert("RGBA")
                background.paste(img_converted, (0, 0), img_converted)
            else:
                img_converted = img.convert(final_mode)
                background.paste(img_converted, (0, 0))
            
            return background

    def process(self, start_frame=0, frame_count=1, background_type="transparent", background_color="#000000", video=None, image=None):
        print(f"[VideoFrames] Starting frame extraction...")
        print(f"[VideoFrames] Frame range: {start_frame} to {start_frame + frame_count - 1} ({frame_count} frame(s))")
        print(f"[VideoFrames] Background: {background_type}")
        if background_type == "solid_color":
            print(f"[VideoFrames] Background color: {background_color}")
        
        # Determine input source
        if video is not None and image is not None:
            # Both inputs are connected - warn user and use video
            print(f"[VideoFrames] ⚠️  WARNING: Both VIDEO and IMAGE inputs are connected!")
            print(f"[VideoFrames] ⚠️  Using VIDEO input and ignoring IMAGE input")
            print(f"[VideoFrames] 💡 Please disconnect one of the inputs for cleaner workflow")
            input_media = video
        elif video is not None:
            input_media = video
            print(f"[VideoFrames] ✅ Using VIDEO input")
        elif image is not None:
            input_media = image
            print(f"[VideoFrames] ✅ Using IMAGE input")
        else:
            raise Exception("[VideoFrames] ❌ No input provided. Please connect either video or image input.")
        
        # Convert input to PIL images
        input_images = self._to_pil_images(input_media)
        total_frames = len(input_images)
        print(f"[VideoFrames] Input has {total_frames} frame(s)")
        
        # Validate frame range
        if start_frame >= total_frames:
            print(f"[VideoFrames] Warning: start_frame ({start_frame}) >= total frames ({total_frames}), using last frame")
            start_frame = total_frames - 1
            frame_count = 1
        
        # Calculate actual end frame
        end_frame = min(start_frame + frame_count, total_frames)
        actual_frame_count = end_frame - start_frame
        
        if actual_frame_count != frame_count:
            print(f"[VideoFrames] Adjusted frame count to {actual_frame_count} (limited by available frames)")
        
        # Parse background color
        background_color_rgb = self._hex_to_rgb(background_color)
        
        # Extract frames
        extracted_frames = []
        for i in range(start_frame, end_frame):
            print(f"[VideoFrames] Extracting frame {i}")
            frame = input_images[i]
            
            # Apply background settings
            frame = self.apply_background(frame, background_type, background_color_rgb)
            extracted_frames.append(frame)
        
        # Convert to ComfyUI format
        print(f"[VideoFrames] Converting frames to output format...")
        frame_output = self._frames_to_comfyui_format(extracted_frames)
        print(f"[VideoFrames] Frame extraction completed successfully! Extracted {len(extracted_frames)} frame(s)")
        
        return (frame_output,)