import numpy as np
from PIL import Image
import torch


class VideoInfoNode:
    """
    Node for analyzing video or image sequence properties.
    Outputs width, height, frame count, duration, and other useful information.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Analyze video or image sequence properties and output detailed information."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "frame_count", "duration", "fps", "info")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def _load_video_frames(self, video_obj):
        """Load frames from ComfyUI VideoFromFile object"""
        from ..dependency_manager import require_dependency
        
        # Require OpenCV for video processing
        if not require_dependency('cv2', 'Video Processing', allow_fallback=False):
            raise Exception("[VideoInfo] OpenCV is required for video processing but not available")
        
        import cv2
        import tempfile
        import os
        
        print(f"[VideoInfo] Processing VideoFromFile object")
        
        try:
            # Try to get video stream source
            if hasattr(video_obj, 'get_stream_source'):
                stream_source = video_obj.get_stream_source()
                print(f"[VideoInfo] Stream source: {stream_source}")
                
                # If stream source is a file path, use it directly
                if isinstance(stream_source, str) and os.path.exists(stream_source):
                    video_path = stream_source
                    print(f"[VideoInfo] Using direct path: {video_path}")
                else:
                    # Save video to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                        print(f"[VideoInfo] Saving video to temporary file: {temp_path}")
                        video_obj.save_to(temp_path)
                        video_path = temp_path
            else:
                # Fallback: save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                    print(f"[VideoInfo] Saving video to temporary file: {temp_path}")
                    video_obj.save_to(temp_path)
                    video_path = temp_path
            
            print(f"[VideoInfo] Loading video from: {video_path}")
            
            # Load video using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"[VideoInfo] Cannot open video file: {video_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Load all frames to verify count
            frames = []
            actual_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                actual_frame_count += 1
            
            cap.release()
            
            # Clean up temporary file if we created one
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"[VideoInfo] Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"[VideoInfo] Warning: Could not clean up temporary file {temp_path}: {e}")
            
            # Use actual frame count if different from metadata
            final_frame_count = actual_frame_count if actual_frame_count > 0 else frame_count
            
            print(f"[VideoInfo] Video analysis complete:")
            print(f"[VideoInfo] - Dimensions: {width}x{height}")
            print(f"[VideoInfo] - Frame Count: {final_frame_count} (metadata: {frame_count})")
            print(f"[VideoInfo] - FPS: {fps}")
            
            return frames, width, height, final_frame_count, fps
            
        except Exception as e:
            print(f"[VideoInfo] Error processing video: {e}")
            
            # Try alternative approach: use get_components if available
            if hasattr(video_obj, 'get_components'):
                try:
                    print(f"[VideoInfo] Trying alternative approach with get_components")
                    components = video_obj.get_components()
                    print(f"[VideoInfo] Components type: {type(components)}")
                    
                    # If components is already a list of frames, convert them
                    if isinstance(components, (list, tuple)):
                        frames = []
                        width, height = 0, 0
                        
                        for i, component in enumerate(components):
                            if hasattr(component, 'shape') or isinstance(component, np.ndarray):
                                # Convert numpy array to PIL
                                if isinstance(component, np.ndarray):
                                    if component.dtype == np.float32 or component.dtype == np.float64:
                                        component = (component * 255).astype(np.uint8)
                                    pil_frame = Image.fromarray(component)
                                    frames.append(pil_frame)
                                    
                                    # Get dimensions from first frame
                                    if i == 0:
                                        width, height = pil_frame.size
                        
                        frame_count = len(frames)
                        fps = 30.0  # Default FPS when not available
                        
                        print(f"[VideoInfo] Alternative analysis complete:")
                        print(f"[VideoInfo] - Dimensions: {width}x{height}")
                        print(f"[VideoInfo] - Frame Count: {frame_count}")
                        print(f"[VideoInfo] - FPS: {fps} (default)")
                        
                        return frames, width, height, frame_count, fps
                    
                except Exception as comp_e:
                    print(f"[VideoInfo] Components approach failed: {comp_e}")
            
            raise Exception(f"[VideoInfo] Failed to analyze video: {e}")

    def _to_pil_images(self, img_input):
        """Convert input to list of PIL images, supporting both single images and image sequences"""
        # Handle ComfyUI VideoFromFile objects
        if hasattr(img_input, '__class__') and 'VideoFromFile' in str(img_input.__class__):
            print(f"[VideoInfo] Detected VideoFromFile object")
            print(f"[VideoInfo] Object type: {type(img_input)}")
            print(f"[VideoInfo] Object attributes: {[attr for attr in dir(img_input) if not attr.startswith('_')]}")
            
            # Try different ways to get the video data or path
            frames, width, height, frame_count, fps = self._load_video_frames(img_input)
            return frames, width, height, frame_count, fps
        
        # Handle image sequences
        frames = []
        if isinstance(img_input, list):
            frames = [self._to_pil_image(img) for img in img_input]
        elif isinstance(img_input, np.ndarray) and img_input.ndim == 4:
            frames = [self._to_pil_image(img_input[i]) for i in range(img_input.shape[0])]
        elif 'torch' in str(type(img_input)) and hasattr(img_input, 'shape') and len(img_input.shape) == 4:
            import torch
            if isinstance(img_input, torch.Tensor):
                arr = img_input.detach().cpu().numpy()
                frames = [self._to_pil_image(arr[i]) for i in range(arr.shape[0])]
        else:
            frames = [self._to_pil_image(img_input)]
        
        # Get dimensions from first frame
        width, height = 0, 0
        if frames:
            width, height = frames[0].size
        
        frame_count = len(frames)
        fps = 30.0  # Default FPS for image sequences
        
        print(f"[VideoInfo] Image sequence analysis:")
        print(f"[VideoInfo] - Dimensions: {width}x{height}")
        print(f"[VideoInfo] - Frame Count: {frame_count}")
        print(f"[VideoInfo] - FPS: {fps} (default for image sequence)")
        
        return frames, width, height, frame_count, fps

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
            if arr.shape[-1] == 3 or arr.shape[-1] == 4:
                # RGB or RGBA image
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = (arr * 255).astype(np.uint8)
                return Image.fromarray(arr)
            elif arr.shape[0] == 3 or arr.shape[0] == 4:
                # Channel-first format
                arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = (arr * 255).astype(np.uint8)
                return Image.fromarray(arr)
        elif arr.ndim == 2:
            # Grayscale image
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        
        raise Exception(f"[VideoInfo] Unsupported image format: shape {original_shape}")

    def process(self, video=None, image=None, **kwargs):
        
        print(f"[VideoInfo] Starting video/image analysis")
        
        # Process input
        if video is not None:
            print(f"[VideoInfo] Analyzing video input")
            frames, width, height, frame_count, detected_fps = self._to_pil_images(video)
            # Use detected FPS for video
            final_fps = detected_fps if detected_fps > 0 else 30.0
        elif image is not None:
            print(f"[VideoInfo] Analyzing image sequence")
            frames, width, height, frame_count, detected_fps = self._to_pil_images(image)
            # For image sequences, FPS is not applicable, use 0 or N/A
            final_fps = 0.0  # Image sequences don't have inherent FPS
        else:
            raise Exception("[VideoInfo] Either video or image input is required")
        
        # Calculate duration (only meaningful for videos)
        if video is not None:
            duration = frame_count / final_fps if final_fps > 0 else 0.0
        else:
            duration = 0.0  # Image sequences don't have duration
        
        # Create info string
        info_parts = [
            f"Dimensions: {width}x{height}",
            f"Frame Count: {frame_count}"
        ]
        
        if video is not None:
            info_parts.extend([
                f"FPS: {final_fps:.2f}",
                f"Duration: {duration:.2f}s",
                "Type: Video"
            ])
        else:
            info_parts.extend([
                "FPS: N/A",
                "Duration: N/A",
                "Type: Image Sequence"
            ])
        
        info_string = " | ".join(info_parts)
        
        print(f"[VideoInfo] Analysis complete:")
        print(f"[VideoInfo] - Width: {width}")
        print(f"[VideoInfo] - Height: {height}")
        print(f"[VideoInfo] - Frame Count: {frame_count}")
        print(f"[VideoInfo] - FPS: {final_fps:.2f}")
        print(f"[VideoInfo] - Duration: {duration:.2f}s")
        print(f"[VideoInfo] - Info: {info_string}")
        
        return (width, height, frame_count, duration, final_fps, info_string)