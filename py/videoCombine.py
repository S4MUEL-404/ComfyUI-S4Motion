import numpy as np
from PIL import Image
import torch


class VideoCombineNode:
    """
    Node for combining two videos or image sequences into a single sequence by concatenation.
    Combines sequences in time - first sequence followed by second sequence.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Combine two videos or image sequences by concatenating them in time sequence."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "video_1": ("VIDEO", {"default": None}),
                "video_2": ("VIDEO", {"default": None}),
                "image_1": ("IMAGE", {"default": None}),
                "image_2": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Combined Sequence",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def _load_video_frames(self, video_obj):
        """Load frames from ComfyUI VideoFromFile object"""
        from ..dependency_manager import require_dependency
        
        # Require OpenCV for video processing
        if not require_dependency('cv2', 'Video Processing', allow_fallback=False):
            raise Exception("[VideoCombine] OpenCV is required for video processing but not available")
        
        import cv2
        import tempfile
        import os
        
        print(f"[VideoCombine] Processing VideoFromFile object")
        
        try:
            # Try to get video stream source
            if hasattr(video_obj, 'get_stream_source'):
                stream_source = video_obj.get_stream_source()
                print(f"[VideoCombine] Stream source: {stream_source}")
                
                # If stream source is a file path, use it directly
                if isinstance(stream_source, str) and os.path.exists(stream_source):
                    video_path = stream_source
                    print(f"[VideoCombine] Using direct path: {video_path}")
                else:
                    # Save video to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                        print(f"[VideoCombine] Saving video to temporary file: {temp_path}")
                        video_obj.save_to(temp_path)
                        video_path = temp_path
            else:
                # Fallback: save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                    print(f"[VideoCombine] Saving video to temporary file: {temp_path}")
                    video_obj.save_to(temp_path)
                    video_path = temp_path
            
            print(f"[VideoCombine] Loading video from: {video_path}")
            
            # Load video using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"[VideoCombine] Cannot open video file: {video_path}")
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[VideoCombine] Total frames in video: {total_frames}")
            
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
                    print(f"[VideoCombine] Loaded {frame_count}/{total_frames} frames")
            
            cap.release()
            
            # Clean up temporary file if we created one
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"[VideoCombine] Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"[VideoCombine] Warning: Could not clean up temporary file {temp_path}: {e}")
            
            print(f"[VideoCombine] Successfully loaded {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            print(f"[VideoCombine] Error processing video: {e}")
            
            # Try alternative approach: use get_components if available
            if hasattr(video_obj, 'get_components'):
                try:
                    print(f"[VideoCombine] Trying alternative approach with get_components")
                    components = video_obj.get_components()
                    print(f"[VideoCombine] Components type: {type(components)}")
                    
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
                                print(f"[VideoCombine] Processed component {i + 1}/{len(components)}")
                        
                        print(f"[VideoCombine] Successfully loaded {len(frames)} frames from components")
                        return frames
                    
                except Exception as comp_e:
                    print(f"[VideoCombine] Components approach failed: {comp_e}")
            
            raise Exception(f"[VideoCombine] Failed to load video frames: {e}")


    def _to_pil_images(self, img_input):
        """Convert input to list of PIL images, supporting both single images and image sequences"""
        # Handle ComfyUI VideoFromFile objects
        if hasattr(img_input, '__class__') and 'VideoFromFile' in str(img_input.__class__):
            print(f"[VideoCombine] Detected VideoFromFile object")
            print(f"[VideoCombine] Object type: {type(img_input)}")
            print(f"[VideoCombine] Object attributes: {[attr for attr in dir(img_input) if not attr.startswith('_')]}")
            
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
        
        raise Exception(f"[VideoCombine] Unsupported image format: shape {original_shape}")


    def process(self, video_1=None, video_2=None, image_1=None, image_2=None, **kwargs):
        
        print(f"[VideoCombine] Starting video/image sequence concatenation")
        
        # Get input sequences in order of priority
        all_sequences = []
        
        # Process first sequence (video_1 or image_1)
        if video_1 is not None:
            sequence_1 = self._to_pil_images(video_1)
            print(f"[VideoCombine] Loaded {len(sequence_1)} frames from video_1")
            all_sequences.append(sequence_1)
        elif image_1 is not None:
            sequence_1 = self._to_pil_images(image_1)
            print(f"[VideoCombine] Loaded {len(sequence_1)} frames from image_1")
            all_sequences.append(sequence_1)
        
        # Process second sequence (video_2 or image_2)
        if video_2 is not None:
            sequence_2 = self._to_pil_images(video_2)
            print(f"[VideoCombine] Loaded {len(sequence_2)} frames from video_2")
            all_sequences.append(sequence_2)
        elif image_2 is not None:
            sequence_2 = self._to_pil_images(image_2)
            print(f"[VideoCombine] Loaded {len(sequence_2)} frames from image_2")
            all_sequences.append(sequence_2)
        
        # Check if we have at least one input
        if not all_sequences:
            raise Exception("[VideoCombine] At least one input (video or image) is required")
        
        # Simple concatenation: first sequence + second sequence + ...
        combined_frames = []
        for sequence in all_sequences:
            combined_frames.extend(sequence)
        
        total_frames = len(combined_frames)
        print(f"[VideoCombine] Concatenated {len(all_sequences)} sequences: {total_frames} total frames")
        
        # Convert back to ComfyUI format
        if not combined_frames:
            raise Exception("[VideoCombine] No frames to output")
        
        # Convert PIL images to tensor format
        result_tensors = []
        for i, frame in enumerate(combined_frames):
            # Convert to RGB if needed
            if frame.mode == 'RGBA':
                frame = frame.convert('RGB')
            elif frame.mode != 'RGB':
                frame = frame.convert('RGB')
            
            # Convert to numpy array and normalize
            arr = np.array(frame).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr)
            result_tensors.append(tensor)
            
            # Progress reporting
            if i % max(1, total_frames // 10) == 0:
                print(f"[VideoCombine] Processed frame {i + 1}/{total_frames}")
        
        # Stack tensors
        result = torch.stack(result_tensors, dim=0)
        
        print(f"[VideoCombine] Output tensor shape: {result.shape}")
        return (result,)
