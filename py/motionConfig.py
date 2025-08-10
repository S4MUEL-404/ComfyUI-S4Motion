import os
import numpy as np
from PIL import Image, ImageSequence

class MotionConfigNode:
    """
    Node for configuring and generating animated images (apng/webp) with transformation effects.
    """
    CATEGORY = "💀S4Motion"
    FUNCTION = "process"
    DESCRIPTION = "Configure animation with layer, background, and effectors. Output apng/webp and frame sequence."
    _DEPENDENCY_STATUS_LOGGED = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE", {"default": None}),
                "background_image": ("IMAGE", {"default": None}),
                "time": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1}),
                "loop": ("BOOLEAN", {"default": False}),
                "format": (["webp", "apng"], {"default": "webp"}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "position_x": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "position_y": ("FLOAT", {"default": 0.0, "min": -4096, "max": 4096, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
                                "optional": {
                       "rotation_effector": ("S4_ROTATION_EFFECTOR", {"default": None}),
                       "position_effector": ("S4_POSITION_EFFECTOR", {"default": None}),
                       "scale_effector": ("S4_SCALE_EFFECTOR", {"default": None}),
                       "opacity_effector": ("S4_OPACITY_EFFECTOR", {"default": None}),
                       "distortion_effector": ("S4_DISTORTION_EFFECTOR", {"default": None}),
                       "mask_effector": ("S4_MASK_EFFECTOR", {"default": None}),
                       "shake_effector": ("S4_SHAKE_EFFECTOR", {"default": None}),
                   },
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("Output path", "Frame sequence",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    def process(self, layer_image, background_image, time=2.0, fps=15, loop=False, format="webp", 
                rotation=0.0, position_x=0.0, position_y=0.0, scale=1.0, opacity=1.0,
                rotation_effector=None, position_effector=None, scale_effector=None, opacity_effector=None,
                distortion_effector=None, mask_effector=None, shake_effector=None):
        # Log dependency status once per session
        if not MotionConfigNode._DEPENDENCY_STATUS_LOGGED:
            try:
                self._log_dependency_status()
            except Exception as e:
                print(f"[S4Motion] Dependency status check failed: {e}")
            MotionConfigNode._DEPENDENCY_STATUS_LOGGED = True
        # Log transformation setup
        print(f"[Motion Config] Base transformation parameters:")
        print(f"[Motion Config] Rotation: {rotation}°, Position: ({position_x}, {position_y}), Scale: {scale}, Opacity: {opacity}")
        
        if rotation_effector or position_effector or scale_effector or opacity_effector:
            print(f"[Motion Config] Using effector(s) for animation (base + effector)")
        else:
            print(f"[Motion Config] Using basic parameters only")
        
        output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        total_frames = int(time * fps)
        
        print(f"[Motion Config] Starting animation generation...")
        print(f"[Motion Config] Total frames: {total_frames}, FPS: {fps}, Duration: {time}s")
        
        # Convert inputs to PIL images (support both single images and image sequences)
        print(f"[Motion Config] Converting input images...")
        layer_images = self._to_pil_images(layer_image)
        background_images = self._to_pil_images(background_image)
        print(f"[Motion Config] Layer images: {len(layer_images)} frame(s)")
        print(f"[Motion Config] Background images: {len(background_images)} frame(s)")
        
        # Get effector data if available
        rotations = None
        positions = None
        scales = None
        opacities = None
        distortions = None
        shakes = None
        mask_specs = None
        mask_source = None
        auto_rotations = None
        is_path_mode = False

        if callable(rotation_effector):
            # Validate effector type
            if hasattr(rotation_effector, 'effector_type') and rotation_effector.effector_type != "rotation":
                raise Exception(f"Rotation effector type mismatch: expected 'rotation', got '{rotation_effector.effector_type}'. Please connect Motion Rotation node to rotation_effector.")
            elif not hasattr(rotation_effector, 'effector_type'):
                raise Exception("Invalid rotation effector: Please connect Motion Rotation node to rotation_effector.")
            print(f"[Motion Config] Getting rotation data from effector...")
            rotations = rotation_effector(total_frames, time)
        
        if callable(position_effector):
            # Validate effector type
            if hasattr(position_effector, 'effector_type') and position_effector.effector_type != "position":
                raise Exception(f"Position effector type mismatch: expected 'position', got '{position_effector.effector_type}'. Please connect Motion Position or Motion Position on Path node to position_effector.")
            elif not hasattr(position_effector, 'effector_type'):
                raise Exception("Invalid position effector: Please connect Motion Position or Motion Position on Path node to position_effector.")
            print(f"[Motion Config] Getting position data from effector...")
            result = position_effector(total_frames, time)
            if isinstance(result, tuple) and len(result) == 2:
                positions, auto_rotations = result
                print(f"[Motion Config] Auto-orientation enabled, got {len(auto_rotations)} rotation values")
            else:
                positions = result
                auto_rotations = None
            is_path_mode = getattr(position_effector, 'is_path_mode', False)
            print(f"[Motion Config] Position mode: {'Path' if is_path_mode else 'Parameter'}")
        
        if callable(scale_effector):
            # Validate effector type
            if hasattr(scale_effector, 'effector_type') and scale_effector.effector_type != "scale":
                raise Exception(f"Scale effector type mismatch: expected 'scale', got '{scale_effector.effector_type}'. Please connect Motion Scale node to scale_effector.")
            elif not hasattr(scale_effector, 'effector_type'):
                raise Exception("Invalid scale effector: Please connect Motion Scale node to scale_effector.")
            print(f"[Motion Config] Getting scale data from effector...")
            scales = scale_effector(total_frames, time)
        
        if callable(opacity_effector):
            # Validate effector type
            if hasattr(opacity_effector, 'effector_type') and opacity_effector.effector_type != "opacity":
                raise Exception(f"Opacity effector type mismatch: expected 'opacity', got '{opacity_effector.effector_type}'. Please connect Motion Opacity node to opacity_effector.")
            elif not hasattr(opacity_effector, 'effector_type'):
                raise Exception("Invalid opacity effector: Please connect Motion Opacity node to opacity_effector.")
            print(f"[Motion Config] Getting opacity data from effector...")
            opacities = opacity_effector(total_frames, time)
        
        if callable(distortion_effector):
            # Validate effector type
            if hasattr(distortion_effector, 'effector_type') and distortion_effector.effector_type != "distortion":
                raise Exception(f"Distortion effector type mismatch: expected 'distortion', got '{distortion_effector.effector_type}'. Please connect Motion Distortion node to distortion_effector.")
            elif not hasattr(distortion_effector, 'effector_type'):
                raise Exception("Invalid distortion effector: Please connect Motion Distortion node to distortion_effector.")
            print(f"[Motion Config] Getting distortion data from effector...")
            distortions = distortion_effector(total_frames, time)
            if isinstance(distortions, list):
                print(f"[Motion Config] Distortion frames: {len(distortions)}")
            else:
                print(f"[Motion Config] Warning: distortion effector did not return a list, it will be ignored.")
                distortions = None

        if callable(shake_effector):
            if hasattr(shake_effector, 'effector_type') and shake_effector.effector_type != "shake":
                raise Exception(f"Shake effector type mismatch: expected 'shake', got '{shake_effector.effector_type}'. Please connect Motion Shake node to shake_effector.")
            elif not hasattr(shake_effector, 'effector_type'):
                raise Exception("Invalid shake effector: Please connect Motion Shake node to shake_effector.")
            print(f"[Motion Config] Getting shake data from effector...")
            shakes = shake_effector(total_frames, time)
            if isinstance(shakes, list):
                print(f"[Motion Config] Shake frames: {len(shakes)}")
            else:
                print(f"[Motion Config] Warning: shake effector did not return a list, it will be ignored.")
                shakes = None

        if callable(mask_effector):
            # Validate effector type
            if hasattr(mask_effector, 'effector_type') and mask_effector.effector_type != "mask":
                raise Exception(f"Mask effector type mismatch: expected 'mask', got '{mask_effector.effector_type}'. Please connect Motion Mask node to mask_effector.")
            elif not hasattr(mask_effector, 'effector_type'):
                raise Exception("Invalid mask effector: Please connect Motion Mask node to mask_effector.")
            print(f"[Motion Config] Getting mask data from effector...")
            mask_specs = mask_effector(total_frames, time)
            mask_source = getattr(mask_effector, 'mask_image', None)
            if isinstance(mask_specs, list):
                print(f"[Motion Config] Mask frames: {len(mask_specs)}")
            else:
                print(f"[Motion Config] Warning: mask effector did not return a list, it will be ignored.")
                mask_specs = None

        
        # Generate frames
        print(f"[Motion Config] Generating {total_frames} frames...")
        frames = []
        for i in range(total_frames):
            # Show progress every 10% or every 10 frames, whichever is smaller
            progress_interval = max(1, min(10, total_frames // 10))
            if i % progress_interval == 0 or i == total_frames - 1:
                progress = (i + 1) / total_frames * 100
                print(f"[Motion Config] Progress: {progress:.1f}% ({i + 1}/{total_frames})")
            
            t = i / (total_frames - 1) if total_frames > 1 else 0
            
            # Get current layer and background images (cycle if sequences)
            layer_img = layer_images[i % len(layer_images)]
            
            # Apply distortion before rotation/scale if provided
            if distortions is not None and i < len(distortions) and isinstance(distortions[i], dict):
                try:
                    layer_img = self.apply_distortion(layer_img, distortions[i])
                except Exception as e:
                    print(f"[Motion Config] Distortion failed on frame {i}: {e}")
            # Apply shake after distortion
            if shakes is not None and i < len(shakes) and isinstance(shakes[i], dict):
                try:
                    layer_img = self.apply_shake(layer_img, shakes[i])
                except Exception as e:
                    print(f"[Motion Config] Shake failed on frame {i}: {e}")
            bg_img = background_images[i % len(background_images)]
            
            # Get current transformation values (base + effector + auto-orientation)
            current_rotation = rotation
            if rotations is not None:
                current_rotation += rotations[i]
            if auto_rotations is not None:
                current_rotation += auto_rotations[i]
            
            current_scale = scale
            if scales is not None:
                current_scale *= scales[i]
            
            current_opacity = opacity
            if opacities is not None:
                current_opacity *= opacities[i]
            
            # Get current position (base + effector)
            current_x = position_x
            current_y = position_y
            if positions is not None:
                current_x += positions[i][0]
                current_y += positions[i][1]
                # Only center align for path mode, otherwise always top-left align
                if is_path_mode:
                    frame = self.compose_frame(
                        layer_img, bg_img,
                        current_rotation, current_scale, current_x, current_y, current_opacity,
                        center_align=True,
                        mask_spec=(mask_specs[i] if (mask_specs is not None and i < len(mask_specs)) else None),
                        mask_source=mask_source,
                    )
                else:
                    frame = self.compose_frame(
                        layer_img, bg_img,
                        current_rotation, current_scale, current_x, current_y, current_opacity,
                        center_align=False,
                        mask_spec=(mask_specs[i] if (mask_specs is not None and i < len(mask_specs)) else None),
                        mask_source=mask_source,
                    )
            else:
                # Use basic position parameters only
                if i == 0:  # Only print once for debugging
                    print(f"[Motion Config] Using basic position: ({current_x}, {current_y})")
                frame = self.compose_frame(
                    layer_img, bg_img,
                    current_rotation, current_scale, current_x, current_y, current_opacity,
                    center_align=False,
                    mask_spec=(mask_specs[i] if (mask_specs is not None and i < len(mask_specs)) else None),
                    mask_source=mask_source,
                )
            frames.append(frame)
        
        # Save animation file
        print(f"[Motion Config] Saving animation file...")
        filename = f"motion_{int(np.random.rand()*1e8)}.{format}"
        output_path = os.path.abspath(os.path.join(output_dir, filename))
        self.save_animation(frames, output_path, fps, loop, format)
        print(f"[Motion Config] Animation saved to: {output_path}")
        
        # Convert frames to ComfyUI format for output
        print(f"[Motion Config] Converting frames to output format...")
        frame_sequence = self._frames_to_comfyui_format(frames)
        print(f"[Motion Config] Animation generation completed successfully!")
        
        return (output_path, frame_sequence)

    def _to_pil_images(self, img_input):
        """Convert input to list of PIL images, supporting both single images and image sequences"""
        if isinstance(img_input, list):
            # Input is already a sequence
            return [self._to_pil_image(img) for img in img_input]
        elif isinstance(img_input, np.ndarray) and img_input.ndim == 4:
            # Multi-frame numpy array: (frames, height, width, channels)
            return [self._to_pil_image(img_input[i]) for i in range(img_input.shape[0])]
        elif 'torch' in str(type(img_input)) and hasattr(img_input, 'shape') and len(img_input.shape) == 4:
            # Multi-frame torch tensor: (frames, height, width, channels)
            import torch
            if isinstance(img_input, torch.Tensor):
                arr = img_input.detach().cpu().numpy()
                return [self._to_pil_image(arr[i]) for i in range(arr.shape[0])]
        else:
            # Input is single image
            return [self._to_pil_image(img_input)]

    def _to_pil_image(self, img):
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

    def _frames_to_comfyui_format(self, frames):
        """Convert PIL frames to ComfyUI format (torch tensor with batch dimension)"""
        import torch
        
        frame_arrays = []
        for frame in frames:
            # Convert PIL to numpy array
            arr = np.array(frame)
            # Add batch dimension if needed
            if arr.ndim == 3:
                arr = arr[None, ...]
            # Convert to float32 [0,1] range
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            frame_arrays.append(arr)
        
        # Stack all frames and convert to torch tensor
        if frame_arrays:
            stacked = np.concatenate(frame_arrays, axis=0)
            return torch.from_numpy(stacked)
        else:
            return torch.tensor([])

    def compose_frame(self, layer_img, bg_img, rotation, scale, x, y, opacity, center_align=False, mask_spec=None, mask_source=None):
        frame = bg_img.copy().convert("RGBA")
        layer = layer_img.copy().convert("RGBA")
        
        # Store original layer center
        original_center_x = layer.width / 2
        original_center_y = layer.height / 2
        
        # Apply rotation first with proper center handling and anti-aliasing
        if rotation != 0:
            # For better quality, use supersampling for small rotations
            use_supersampling = abs(rotation % 90) > 1  # Skip supersampling for 90-degree multiples
            
            if use_supersampling and min(layer.width, layer.height) < 500:
                # Supersample for better quality on smaller images
                supersample_factor = 2
                temp_layer = layer.resize(
                    (layer.width * supersample_factor, layer.height * supersample_factor), 
                    Image.BICUBIC
                )
                temp_rotated = temp_layer.rotate(
                    rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)
                )
                # Downsample back to original scale
                final_width = temp_rotated.width // supersample_factor
                final_height = temp_rotated.height // supersample_factor
                layer_rotated = temp_rotated.resize((final_width, final_height), Image.BICUBIC)
            else:
                # Use high-quality rotation with anti-aliasing
                # BICUBIC provides good anti-aliasing for rotation
                layer_rotated = layer.rotate(rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
            
            # Calculate the offset caused by expansion
            offset_x = (layer_rotated.width - layer.width) / 2
            offset_y = (layer_rotated.height - layer.height) / 2
            layer = layer_rotated
        else:
            offset_x = 0
            offset_y = 0
        
        # Apply scale after rotation with high-quality resampling
        if scale != 1.0:
            # Store rotated dimensions before scaling
            pre_scale_width = layer.width
            pre_scale_height = layer.height
            
            new_width = max(1, int(layer.width * scale))  # Ensure minimum 1 pixel
            new_height = max(1, int(layer.height * scale))  # Ensure minimum 1 pixel
            
            # Choose resampling method based on scaling direction
            # LANCZOS is superior for downscaling; BICUBIC is fine for upscaling
            resample_method = Image.LANCZOS if scale < 1.0 else Image.BICUBIC
            layer = layer.resize((new_width, new_height), resample_method)
            
            # Update offset for scaling
            scale_offset_x = (layer.width - pre_scale_width) / 2
            scale_offset_y = (layer.height - pre_scale_height) / 2
            offset_x += scale_offset_x
            offset_y += scale_offset_y
        
        # Apply opacity to the layer
        if opacity != 1.0:
            # Create a copy to avoid modifying the original
            layer = layer.copy()
            # Apply opacity by modifying the alpha channel
            if layer.mode == 'RGBA':
                r, g, b, a = layer.split()
                # Apply opacity to alpha channel
                a = a.point(lambda p: int(p * opacity))
                layer = Image.merge('RGBA', (r, g, b, a))
            else:
                # Convert to RGBA and apply opacity
                layer = layer.convert('RGBA')
                r, g, b, a = layer.split()
                a = a.point(lambda p: int(p * opacity))
                layer = Image.merge('RGBA', (r, g, b, a))
        
        # Calculate final position considering center alignment and offsets
        if center_align:
            # For center align mode, the x,y position represents the center of the final image
            final_x = x - layer.width / 2
            final_y = y - layer.height / 2
        else:
            # For normal positioning, compensate for the rotation/scale offsets to maintain visual center
            final_x = x - offset_x
            final_y = y - offset_y
        
        # Apply optional mask before compositing
        if isinstance(mask_spec, dict):
            try:
                layer = self.apply_mask(layer, mask_spec, mask_source)
            except Exception as e:
                print(f"[Motion Config] Mask apply failed on frame: {e}")

        # Allow negative coordinates (no boundary restriction)
        # The layer can be positioned outside the background image bounds
        frame.alpha_composite(layer, dest=(int(final_x), int(final_y)))
        return frame

    

    def _log_dependency_status(self):
        """Print optional/runtime dependency status once to ComfyUI log."""
        def check(pkg_name, module_name):
            try:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "unknown")
                print(f"[S4Motion] Dependency OK: {pkg_name} ({module_name}) v{version}")
            except Exception as e:
                print(f"[S4Motion] Dependency MISSING: {pkg_name} ({module_name}). Install via: pip install {pkg_name}")
        print("[S4Motion] Checking optional/runtime dependencies...")
        check("opencv-python", "cv2")
        check("scikit-image", "skimage")
        check("scipy", "scipy")
        # bezier is optional for smoothing motion curves
        try:
            import bezier  # noqa: F401
            version = getattr(bezier, "__version__", "unknown")
            print(f"[S4Motion] Optional OK: bezier v{version}")
        except Exception:
            print("[S4Motion] Optional MISSING: bezier. Install via: pip install bezier (falls back to mathematical easing)")

    def apply_distortion(self, layer_img, spec):
        """Apply image distortion based on provided spec.

        spec example:
        {
            "mode": "wave" | "vortex" | "radial",
            "params": { ... }
        }
        """
        from PIL import Image
        import numpy as np
        mode = spec.get("mode", "wave")
        params = dict(spec.get("params", {}))

        img = layer_img.convert("RGBA")
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]

        if mode == "wave":
            import cv2
            amplitude = float(params.get("amplitude_px", 10.0))
            wavelength = float(params.get("wavelength_px", 100.0))
            phase_deg = float(params.get("phase_deg", 0.0))
            direction = params.get("direction", "x")  # x | y | xy
            phase = np.deg2rad(phase_deg)

            yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
            map_x = xx.copy()
            map_y = yy.copy()

            if direction in ("x", "xy"):
                map_x = map_x + amplitude * np.sin(2 * np.pi * yy / max(1.0, wavelength) + phase)
            if direction in ("y", "xy"):
                map_y = map_y + amplitude * np.sin(2 * np.pi * xx / max(1.0, wavelength) + phase)

            # Remap for all channels
            remapped = cv2.remap(arr, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            return Image.fromarray(remapped)

        elif mode == "vortex":
            import cv2
            strength_deg = float(params.get("strength_deg", 20.0))
            radius_px = max(1.0, float(params.get("radius_px", min(w, h) / 2.0)))
            cx = float(params.get("center_x", 0.5)) * w
            cy = float(params.get("center_y", 0.5)) * h
            phase_deg = float(params.get("phase_deg", 0.0))
            phase = np.deg2rad(phase_deg)

            yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
            dx = xx - cx
            dy = yy - cy
            r = np.sqrt(dx * dx + dy * dy)

            base_angle = np.deg2rad(strength_deg)
            falloff = np.clip(1.0 - (r / radius_px), 0.0, 1.0)
            angle_offset = base_angle * falloff + phase

            # Use inverse rotation to compute source coordinates for remap (dest -> src)
            sin_a = np.sin(angle_offset)
            cos_a = np.cos(angle_offset)
            x_src = cx + (dx * cos_a + dy * sin_a)
            y_src = cy + (-dx * sin_a + dy * cos_a)

            mask = (r <= radius_px).astype(np.float32)
            map_x = x_src * mask + xx * (1.0 - mask)
            map_y = y_src * mask + yy * (1.0 - mask)

            remapped = cv2.remap(arr, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            return Image.fromarray(remapped)

        elif mode == "radial":
            import cv2
            # Barrel/pincushion via simple radial model: x' = x + k * r^2 * (x - cx)
            k = float(params.get("k", 0.0))
            center_x = float(params.get("center_x", 0.5)) * w
            center_y = float(params.get("center_y", 0.5)) * h

            yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
            dx = xx - center_x
            dy = yy - center_y
            r2 = dx * dx + dy * dy
            max_r2 = (max(w, h) * 0.5) ** 2
            # Normalize r^2 to make k stable across sizes
            r2n = r2 / max(1.0, max_r2)
            factor = 1.0 + k * r2n
            map_x = center_x + dx * factor
            map_y = center_y + dy * factor
            remapped = cv2.remap(arr, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            return Image.fromarray(remapped)

        else:
            # Unknown mode, return original
            return img

    def apply_shake(self, layer_img, spec):
        """Apply shake-like effect: offset and optional chromatic edge ghosts (signal mode)."""
        from PIL import Image
        import numpy as np

        img = layer_img.convert("RGBA")
        params = dict(spec.get("params", {}))
        mode = spec.get("mode", "horizontal")

        offset_x = float(params.get("offset_x", 0.0))
        offset_y = float(params.get("offset_y", 0.0))

        # Base translation (keeps original color, no tint)
        translated = Image.new("RGBA", img.size, (0, 0, 0, 0))
        translated.alpha_composite(img, dest=(int(offset_x), int(offset_y)))

        # Signal mode now returns pure jitter (no colored edges)
        return translated

    def apply_mask(self, layer_img, spec, mask_source):
        """Apply mask with transform/opacity to layer image.

        spec example:
        {
            "x": float,
            "y": float,
            "rotation": float,
            "scale": float,
            "opacity": float
        }
        mask_source: PIL image provided by mask effector as the base mask
        """
        from PIL import Image, ImageChops
        import numpy as np

        if mask_source is None:
            return layer_img

        # Prepare mask image
        mask_img = mask_source.convert("L")
        # Track offsets so that rotation/scale keep the visual center fixed
        original_w, original_h = mask_img.width, mask_img.height
        offset_x_correction = 0.0
        offset_y_correction = 0.0

        m_rotation = float(spec.get("rotation", 0.0))
        m_scale = float(spec.get("scale", 1.0))
        m_x = float(spec.get("x", 0.0))
        m_y = float(spec.get("y", 0.0))
        m_opacity = float(spec.get("opacity", 1.0))

        # Transform mask: rotate then scale similar to layer processing
        if m_rotation != 0:
            mask_img = mask_img.rotate(m_rotation, expand=True, resample=Image.BICUBIC, fillcolor=0)
            # Compensate expansion so that the mask appears to rotate around its own center
            rotated_w, rotated_h = mask_img.width, mask_img.height
            offset_x_correction += (rotated_w - original_w) / 2.0
            offset_y_correction += (rotated_h - original_h) / 2.0
            original_w, original_h = rotated_w, rotated_h
        if m_scale != 1.0:
            pre_scale_w, pre_scale_h = mask_img.width, mask_img.height
            new_w = max(1, int(pre_scale_w * m_scale))
            new_h = max(1, int(pre_scale_h * m_scale))
            resample_method = Image.LANCZOS if m_scale < 1.0 else Image.BICUBIC
            mask_img = mask_img.resize((new_w, new_h), resample_method)
            # Compensate scaling so that the mask scales around its center
            offset_x_correction += (new_w - pre_scale_w) / 2.0
            offset_y_correction += (new_h - pre_scale_h) / 2.0

        # Apply mask opacity
        if m_opacity != 1.0:
            mask_img = mask_img.point(lambda p: int(p * np.clip(m_opacity, 0.0, 1.0)))

        # Create a full-size blank mask canvas and paste transformed mask at (x,y)
        canvas = Image.new("L", layer_img.size, 0)
        paste_x = int(m_x - offset_x_correction)
        paste_y = int(m_y - offset_y_correction)
        canvas.paste(mask_img, (paste_x, paste_y))

        # If layer has alpha, multiply with mask
        if layer_img.mode != 'RGBA':
            layer_img = layer_img.convert('RGBA')
        r, g, b, a = layer_img.split()
        new_alpha = ImageChops.multiply(a, canvas)
        return Image.merge('RGBA', (r, g, b, new_alpha))

    def save_animation(self, frames, path, fps, loop, format):
        duration = int(1000 / fps)
        if format == "webp":
            # Explicitly set loop semantics to avoid library defaulting to infinite
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0 if loop else 1,
                format="WEBP",
                lossless=True,
                transparency=0,
            )
        elif format == "apng":
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0 if loop else 1,
                format="PNG",
            )
        else:
            raise Exception("Unsupported format: " + format)
