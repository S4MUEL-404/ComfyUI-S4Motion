__version__ = "1.0.0"

# Import dependency manager first
from .dependency_manager import check_startup_dependencies, S4MotionLogger

# Check dependencies at startup
all_deps_ok, dep_status = check_startup_dependencies()

if not all_deps_ok:
    S4MotionLogger.error("Startup", "Core dependencies missing - plugin will not function")
    S4MotionLogger.error("Startup", "Please install core dependencies and restart ComfyUI")
else:
    S4MotionLogger.success("Startup", "CORE READY - Basic S4Motion functionality available")
    
    # Check optional dependencies
    optional_count = sum(1 for k in ['cv2', 'skimage', 'scipy', 'bezier'] if dep_status.get(k, False))
    if optional_count == 4:
        S4MotionLogger.success("Startup", "FULL FEATURES - All optional enhancements available")
    elif optional_count > 0:
        S4MotionLogger.info("Startup", f"ENHANCED - {optional_count}/4 optional enhancements available")
    else:
        S4MotionLogger.warning("Startup", "BASIC MODE - No optional enhancements available")

from .py.motionConfig import MotionConfigNode
from .py.motionPosition import MotionPositionNode
from .py.motionRotation import MotionRotationNode
from .py.motionScale import MotionScaleNode
from .py.motionOpacity import MotionOpacityNode
from .py.motionPositionOnPath import MotionPositionOnPathNode
from .py.motionDistortion import MotionDistortionNode
from .py.motionShake import MotionShakeNode
from .py.motionMask import MotionMaskNode
from .py.videoCrop import VideoCropNode
from .py.videoFrames import VideoFramesNode
import os

NODE_CLASS_MAPPINGS = {
    "💀Motion Config": MotionConfigNode,
    "💀Motion Position": MotionPositionNode,
    "💀Motion Rotation": MotionRotationNode,
    "💀Motion Scale": MotionScaleNode,
    "💀Motion Opacity": MotionOpacityNode,
    "💀Motion Position On Path": MotionPositionOnPathNode,
    "💀Motion Distortion": MotionDistortionNode,
    "💀Motion Shake": MotionShakeNode,
    "💀Motion Mask": MotionMaskNode,
    "💀Video Crop": VideoCropNode,
    "💀Video Frames": VideoFramesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "💀Motion Config": "💀Motion Config",
    "💀Motion Position": "💀Motion Position",
    "💀Motion Rotation": "💀Motion Rotation",
    "💀Motion Scale": "💀Motion Scale",
    "💀Motion Opacity": "💀Motion Opacity",
    "💀Motion Position On Path": "💀Motion Position On Path",
    "💀Motion Distortion": "💀Motion Distortion",
    "💀Motion Shake": "💀Motion Shake",
    "💀Motion Mask": "💀Motion Mask",
    "💀Video Crop": "💀Video Crop",
    "💀Video Frames": "💀Video Frames",
}
