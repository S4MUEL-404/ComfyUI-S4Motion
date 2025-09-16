"""
S4Motion Dependency Manager and Unified Logging System
Production Quality Above All - No Feature Fallbacks
"""
import sys
import importlib
from typing import Dict, List, Tuple, Optional
import warnings

# ANSI color codes for terminal output
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

class S4MotionLogger:
    """Unified logging system for S4Motion nodes"""
    
    PREFIX = "S4Motion"
    
    @staticmethod
    def _format_message(level: str, node: str, message: str) -> str:
        """Format log message with consistent style"""
        if level == "ERROR":
            icon = "❌"
            color = Colors.RED
        elif level == "WARNING": 
            icon = "⚠️"
            color = Colors.YELLOW
        elif level == "SUCCESS":
            icon = "✅"
            color = Colors.GREEN
        elif level == "INFO":
            icon = "ℹ️"
            color = Colors.CYAN
        elif level == "DEBUG":
            icon = "🔧"
            color = Colors.DIM
        else:
            icon = "📝"
            color = Colors.WHITE
        
        return f"{color}{Colors.BOLD}[{S4MotionLogger.PREFIX}]{Colors.RESET} {icon} {color}{node}:{Colors.RESET} {message}"
    
    @staticmethod
    def error(node: str, message: str):
        """Log error message"""
        print(S4MotionLogger._format_message("ERROR", node, message))
    
    @staticmethod
    def warning(node: str, message: str):
        """Log warning message"""  
        print(S4MotionLogger._format_message("WARNING", node, message))
    
    @staticmethod
    def success(node: str, message: str):
        """Log success message"""
        print(S4MotionLogger._format_message("SUCCESS", node, message))
    
    @staticmethod
    def info(node: str, message: str):
        """Log info message"""
        print(S4MotionLogger._format_message("INFO", node, message))
    
    @staticmethod
    def debug(node: str, message: str):
        """Log debug message"""
        print(S4MotionLogger._format_message("DEBUG", node, message))

class DependencyManager:
    """Manages dependency checking for S4Motion nodes - Production Quality"""
    
    # Core dependencies are REQUIRED for basic functionality
    CORE_DEPENDENCIES = {
        'torch': {
            'package': 'torch',
            'description': 'PyTorch - Core tensor operations and ComfyUI compatibility',
            'install_cmd': 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu',
            'features': ['ComfyUI Integration', 'Tensor Operations', 'Animation Processing']
        },
        'PIL': {
            'package': 'Pillow',
            'import_name': 'PIL',
            'description': 'Pillow - Professional image processing',
            'install_cmd': 'pip install Pillow>=8.0.0',
            'features': ['Image Processing', 'Animation Frames', 'Alpha Compositing']
        },
        'numpy': {
            'package': 'numpy',
            'description': 'NumPy - High-performance numerical computing',
            'install_cmd': 'pip install numpy>=1.21.0',
            'features': ['Mathematical Operations', 'Motion Curves', 'Performance']
        }
    }
    
    # Optional dependencies for enhanced functionality
    OPTIONAL_DEPENDENCIES = {
        'cv2': {
            'package': 'opencv-python',
            'import_name': 'cv2',
            'description': 'OpenCV - Video processing and advanced distortion effects',
            'install_cmd': 'pip install opencv-python>=4.5.0',
            'features': ['Video File Support', 'Advanced Distortion', 'Video Processing'],
            'fallback': 'Video processing will be limited without OpenCV'
        },
        'skimage': {
            'package': 'scikit-image',
            'import_name': 'skimage',
            'description': 'Scikit-image - Advanced image transformations and distortion effects',
            'install_cmd': 'pip install scikit-image>=0.19.0',
            'features': ['High-Quality Distortion', 'Professional Transforms', 'Advanced Effects'],
            'fallback': 'Some distortion effects will use basic implementations'
        },
        'scipy': {
            'package': 'scipy',
            'description': 'SciPy - Scientific computing for advanced motion curves',
            'install_cmd': 'pip install scipy>=1.8.0',
            'features': ['Advanced Interpolation', 'Scientific Motion', 'Smooth Curves'],
            'fallback': 'Motion curves will use basic mathematical implementations'
        },
        'bezier': {
            'package': 'bezier',
            'description': 'Bezier - Professional motion curve calculations',
            'install_cmd': 'pip install bezier>=0.12.0',
            'features': ['Professional Bezier Curves', 'Smooth Motion', 'Advanced Easing'],
            'fallback': 'Motion curves will use mathematical approximations'
        }
    }
    
    def __init__(self):
        self.core_status: Dict[str, bool] = {}
        self.optional_status: Dict[str, bool] = {}
        self.missing_core: List[str] = []
        self.missing_optional: List[str] = []
        
    def check_import(self, import_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a module can be imported and get version"""
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except ImportError:
            return False, None
    
    def check_all_dependencies(self) -> Tuple[bool, Dict[str, bool]]:
        """Check all dependencies - Core required, Optional enhances functionality"""
        
        all_status = {}
        
        # Check core dependencies (REQUIRED)
        for dep_key, dep_info in self.CORE_DEPENDENCIES.items():
            import_name = dep_info.get('import_name', dep_key)
            is_available, version = self.check_import(import_name)
            self.core_status[dep_key] = is_available
            all_status[dep_key] = is_available
            
            if not is_available:
                self.missing_core.append(dep_key)
        
        # Check optional dependencies (ENHANCES functionality)
        for dep_key, dep_info in self.OPTIONAL_DEPENDENCIES.items():
            import_name = dep_info.get('import_name', dep_key)
            is_available, version = self.check_import(import_name)
            self.optional_status[dep_key] = is_available
            all_status[dep_key] = is_available
            
            if not is_available:
                self.missing_optional.append(dep_key)
        
        # Core dependencies must be available for basic functionality
        core_deps_ok = len(self.missing_core) == 0
        
        return core_deps_ok, all_status
    
    def print_startup_report(self):
        """Print production-quality dependency status report at startup"""
        core_deps_ok, all_status = self.check_all_dependencies()
        
        # Header
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}💀 S4Motion Production Dependency Status Report{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}")
        
        # Production quality statement
        print(f"\n{Colors.BOLD}🎯 Production Quality Statement:{Colors.RESET}")
        print(f"  {Colors.CYAN}• Core dependencies are REQUIRED for basic functionality{Colors.RESET}")
        print(f"  {Colors.CYAN}• Optional dependencies enhance features with graceful fallbacks{Colors.RESET}")
        
        # Core Dependencies section
        print(f"\n{Colors.BOLD}🔧 Core Dependencies (MANDATORY):{Colors.RESET}")
        for dep_key, dep_info in self.CORE_DEPENDENCIES.items():
            import_name = dep_info.get('import_name', dep_key)
            is_available, version = self.check_import(import_name)
            
            if is_available:
                status_icon = "✅"
                status_color = Colors.GREEN
                version_info = f" v{version}" if version != 'unknown' else ""
                print(f"  {status_icon} {status_color}{dep_info['package']:<25}{Colors.RESET}{version_info}")
                features = ", ".join(dep_info.get('features', []))
                print(f"    {Colors.DIM}└─ {features}{Colors.RESET}")
            else:
                status_icon = "❌"
                status_color = Colors.RED  
                print(f"  {status_icon} {status_color}{dep_info['package']:<25}{Colors.RESET} - MISSING")
                print(f"    {Colors.RED}📦 Install: {dep_info['install_cmd']}{Colors.RESET}")
                features = ", ".join(dep_info.get('features', []))
                print(f"    {Colors.RED}⚠️  Missing Features: {features}{Colors.RESET}")
        
        # Optional Dependencies section
        print(f"\n{Colors.BOLD}🌟 Optional Dependencies (ENHANCES functionality):{Colors.RESET}")
        for dep_key, dep_info in self.OPTIONAL_DEPENDENCIES.items():
            import_name = dep_info.get('import_name', dep_key)
            is_available, version = self.check_import(import_name)
            
            if is_available:
                status_icon = "✅"
                status_color = Colors.GREEN
                version_info = f" v{version}" if version != 'unknown' else ""
                print(f"  {status_icon} {status_color}{dep_info['package']:<25}{Colors.RESET}{version_info}")
                features = ", ".join(dep_info.get('features', []))
                print(f"    {Colors.DIM}└─ {features}{Colors.RESET}")
            else:
                status_icon = "⚠️"
                status_color = Colors.YELLOW  
                print(f"  {status_icon} {status_color}{dep_info['package']:<25}{Colors.RESET} - OPTIONAL")
                print(f"    {Colors.YELLOW}📦 Install: {dep_info['install_cmd']}{Colors.RESET}")
                features = ", ".join(dep_info.get('features', []))
                print(f"    {Colors.YELLOW}🌟 Enhanced Features: {features}{Colors.RESET}")
                if 'fallback' in dep_info:
                    print(f"    {Colors.DIM}💡 Fallback: {dep_info['fallback']}{Colors.RESET}")
        
        # Summary
        print(f"\n{Colors.BOLD}📊 Production Quality Summary:{Colors.RESET}")
        total_core = len(self.CORE_DEPENDENCIES)
        available_core = sum(1 for available in self.core_status.values() if available)
        total_optional = len(self.OPTIONAL_DEPENDENCIES)
        available_optional = sum(1 for available in self.optional_status.values() if available)
        
        if core_deps_ok:
            print(f"  ✅ {Colors.GREEN}{Colors.BOLD}CORE READY{Colors.RESET} - All core dependencies satisfied")
            print(f"  🎨 {Colors.GREEN}Basic S4Motion functionality available{Colors.RESET}")
        else:
            print(f"  ❌ {Colors.RED}{Colors.BOLD}NOT READY{Colors.RESET} - Missing core dependencies")
            print(f"  ⚠️  {Colors.RED}Plugin will NOT work until core dependencies are installed{Colors.RESET}")
        
        print(f"  📈 {Colors.CYAN}Core dependencies: {available_core}/{total_core} ({available_core/total_core*100:.1f}%){Colors.RESET}")
        print(f"  🌟 {Colors.YELLOW}Optional enhancements: {available_optional}/{total_optional} ({available_optional/total_optional*100:.1f}%){Colors.RESET}")
        
        # Installation commands
        if self.missing_core or self.missing_optional:
            print(f"\n{Colors.BOLD}🔨 Installation Commands:{Colors.RESET}")
            
            if self.missing_core:
                print(f"  {Colors.RED}❗ REQUIRED (Core):{Colors.RESET}")
                for dep_key in self.missing_core:
                    dep_info = self.CORE_DEPENDENCIES[dep_key]
                    print(f"    {Colors.RED}{dep_info['install_cmd']}{Colors.RESET}")
            
            if self.missing_optional:
                print(f"  {Colors.YELLOW}💡 RECOMMENDED (Optional):{Colors.RESET}")
                for dep_key in self.missing_optional:
                    dep_info = self.OPTIONAL_DEPENDENCIES[dep_key]
                    print(f"    {Colors.YELLOW}{dep_info['install_cmd']}{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}\n")
        
        # Return status for __init__.py
        return core_deps_ok, all_status
    
    def require_dependency(self, dep_key: str, feature_name: str = "", allow_fallback: bool = False):
        """Require a dependency or provide fallback for optional dependencies"""
        # Check if it's a core or optional dependency
        is_core = dep_key in self.CORE_DEPENDENCIES
        is_optional = dep_key in self.OPTIONAL_DEPENDENCIES
        
        if not is_core and not is_optional:
            raise ImportError(f"Unknown dependency: {dep_key}")
        
        # Check availability
        if dep_key not in self.core_status and dep_key not in self.optional_status:
            if is_core:
                dep_info = self.CORE_DEPENDENCIES[dep_key]
            else:
                dep_info = self.OPTIONAL_DEPENDENCIES[dep_key]
            
            import_name = dep_info.get('import_name', dep_key)
            is_available, _ = self.check_import(import_name)
            
            if is_core:
                self.core_status[dep_key] = is_available
            else:
                self.optional_status[dep_key] = is_available
        
        # Get availability status
        is_available = self.core_status.get(dep_key) or self.optional_status.get(dep_key, False)
        
        if not is_available:
            if is_core or not allow_fallback:
                # Core dependency missing or fallback not allowed - raise error
                if is_core:
                    dep_info = self.CORE_DEPENDENCIES[dep_key]
                    severity = "CRITICAL"
                else:
                    dep_info = self.OPTIONAL_DEPENDENCIES[dep_key]
                    severity = "FEATURE"
                
                feature_desc = f" for {feature_name}" if feature_name else ""
                error_msg = f"""
❌ {severity} DEPENDENCY MISSING{feature_desc}

📦 Required Package: {dep_info['package']}
📄 Description: {dep_info['description']}
🔧 Install Command: {dep_info['install_cmd']}

{'🚨 This is a CORE dependency required for basic functionality.' if is_core else '💡 This is an optional dependency that enhances functionality.'}
   
💡 Please install the required package and restart ComfyUI.
"""
                S4MotionLogger.error("DependencyManager", error_msg)
                raise ImportError(f"Missing {'core' if is_core else 'optional'} dependency: {dep_info['package']}")
            else:
                # Optional dependency with fallback allowed
                dep_info = self.OPTIONAL_DEPENDENCIES[dep_key]
                fallback_msg = dep_info.get('fallback', 'Basic implementation will be used')
                S4MotionLogger.warning("DependencyManager", 
                    f"Optional dependency '{dep_info['package']}' not found. {fallback_msg}")
                return False
        
        return True

# Global dependency manager instance
_dependency_manager = None

def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager instance"""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager

def require_dependency(dep_key: str, feature_name: str = "", allow_fallback: bool = False):
    """Convenient function to require a dependency"""
    return get_dependency_manager().require_dependency(dep_key, feature_name, allow_fallback)

def check_startup_dependencies():
    """Check and print dependency status at startup"""
    return get_dependency_manager().print_startup_report()