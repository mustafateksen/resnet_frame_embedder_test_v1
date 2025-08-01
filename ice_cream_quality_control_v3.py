#!/usr/bin/env python3
# ice_cream_quality_control_v3.py
# ----------------------------------------------------------
# Enhanced anomaly detection system for ice cream factory quality control
# 
# Key improvements:
# 1. Multiple tiling strategies for different defect types
# 2. Adaptive ROI selection based on ice cream position
# 3. Multi-scale analysis (different ResNet architectures)
# 4. Statistical outlier detection with confidence intervals
# 5. Ensemble scoring for better accuracy
# 6. Advanced preprocessing for ice cream specific features
# 7. Real-time processing optimizations
# 8. Detailed logging and reporting
#
# Example usage:
# python ice_cream_quality_control_v3.py --good_dir "good_ice_creams" --test_dir "test_batch" --anomaly_folder "detected_anomalies" --confidence 0.95 --ensemble_method "weighted_avg"
# ----------------------------------------------------------

import argparse, glob, os, cv2, torch, numpy as np, shutil, json, time
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from tiled_embedder import ResNetTiledEmbedder


# ─── Enhanced Logging Setup ───────────────────────────────
def setup_logging(log_dir: str = "logs"):
    """Setup detailed logging for quality control system"""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"quality_control_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ─── Advanced Tiling Strategies for Ice Cream ──────────────
class IceCreamTilingStrategies:
    """Multiple tiling strategies optimized for ice cream defect detection"""
    
    @staticmethod
    def center_focused_tiler(img: np.ndarray) -> List[np.ndarray]:
        """Focus on center where ice cream typically appears"""
        h, w = img.shape[:2]
        center_h, center_w = h // 2, w // 2
        # Create overlapping tiles around center
        tile_size_h, tile_size_w = h // 3, w // 3
        
        tiles = []
        for dy in [-tile_size_h//2, 0, tile_size_h//2]:
            for dx in [-tile_size_w//2, 0, tile_size_w//2]:
                y1 = max(0, center_h + dy)
                y2 = min(h, y1 + tile_size_h)
                x1 = max(0, center_w + dx)
                x2 = min(w, x1 + tile_size_w)
                tiles.append(img[y1:y2, x1:x2])
        return tiles
    
    @staticmethod
    def radial_tiler(img: np.ndarray) -> List[np.ndarray]:
        """Radial tiling for circular ice cream detection"""
        h, w = img.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Create concentric rings
        tiles = []
        for radius in [min(h, w) // 6, min(h, w) // 4, min(h, w) // 3]:
            mask = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            inner_radius = radius - radius // 3
            mask = (distance >= inner_radius) & (distance <= radius)
            
            if np.any(mask):
                tile = img.copy()
                tile[~mask] = 0  # Zero out non-ring areas
                tiles.append(tile)
        
        return tiles if tiles else [img]  # Fallback to full image
    
    @staticmethod
    def edge_detection_tiler(img: np.ndarray) -> List[np.ndarray]:
        """Focus on edges where defects commonly appear"""
        h, w = img.shape[:2]
        
        # Create edge-focused tiles
        edge_width = min(h, w) // 8
        tiles = [
            img[:edge_width, :],           # Top edge
            img[-edge_width:, :],          # Bottom edge
            img[:, :edge_width],           # Left edge
            img[:, -edge_width:],          # Right edge
            img[h//4:3*h//4, w//4:3*w//4]  # Center region
        ]
        return tiles
    
    @staticmethod
    def multi_scale_tiler(img: np.ndarray) -> List[np.ndarray]:
        """Multi-scale analysis for different defect sizes"""
        h, w = img.shape[:2]
        tiles = []
        
        # Different scale tiles
        scales = [(2, 2), (3, 3), (4, 4), (1, 4)]  # (rows, cols)
        for rows, cols in scales:
            tile_h, tile_w = h // rows, w // cols
            for i in range(rows):
                for j in range(cols):
                    y1, y2 = i * tile_h, (i + 1) * tile_h
                    x1, x2 = j * tile_w, (j + 1) * tile_w
                    tiles.append(img[y1:y2, x1:x2])
        
        return tiles
    
    @staticmethod
    def magnum_stick_tiler(img: np.ndarray) -> List[np.ndarray]:
        """Specialized tiling for Magnum stick ice cream avoiding foil interference"""
        h, w = img.shape[:2]
        tiles = []
        
        # Main ice cream body (avoiding side foil areas)
        # Focus on central 60% width to avoid side foils
        safe_margin = w // 5  # 20% margin on each side
        center_region = img[:, safe_margin:w-safe_margin]
        
        # Vertical strips in the safe center region (avoiding foil)
        center_w = center_region.shape[1]
        strip_width = center_w // 3
        for i in range(3):
            x1 = i * strip_width
            x2 = (i + 1) * strip_width if i < 2 else center_w
            tiles.append(center_region[:, x1:x2])
        
        # Horizontal layers (top, middle, bottom) in safe zone
        layer_height = h // 3
        for i in range(3):
            y1 = i * layer_height
            y2 = (i + 1) * layer_height if i < 2 else h
            tiles.append(center_region[y1:y2, :])
        
        # Small focused tiles on ice cream tip and center
        # Top section (ice cream tip)
        tip_height = h // 4
        tiles.append(center_region[:tip_height, :])
        
        # Center core region (most important for quality)
        core_y1, core_y2 = h // 4, 3 * h // 4
        core_x1, core_x2 = center_w // 4, 3 * center_w // 4
        tiles.append(center_region[core_y1:core_y2, core_x1:core_x2])
        
        # Bottom section near stick
        bottom_height = h // 4
        tiles.append(center_region[-bottom_height:, :])
        
        return tiles
    
    @staticmethod
    def foil_resistant_tiler(img: np.ndarray) -> List[np.ndarray]:
        """Tiling strategy specifically designed to avoid packaging foil interference"""
        h, w = img.shape[:2]
        tiles = []
        
        # Define safe zones avoiding typical foil positions
        # Foil usually comes from sides (left 15%, right 15%)
        foil_margin = int(w * 0.15)
        safe_zone = img[:, foil_margin:w-foil_margin]
        safe_w = safe_zone.shape[1]
        
        # Create overlapping tiles in safe zone only
        tile_overlap = 0.2  # 20% overlap between tiles
        
        # Vertical tiles (avoiding foil sides)
        num_v_tiles = 4
        tile_width = safe_w // (num_v_tiles - tile_overlap * (num_v_tiles - 1))
        step_width = int(tile_width * (1 - tile_overlap))
        
        for i in range(num_v_tiles):
            x1 = min(i * step_width, safe_w - tile_width)
            x2 = min(x1 + tile_width, safe_w)
            if x2 - x1 > safe_w // 6:  # Only if tile is meaningful size
                tiles.append(safe_zone[:, x1:x2])
        
        # Horizontal tiles (full height analysis)
        num_h_tiles = 5
        tile_height = h // (num_h_tiles - tile_overlap * (num_h_tiles - 1))
        step_height = int(tile_height * (1 - tile_overlap))
        
        for i in range(num_h_tiles):
            y1 = min(i * step_height, h - tile_height)
            y2 = min(y1 + tile_height, h)
            if y2 - y1 > h // 8:  # Only if tile is meaningful size
                tiles.append(safe_zone[y1:y2, :])
        
        # Central focus tile (most important area)
        center_y1, center_y2 = h // 3, 2 * h // 3
        center_x1, center_x2 = safe_w // 4, 3 * safe_w // 4
        tiles.append(safe_zone[center_y1:center_y2, center_x1:center_x2])
        
        return tiles if tiles else [safe_zone]  # Fallback to safe zone


# ─── Enhanced Image Preprocessing ──────────────────────────
class IceCreamPreprocessor:
    """Advanced preprocessing specifically for ice cream images"""
    
    @staticmethod
    def enhance_ice_cream_features(img: np.ndarray) -> np.ndarray:
        """Enhance features relevant to ice cream quality"""
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Enhance L channel (lightness) - important for ice cream texture
        l_channel = lab[:, :, 0]
        l_enhanced = cv2.equalizeHist(l_channel)
        lab[:, :, 0] = l_enhanced
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply gentle sharpening for texture details
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1 + np.eye(3).flatten().reshape(3,3) * 0.9)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def adaptive_roi_detection(img: np.ndarray) -> Tuple[float, float, float, float]:
        """Automatically detect ice cream ROI using image analysis - optimized for Magnum sticks"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the ice cream)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding and normalize
            padding = 20
            img_h, img_w = img.shape[:2]
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_w - x, w + 2 * padding)
            h = min(img_h - y, h + 2 * padding)
            
            # Return normalized coordinates
            return (x / img_w, y / img_h, w / img_w, h / img_h)
        
        # Fallback to center region
        return (0.25, 0.25, 0.5, 0.5)
    
    @staticmethod
    def magnum_roi_detection(img: np.ndarray) -> Tuple[float, float, float, float]:
        """Specialized ROI detection for Magnum stick ice cream avoiding foil areas"""
        h, w = img.shape[:2]
        
        # Convert to different color spaces for better foil detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Detect potential foil areas (usually bright/metallic)
        # Foil typically has high brightness and low saturation
        brightness = hsv[:, :, 2]  # V channel
        saturation = hsv[:, :, 1]  # S channel
        
        # Create mask for potential foil areas (bright + low saturation)
        foil_mask = (brightness > 200) & (saturation < 50)
        
        # Detect foil columns (vertical areas with high foil density)
        foil_column_density = np.mean(foil_mask, axis=0)
        foil_columns = foil_column_density > 0.3  # 30% foil density threshold
        
        # Find safe horizontal range (avoiding foil columns)
        safe_columns = ~foil_columns
        if np.any(safe_columns):
            safe_indices = np.where(safe_columns)[0]
            x_start = safe_indices[0] / w
            x_end = safe_indices[-1] / w
            roi_width = x_end - x_start
        else:
            # Fallback: use center 60% to avoid typical foil areas
            x_start = 0.2
            roi_width = 0.6
        
        # For Magnum sticks, usually want full height but safe width
        y_start = 0.05  # Small top margin
        roi_height = 0.9  # Most of the height
        
        # Ensure minimum width
        if roi_width < 0.3:
            center_x = x_start + roi_width / 2
            x_start = max(0, min(0.7, center_x - 0.15))
            roi_width = 0.3
        
        return (x_start, y_start, roi_width, roi_height)
    
    @staticmethod
    def detect_foil_interference(img: np.ndarray) -> Dict[str, float]:
        """Detect and quantify foil interference in the image"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Foil detection parameters
        brightness = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        
        # Potential foil areas
        foil_mask = (brightness > 180) & (saturation < 80)
        foil_percentage = np.mean(foil_mask) * 100
        
        # Side foil detection (left and right 20%)
        h, w = img.shape[:2]
        left_margin = w // 5
        right_margin = 4 * w // 5
        
        left_foil = np.mean(foil_mask[:, :left_margin]) * 100
        right_foil = np.mean(foil_mask[:, right_margin:]) * 100
        
        # Top foil detection
        top_margin = h // 5
        top_foil = np.mean(foil_mask[:top_margin, :]) * 100
        
        return {
            "total_foil_percentage": float(foil_percentage),
            "left_foil_percentage": float(left_foil),
            "right_foil_percentage": float(right_foil),
            "top_foil_percentage": float(top_foil),
            "has_significant_foil": foil_percentage > 15,
            "has_side_foil": max(left_foil, right_foil) > 20
        }


# ─── Enhanced Embedding and Analysis ───────────────────────
def load_and_embed_enhanced(folder: str, embedders: Dict[str, ResNetTiledEmbedder], 
                          tiling_strategy: str = "multi_scale",
                          use_adaptive_roi: bool = True,
                          logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Enhanced embedding with multiple architectures and strategies"""
    
    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for p in glob.glob(os.path.join(folder, ext))
    )
    
    if not paths:
        raise ValueError(f"No images found in {folder!r}")
    
    if logger:
        logger.info(f"Found {len(paths)} images in {folder}")
    
    # Initialize tiling strategy
    tiler_map = {
        "center_focused": IceCreamTilingStrategies.center_focused_tiler,
        "radial": IceCreamTilingStrategies.radial_tiler,
        "edge_detection": IceCreamTilingStrategies.edge_detection_tiler,
        "multi_scale": IceCreamTilingStrategies.multi_scale_tiler,
        "magnum_stick": IceCreamTilingStrategies.magnum_stick_tiler,
        "foil_resistant": IceCreamTilingStrategies.foil_resistant_tiler
    }
    
    tiler = tiler_map.get(tiling_strategy, IceCreamTilingStrategies.multi_scale_tiler)
    preprocessor = IceCreamPreprocessor()
    
    # Store embeddings for each architecture
    all_embeddings = {arch: [] for arch in embedders.keys()}
    
    for p in tqdm(paths, desc=f"Processing {os.path.basename(folder)}"):
        try:
            # Load and preprocess image
            frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            enhanced_frame = preprocessor.enhance_ice_cream_features(frame)
            
            # Detect foil interference
            foil_info = preprocessor.detect_foil_interference(enhanced_frame)
            
            # Adaptive ROI detection based on foil presence
            if use_adaptive_roi:
                if foil_info["has_side_foil"] or tiling_strategy in ["magnum_stick", "foil_resistant"]:
                    roi = preprocessor.magnum_roi_detection(enhanced_frame)
                else:
                    roi = preprocessor.adaptive_roi_detection(enhanced_frame)
            else:
                roi = (0.275, 0.0, 0.431, 1.0)  # Default ROI
            
            # Log foil detection if significant
            if logger and foil_info["has_significant_foil"]:
                logger.info(f"Foil detected in {os.path.basename(p)}: {foil_info['total_foil_percentage']:.1f}% coverage")
            
            # Generate embeddings with each architecture
            for arch_name, embedder in embedders.items():
                emb = embedder.embed_frame(enhanced_frame, roi, tiler)
                all_embeddings[arch_name].append(emb)
                
        except Exception as e:
            if logger:
                logger.warning(f"Failed to process {p}: {e}")
            continue
    
    # Concatenate embeddings
    result = {}
    for arch_name in embedders.keys():
        if all_embeddings[arch_name]:
            result[arch_name] = torch.cat(all_embeddings[arch_name], dim=0)
        else:
            if logger:
                logger.warning(f"No valid embeddings for architecture {arch_name}")
    
    return result, paths


# ─── Advanced Anomaly Scoring ──────────────────────────────
class AdvancedAnomalyScorer:
    """Enhanced anomaly scoring with multiple methods"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
    
    def compute_ensemble_scores(self, reference_embeddings: Dict[str, torch.Tensor], 
                              test_embeddings: Dict[str, torch.Tensor],
                              k: int = 5, method: str = "weighted_avg") -> Tuple[List[float], Dict]:
        """Compute ensemble anomaly scores using multiple architectures"""
        
        all_scores = {}
        weights = {"resnet18": 1.0, "resnet34": 1.2, "resnet50": 1.5, "resnet101": 1.8, "resnet152": 2.0}
        
        # Compute scores for each architecture
        for arch in reference_embeddings.keys():
            if arch in test_embeddings:
                arch_scores = []
                ref_emb = reference_embeddings[arch]
                test_emb = test_embeddings[arch]
                
                for i in range(test_emb.shape[0]):
                    score = ResNetTiledEmbedder.kmeans_distance(ref_emb, test_emb[i:i+1], k=k)
                    arch_scores.append(score)
                
                all_scores[arch] = arch_scores
        
        if not all_scores:
            raise ValueError("No valid scores computed")
        
        # Ensemble scoring
        num_samples = len(next(iter(all_scores.values())))
        ensemble_scores = []
        
        for i in range(num_samples):
            if method == "weighted_avg":
                weighted_sum = sum(weights.get(arch, 1.0) * scores[i] for arch, scores in all_scores.items())
                total_weight = sum(weights.get(arch, 1.0) for arch in all_scores.keys())
                ensemble_scores.append(weighted_sum / total_weight)
            elif method == "max":
                ensemble_scores.append(max(scores[i] for scores in all_scores.values()))
            elif method == "median":
                ensemble_scores.append(np.median([scores[i] for scores in all_scores.values()]))
            else:  # simple average
                ensemble_scores.append(np.mean([scores[i] for scores in all_scores.values()]))
        
        return ensemble_scores, all_scores
    
    def compute_statistical_threshold(self, scores: List[float]) -> Tuple[float, Dict]:
        """Compute adaptive threshold using statistical methods"""
        scores_array = np.array(scores)
        
        # Basic statistics
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        median_score = np.median(scores_array)
        
        # Confidence interval based threshold
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        confidence_threshold = mean_score + z_score * std_score
        
        # Interquartile range based threshold
        q75, q25 = np.percentile(scores_array, [75, 25])
        iqr = q75 - q25
        iqr_threshold = q75 + 1.5 * iqr
        
        # Isolation Forest based threshold
        scores_reshaped = scores_array.reshape(-1, 1)
        scaled_scores = self.scaler.fit_transform(scores_reshaped)
        outlier_predictions = self.isolation_forest.fit_predict(scaled_scores)
        if np.any(outlier_predictions == -1):
            isolation_threshold = np.min(scores_array[outlier_predictions == -1])
        else:
            isolation_threshold = confidence_threshold
        
        # Choose the most conservative threshold
        adaptive_threshold = min(confidence_threshold, iqr_threshold, isolation_threshold)
        
        stats_info = {
            "mean": mean_score,
            "std": std_score,
            "median": median_score,
            "confidence_threshold": confidence_threshold,
            "iqr_threshold": iqr_threshold,
            "isolation_threshold": isolation_threshold,
            "adaptive_threshold": adaptive_threshold,
            "confidence_level": self.confidence_level
        }
        
        return adaptive_threshold, stats_info


# ─── Enhanced Main Function ─────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Advanced Ice Cream Quality Control System with Ensemble Learning")
    
    # Basic arguments with default paths
    ap.add_argument("--good_dir", default="good_images", 
                    help="Directory with good ice cream images (default: good_images)")
    ap.add_argument("--test_dir", default="test_images", 
                    help="Directory with test images (default: test_images)")
    ap.add_argument("--anomaly_folder", default="seperated_images", 
                    help="Output folder for detected anomalies (default: seperated_images)")
    
    # Advanced parameters
    ap.add_argument("--architectures", nargs="+", default=["resnet18", "resnet50"],
                    choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                    help="ResNet architectures to use in ensemble")
    ap.add_argument("--tiling_strategy", default="magnum_stick",
                    choices=["center_focused", "radial", "edge_detection", "multi_scale", "magnum_stick", "foil_resistant"],
                    help="Tiling strategy for defect detection (magnum_stick recommended for Magnum ice creams)")
    ap.add_argument("--ensemble_method", default="weighted_avg",
                    choices=["weighted_avg", "max", "median", "simple_avg"],
                    help="Ensemble scoring method")
    ap.add_argument("--confidence", type=float, default=0.95,
                    help="Confidence level for adaptive thresholding")
    ap.add_argument("--k", type=int, default=5, help="k for k-NN distance")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", None])
    ap.add_argument("--use_adaptive_roi", action="store_true", default=True,
                    help="Use adaptive ROI detection")
    ap.add_argument("--manual_threshold", type=float, default=None,
                    help="Manual threshold (overrides adaptive)")
    ap.add_argument("--save_reports", action="store_true", default=True,
                    help="Save detailed analysis reports")
    
    args = ap.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Enhanced Ice Cream Quality Control System")
    logger.info(f"Parameters: {vars(args)}")
    
    # Prepare output directories
    anomaly_dir = Path(args.anomaly_folder)
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_reports:
        reports_dir = anomaly_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
    
    # Initialize embedders for each architecture
    logger.info(f"Initializing embedders for architectures: {args.architectures}")
    embedders = {}
    for arch in args.architectures:
        try:
            embedders[arch] = ResNetTiledEmbedder(arch, args.device)
            logger.info(f"Successfully initialized {arch}")
        except Exception as e:
            logger.error(f"Failed to initialize {arch}: {e}")
    
    if not embedders:
        logger.error("No embedders initialized successfully")
        return
    
    # Load and embed reference set (good images)
    logger.info("Processing reference set (good images)...")
    start_time = time.time()
    reference_embeddings, _ = load_and_embed_enhanced(
        args.good_dir, embedders, args.tiling_strategy, args.use_adaptive_roi, logger
    )
    ref_time = time.time() - start_time
    logger.info(f"Reference processing completed in {ref_time:.2f} seconds")
    
    # Load and embed test images
    logger.info("Processing test images...")
    start_time = time.time()
    test_embeddings, test_paths = load_and_embed_enhanced(
        args.test_dir, embedders, args.tiling_strategy, args.use_adaptive_roi, logger
    )
    test_time = time.time() - start_time
    logger.info(f"Test processing completed in {test_time:.2f} seconds")
    
    # Initialize advanced scorer
    scorer = AdvancedAnomalyScorer(args.confidence)
    
    # Compute ensemble scores
    logger.info("Computing ensemble anomaly scores...")
    ensemble_scores, individual_scores = scorer.compute_ensemble_scores(
        reference_embeddings, test_embeddings, args.k, args.ensemble_method
    )
    
    # Determine threshold
    if args.manual_threshold is not None:
        threshold = args.manual_threshold
        stats_info = {"manual_threshold": threshold}
        logger.info(f"Using manual threshold: {threshold}")
    else:
        threshold, stats_info = scorer.compute_statistical_threshold(ensemble_scores)
        logger.info(f"Adaptive threshold computed: {threshold:.6f}")
        logger.info(f"Threshold statistics: {stats_info}")
    
    # Identify anomalies
    anomaly_indices = [i for i, score in enumerate(ensemble_scores) if score > threshold]
    logger.info(f"Detected {len(anomaly_indices)} anomalies out of {len(ensemble_scores)} images")
    
    # Sort and save anomalies
    ranked_anomalies = sorted(
        [(i, ensemble_scores[i], test_paths[i]) for i in anomaly_indices],
        key=lambda x: x[1], reverse=True
    )
    
    for rank, (idx, score, src_path) in enumerate(ranked_anomalies, 1):
        confidence_level = min(100, (score - threshold) / threshold * 100)
        name = f"{rank:05d}_score_{score:.6f}_conf_{confidence_level:.1f}%_{os.path.basename(src_path)}"
        dest_path = anomaly_dir / name
        shutil.copy2(src_path, dest_path)
        logger.info(f"Anomaly {rank}: {os.path.basename(src_path)} (score: {score:.6f}, confidence: {confidence_level:.1f}%)")
    
    # Generate detailed report
    if args.save_reports:
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "parameters": vars(args),
            "processing_times": {
                "reference_processing": ref_time,
                "test_processing": test_time
            },
            "statistics": stats_info,
            "results": {
                "total_test_images": len(ensemble_scores),
                "detected_anomalies": len(anomaly_indices),
                "anomaly_rate": len(anomaly_indices) / len(ensemble_scores) * 100,
                "threshold_used": threshold
            },
            "individual_architecture_scores": {
                arch: [float(score) for score in scores] 
                for arch, scores in individual_scores.items()
            },
            "ensemble_scores": [float(score) for score in ensemble_scores],
            "anomaly_details": [
                {
                    "rank": rank,
                    "filename": os.path.basename(src_path),
                    "score": float(score),
                    "confidence_percentage": min(100, (score - threshold) / threshold * 100)
                }
                for rank, (idx, score, src_path) in enumerate(ranked_anomalies, 1)
            ]
        }
        
        report_file = reports_dir / f"quality_control_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Detailed report saved to {report_file}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(ensemble_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    plt.axvline(np.mean(ensemble_scores), color='green', linestyle=':', label=f'Mean: {np.mean(ensemble_scores):.4f}')
    plt.xlabel('Ensemble Anomaly Score')
    plt.ylabel('Number of Images')
    plt.title('Ice Cream Quality Control - Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Architecture comparison
    plt.subplot(2, 2, 2)
    arch_means = {arch: np.mean(scores) for arch, scores in individual_scores.items()}
    plt.bar(arch_means.keys(), arch_means.values(), color='skyblue', edgecolor='navy')
    plt.xlabel('Architecture')
    plt.ylabel('Mean Score')
    plt.title('Architecture Performance Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Score correlation between architectures
    if len(individual_scores) >= 2:
        plt.subplot(2, 2, 3)
        arch_names = list(individual_scores.keys())
        arch1_scores = individual_scores[arch_names[0]]
        arch2_scores = individual_scores[arch_names[1]]
        plt.scatter(arch1_scores, arch2_scores, alpha=0.6, color='purple')
        plt.xlabel(f'{arch_names[0]} Scores')
        plt.ylabel(f'{arch_names[1]} Scores')
        plt.title('Architecture Score Correlation')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(arch1_scores, arch2_scores)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Anomaly confidence distribution
    plt.subplot(2, 2, 4)
    if ranked_anomalies:
        confidence_levels = [min(100, (score - threshold) / threshold * 100) for _, score, _ in ranked_anomalies]
        plt.hist(confidence_levels, bins=20, color='red', alpha=0.7, edgecolor='darkred')
        plt.xlabel('Anomaly Confidence %')
        plt.ylabel('Number of Anomalies')
        plt.title('Anomaly Confidence Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Anomalies Detected', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('Anomaly Confidence Distribution')
    
    plt.tight_layout()
    
    # Save plot
    if args.save_reports:
        plot_file = reports_dir / f"analysis_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis plots saved to {plot_file}")
    
    plt.show()
    
    # Final summary
    print(f"\n{'='*60}")
    print("🍦 ICE CREAM QUALITY CONTROL SUMMARY 🍦")
    print(f"{'='*60}")
    print(f"📊 Total images processed: {len(ensemble_scores)}")
    print(f"🚨 Anomalies detected: {len(anomaly_indices)} ({len(anomaly_indices)/len(ensemble_scores)*100:.1f}%)")
    print(f"🎯 Threshold used: {threshold:.6f}")
    print(f"📈 Mean score: {np.mean(ensemble_scores):.6f}")
    print(f"📉 Score range: {min(ensemble_scores):.6f} - {max(ensemble_scores):.6f}")
    print(f"⏱️  Processing time: {ref_time + test_time:.1f} seconds")
    print(f"💾 Results saved to: {anomaly_dir}")
    if args.save_reports:
        print(f"📋 Detailed reports: {reports_dir}")
    print(f"{'='*60}")
    
    logger.info("Ice Cream Quality Control System completed successfully")


if __name__ == "__main__":
    main()
