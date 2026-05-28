"""Helpers for the SuperBitDev/fire1 YOLO fire model."""

from __future__ import annotations

import json
import math
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from yolo_utils import Detection


FIRE_MODEL_REPO_ID = "SuperBitDev/fire1"
FIRE_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "fire1"
FIRE_CLASS_NAMES = ["fire", "smoke", "fire extinguisher"]
MODEL_CLASS_ORDER = ["fire", "fire extinguisher", "smoke"]
DEFAULT_FIRE_SAMPLE_URL = (
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
    "Woolsey%20Flames%20%2854811019352%29.jpg?width=1024"
)
DEFAULT_FIRE_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "samples" / "fire" / "woolsey_flames.jpg"
USER_AGENT = "swarm-tello-drone/1.0 (+https://github.com/swarm-subnet/swarm-tello-drone)"


def open_url(url: str, timeout: int):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def list_huggingface_repo_files(repo_id: str = FIRE_MODEL_REPO_ID) -> list[str]:
    """List files in a public Hugging Face model repo without extra dependencies."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    with open_url(api_url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    siblings = payload.get("siblings", [])
    return [item["rfilename"] for item in siblings if item.get("rfilename")]


def select_model_weight_file(files: list[str]) -> str:
    """Pick the most likely YOLO weight file from a Hugging Face repo."""
    weight_files = [
        file for file in files
        if file.lower().endswith((".pt", ".onnx"))
    ]
    if not weight_files:
        raise RuntimeError("No .pt or .onnx YOLO weight file found in the Hugging Face repo.")

    def score(path: str) -> tuple[int, int, str]:
        lower = path.lower()
        if lower == "weights.onnx":
            priority = 0
        elif "best" in lower:
            priority = 0
        elif "last" in lower:
            priority = 1
        else:
            priority = 2
        return priority, len(path), path

    return sorted(weight_files, key=score)[0]


def download_huggingface_file(repo_id: str, filename: str, output_path: Path) -> Path:
    """Download one public file from Hugging Face if it is not already cached."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    encoded_filename = quote(filename, safe="/")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{encoded_filename}"
    print(f"Downloading model weight: {url}")
    with open_url(url, timeout=120) as response:
        with output_path.open("wb") as file:
            shutil.copyfileobj(response, file)
    return output_path


def get_fire_model_path(repo_id: str = FIRE_MODEL_REPO_ID, model_dir: Path = FIRE_MODEL_DIR) -> Path:
    """Find and cache the fire model's YOLO weight file."""
    cached_weights = model_dir / "weights.onnx"
    if cached_weights.exists() and cached_weights.stat().st_size > 0:
        return cached_weights

    files = list_huggingface_repo_files(repo_id)
    selected_file = select_model_weight_file(files)
    output_path = model_dir / Path(selected_file).name
    print(f"Selected fire model weight from {repo_id}: {selected_file}")
    return download_huggingface_file(repo_id, selected_file, output_path)


def download_fire_sample(output_path: Path = DEFAULT_FIRE_SAMPLE_PATH) -> Path:
    """Download a sample fire image for the offline model test."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    print(f"Downloading fire sample image: {DEFAULT_FIRE_SAMPLE_URL}")
    with open_url(DEFAULT_FIRE_SAMPLE_URL, timeout=60) as response:
        with output_path.open("wb") as file:
            shutil.copyfileobj(response, file)
    return output_path


class FireOnnxDetector:
    """ONNX Runtime detector using the SuperBitDev/fire1 miner post-processing."""

    iou_threshold = 0.55
    cross_iou_threshold = 0.8
    max_detections = 150
    miner_thresholds = np.array([0.6, 0.4, 0.3], dtype=np.float32)
    miner_bonuses = np.array([0.0, 0.1, 0.15], dtype=np.float32)

    def __init__(self, model_path: Path, profile: str = "miner") -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.class_names = FIRE_CLASS_NAMES
        self.class_remap = np.array(
            [self.class_names.index(name) for name in MODEL_CLASS_ORDER],
            dtype=np.int32,
        )
        self.profile = profile
        self.confidence_thresholds = self.miner_thresholds.copy()
        self.rescue_bonuses = self.miner_bonuses.copy()
        self.min_box_area = 14 * 14
        self.min_side = 8
        self.max_aspect_ratio = 8.0

        if profile == "candle":
            # Keep the miner pipeline, but admit smaller/lower-confidence fire.
            self.confidence_thresholds = np.array([0.22, 0.30, 0.3], dtype=np.float32)
            self.rescue_bonuses = np.array([0.08, 0.10, 0.15], dtype=np.float32)
            self.min_box_area = 6 * 6
            self.min_side = 4
        elif profile != "miner":
            raise ValueError("profile must be 'miner' or 'candle'")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            try:
                ort.preload_dlls()
            except Exception:
                pass
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
        except Exception:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = self._safe_dim(input_shape[2], default=1280)
        self.input_width = self._safe_dim(input_shape[3], default=1280)
        self.use_tta = True

    @staticmethod
    def _safe_dim(value, default: int) -> int:
        return value if isinstance(value, int) and value > 0 else default

    def _letterbox(self, image, new_shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
        height, width = image.shape[:2]
        new_width, new_height = new_shape
        ratio = min(new_width / width, new_height / height)
        resized_width = int(round(width * ratio))
        resized_height = int(round(height * ratio))

        if (resized_width, resized_height) != (width, height):
            interpolation = cv2.INTER_CUBIC if ratio > 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

        dw = (new_width - resized_width) / 2.0
        dh = (new_height - resized_height) / 2.0
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, ratio, (dw, dh)

    def _preprocess(self, image_bgr) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
        original_height, original_width = image_bgr.shape[:2]
        image, ratio, pad = self._letterbox(image_bgr, (self.input_width, self.input_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(image, dtype=np.float32), ratio, pad, (original_width, original_height)

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        output = np.empty_like(boxes)
        output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return output

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        width, height = image_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
        return boxes

    @staticmethod
    def _hard_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty(0, dtype=np.intp)

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(-scores)
        keep: list[int] = []

        while len(order) > 0:
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[current], x1[rest])
            yy1 = np.maximum(y1[current], y1[rest])
            xx2 = np.minimum(x2[current], x2[rest])
            yy2 = np.minimum(y2[current], y2[rest])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[current] + areas[rest] - intersection + 1e-7
            iou = intersection / union
            order = rest[iou <= iou_threshold]

        return np.array(keep, dtype=np.intp)

    def _per_class_hard_nms(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=np.intp)

        kept_indices: list[int] = []
        for class_id in np.unique(class_ids):
            indices = np.where(class_ids == class_id)[0]
            class_keep = self._hard_nms(boxes[indices], scores[indices], self.iou_threshold)
            kept_indices.extend(indices[class_keep].tolist())
        kept_indices.sort()
        return np.array(kept_indices, dtype=np.intp)

    def _cross_class_dedup(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) <= 1:
            return boxes, scores, class_ids

        areas = (
            np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        )
        margins = scores - self.confidence_thresholds[class_ids]
        order = np.lexsort((-areas, -margins))
        suppressed = np.zeros(len(boxes), dtype=bool)
        keep: list[int] = []

        for index in order:
            if suppressed[index]:
                continue
            keep.append(int(index))
            box = boxes[index]
            xx1 = np.maximum(box[0], boxes[:, 0])
            yy1 = np.maximum(box[1], boxes[:, 1])
            xx2 = np.minimum(box[2], boxes[:, 2])
            yy2 = np.minimum(box[3], boxes[:, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            box_area = max(1e-7, float((box[2] - box[0]) * (box[3] - box[1])))
            iou = intersection / (box_area + areas - intersection + 1e-7)
            duplicates = iou > self.cross_iou_threshold
            duplicates[index] = False
            suppressed |= duplicates

        keep_indices = np.array(keep, dtype=np.intp)
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    @staticmethod
    def _max_score_per_cluster(
        post_boxes: np.ndarray,
        post_classes: np.ndarray,
        full_boxes: np.ndarray,
        full_scores: np.ndarray,
        full_classes: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        if len(post_boxes) == 0:
            return np.empty(0, dtype=np.float32)

        full_areas = (
            np.maximum(0.0, full_boxes[:, 2] - full_boxes[:, 0])
            * np.maximum(0.0, full_boxes[:, 3] - full_boxes[:, 1])
        )
        output = np.empty(len(post_boxes), dtype=np.float32)
        for index, box in enumerate(post_boxes):
            xx1 = np.maximum(box[0], full_boxes[:, 0])
            yy1 = np.maximum(box[1], full_boxes[:, 1])
            xx2 = np.minimum(box[2], full_boxes[:, 2])
            yy2 = np.minimum(box[3], full_boxes[:, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
            iou = intersection / (box_area + full_areas - intersection + 1e-7)
            cluster = (iou >= iou_threshold) & (full_classes == post_classes[index])
            output[index] = float(np.max(full_scores[cluster])) if np.any(cluster) else 0.0
        return output

    def _confidence_filter_mask(self, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return np.zeros(0, dtype=bool)

        thresholds = self.confidence_thresholds[class_ids]
        keep = scores >= thresholds
        for class_id in np.unique(class_ids):
            bonus = float(self.rescue_bonuses[class_id])
            if bonus <= 0.0:
                continue
            class_mask = class_ids == class_id
            if keep[class_mask].any():
                continue
            indices = np.where(class_mask)[0]
            top = int(indices[int(np.argmax(scores[indices]))])
            if scores[top] >= self.confidence_thresholds[class_id] - bonus:
                keep[top] = True
        return keep

    def _filter_sane_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        original_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return boxes, scores, class_ids

        original_width, original_height = original_size
        image_area = float(original_width * original_height)
        keep: list[int] = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            if width < self.min_side or height < self.min_side:
                continue
            area = width * height
            if area < self.min_box_area:
                continue
            if area > 0.95 * image_area:
                continue
            aspect_ratio = max(width / max(height, 1e-6), height / max(width, 1e-6))
            if aspect_ratio > self.max_aspect_ratio:
                continue
            keep.append(index)

        if not keep:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        keep_indices = np.array(keep, dtype=np.intp)
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    def _per_view_pipeline(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) > 1:
            keep = self._per_class_hard_nms(boxes, scores, class_ids)
            boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        if len(scores) > self.max_detections:
            top = np.argsort(-scores)[: self.max_detections]
            boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]
        if len(boxes) > 1:
            boxes, scores, class_ids = self._cross_class_dedup(boxes, scores, class_ids)
        return boxes, scores, class_ids

    def _build_detections(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    label=self.class_names[int(class_id)],
                    confidence=float(score),
                    xyxy=(
                        int(math.floor(x1)),
                        int(math.floor(y1)),
                        int(math.ceil(x2)),
                        int(math.ceil(y2)),
                    ),
                )
            )
        return detections

    def _decode_final_detections(
        self,
        predictions: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        original_size: tuple[int, int],
    ) -> list[Detection]:
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            raise ValueError(f"Unexpected ONNX final-det output shape: {predictions.shape}")

        boxes = predictions[:, :4].astype(np.float32)
        scores = predictions[:, 4].astype(np.float32)
        class_ids = predictions[:, 5].astype(np.int32)
        valid_classes = class_ids < len(self.class_remap)
        boxes = boxes[valid_classes]
        scores = scores[valid_classes]
        class_ids = self.class_remap[class_ids[valid_classes]]

        keep = self._confidence_filter_mask(scores, class_ids)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if len(boxes) == 0:
            return []

        pad_width, pad_height = pad
        boxes[:, [0, 2]] -= pad_width
        boxes[:, [1, 3]] -= pad_height
        boxes /= ratio
        boxes = self._clip_boxes(boxes, original_size)
        boxes, scores, class_ids = self._filter_sane_boxes(boxes, scores, class_ids, original_size)
        if len(boxes) == 0:
            return []

        boxes, scores, class_ids = self._per_view_pipeline(boxes, scores, class_ids)
        return self._build_detections(boxes, scores, class_ids)

    def _decode_raw_yolo(
        self,
        predictions: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        original_size: tuple[int, int],
    ) -> list[Detection]:
        if predictions.ndim != 3 or predictions.shape[0] != 1:
            raise ValueError(f"Unexpected raw ONNX output shape: {predictions.shape}")
        predictions = predictions[0]
        if predictions.shape[0] <= 16 and predictions.shape[1] > predictions.shape[0]:
            predictions = predictions.T
        if predictions.ndim != 2 or predictions.shape[1] < 5:
            raise ValueError(f"Unexpected raw output shape: {predictions.shape}")

        boxes_xywh = predictions[:, :4].astype(np.float32)
        class_scores = predictions[:, 4:].astype(np.float32)
        if class_scores.shape[1] == 1:
            scores = class_scores[:, 0]
            class_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
            scores = class_scores[np.arange(len(class_scores)), class_ids]
        valid_classes = class_ids < len(self.class_remap)
        boxes_xywh = boxes_xywh[valid_classes]
        scores = scores[valid_classes]
        class_ids = self.class_remap[class_ids[valid_classes]]

        keep = self._confidence_filter_mask(scores, class_ids)
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if len(boxes_xywh) == 0:
            return []

        boxes = self._xywh_to_xyxy(boxes_xywh)
        pad_width, pad_height = pad
        boxes[:, [0, 2]] -= pad_width
        boxes[:, [1, 3]] -= pad_height
        boxes /= ratio
        boxes = self._clip_boxes(boxes, original_size)
        boxes, scores, class_ids = self._filter_sane_boxes(boxes, scores, class_ids, original_size)
        if len(boxes) == 0:
            return []

        boxes, scores, class_ids = self._per_view_pipeline(boxes, scores, class_ids)
        return self._build_detections(boxes, scores, class_ids)

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        original_size: tuple[int, int],
    ) -> list[Detection]:
        if output.ndim == 2 and output.shape[1] >= 6:
            return self._decode_final_detections(output, ratio, pad, original_size)
        if output.ndim == 3 and output.shape[0] == 1 and output.shape[2] == 6:
            return self._decode_final_detections(output, ratio, pad, original_size)
        return self._decode_raw_yolo(output, ratio, pad, original_size)

    def _predict_single(self, image_bgr) -> list[Detection]:
        input_tensor, ratio, pad, original_size = self._preprocess(image_bgr)
        expected_shape = (1, 3, self.input_height, self.input_width)
        if input_tensor.shape != expected_shape:
            raise ValueError(f"Bad input tensor shape={input_tensor.shape}, expected={expected_shape}")

        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self._postprocess(outputs[0], ratio, pad, original_size)

    def _predict_tta(self, image_bgr) -> list[Detection]:
        detections_original = self._predict_single(image_bgr)
        flipped = cv2.flip(image_bgr, 1)
        detections_flipped = self._predict_single(flipped)
        width = image_bgr.shape[1]

        mapped_flipped = [
            Detection(
                label=detection.label,
                confidence=detection.confidence,
                xyxy=(
                    width - detection.xyxy[2],
                    detection.xyxy[1],
                    width - detection.xyxy[0],
                    detection.xyxy[3],
                ),
            )
            for detection in detections_flipped
        ]
        all_detections = detections_original + mapped_flipped
        if not all_detections:
            return []

        boxes = np.array([detection.xyxy for detection in all_detections], dtype=np.float32)
        scores = np.array([detection.confidence for detection in all_detections], dtype=np.float32)
        class_ids = np.array([self.class_names.index(detection.label) for detection in all_detections], dtype=np.int32)

        hard_keep = self._per_class_hard_nms(boxes, scores, class_ids)
        if len(hard_keep) == 0:
            return []
        if len(hard_keep) > self.max_detections:
            top = np.argsort(-scores[hard_keep])[: self.max_detections]
            hard_keep = hard_keep[top]

        boosted_scores = self._max_score_per_cluster(
            boxes[hard_keep],
            class_ids[hard_keep],
            boxes,
            scores,
            class_ids,
            self.iou_threshold,
        )
        kept_boxes = boxes[hard_keep]
        kept_classes = class_ids[hard_keep]
        if len(kept_boxes) > 1:
            kept_boxes, boosted_scores, kept_classes = self._cross_class_dedup(
                kept_boxes,
                boosted_scores,
                kept_classes,
            )

        return self._build_detections(kept_boxes, boosted_scores, kept_classes)

    def predict(self, image_bgr) -> list[Detection]:
        if image_bgr is None:
            raise ValueError("Input image is None")
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError(f"Input is not numpy array: {type(image_bgr)}")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HWC 3-channel image, got shape={image_bgr.shape}")
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)

        detections = self._predict_tta(image_bgr) if self.use_tta else self._predict_single(image_bgr)
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


def keep_hazard_detections(detections: list[Detection], include_extinguisher: bool = False) -> list[Detection]:
    allowed = {"fire", "smoke"}
    if include_extinguisher:
        allowed.add("fire extinguisher")
    return [detection for detection in detections if detection.label.lower() in allowed]


def keep_fire_detections(detections: list[Detection], include_smoke: bool = True) -> list[Detection]:
    allowed = {"fire"}
    if include_smoke:
        allowed.add("smoke")
    return [detection for detection in detections if detection.label.lower() in allowed]


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def merge_overlapping_detections(detections: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    """Merge duplicate detections produced by overlapping tiles."""
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for kept in merged:
            if kept.label == detection.label and _box_iou(kept.xyxy, detection.xyxy) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            merged.append(detection)
    return merged


def iter_overlapping_tiles(frame_width: int, frame_height: int, overlap: float = 0.25):
    """Yield 2x2 tile boxes with configurable overlap, clipped to the frame."""
    overlap = max(0.0, min(0.45, overlap))
    tile_width = int(round(frame_width * (0.5 + overlap / 2.0)))
    tile_height = int(round(frame_height * (0.5 + overlap / 2.0)))
    starts_x = [0, max(0, frame_width - tile_width)]
    starts_y = [0, max(0, frame_height - tile_height)]

    seen: set[tuple[int, int, int, int]] = set()
    for y1 in starts_y:
        for x1 in starts_x:
            x2 = min(frame_width, x1 + tile_width)
            y2 = min(frame_height, y1 + tile_height)
            tile = (x1, y1, x2, y2)
            if tile not in seen:
                seen.add(tile)
                yield tile


def predict_tiled(
    detector: FireOnnxDetector,
    frame_bgr,
    include_full_frame: bool = True,
    overlap: float = 0.25,
) -> list[Detection]:
    """Run detector on full frame plus overlapping 2x2 tiles and map boxes back."""
    frame_height, frame_width = frame_bgr.shape[:2]
    detections: list[Detection] = []

    if include_full_frame:
        detections.extend(detector.predict(frame_bgr))

    for x1, y1, x2, y2 in iter_overlapping_tiles(frame_width, frame_height, overlap=overlap):
        tile = frame_bgr[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        for detection in detector.predict(tile):
            dx1, dy1, dx2, dy2 = detection.xyxy
            detections.append(
                Detection(
                    label=detection.label,
                    confidence=detection.confidence,
                    xyxy=(dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1),
                )
            )

    return merge_overlapping_detections(detections)


def predict_center_zoom(
    detector: FireOnnxDetector,
    frame_bgr,
    crop_fraction: float = 0.70,
    include_full_frame: bool = False,
) -> list[Detection]:
    """Run detector on a center crop and map boxes back to the full frame."""
    frame_height, frame_width = frame_bgr.shape[:2]
    crop_fraction = max(0.10, min(1.0, crop_fraction))
    crop_width = int(round(frame_width * crop_fraction))
    crop_height = int(round(frame_height * crop_fraction))
    x1 = max(0, (frame_width - crop_width) // 2)
    y1 = max(0, (frame_height - crop_height) // 2)
    x2 = min(frame_width, x1 + crop_width)
    y2 = min(frame_height, y1 + crop_height)

    detections: list[Detection] = []
    if include_full_frame:
        detections.extend(detector.predict(frame_bgr))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return detections

    for detection in detector.predict(crop):
        dx1, dy1, dx2, dy2 = detection.xyxy
        detections.append(
            Detection(
                label=detection.label,
                confidence=detection.confidence,
                xyxy=(dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1),
            )
        )

    return merge_overlapping_detections(detections)
