import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
from pathlib import Path
from uuid import uuid4
from collections import Counter
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, send_from_directory, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import torch

ZERO_SHOT_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
DATASET_DIR = BASE_DIR / "TrashType_Image_Dataset"
CUSTOM_MODEL_CANDIDATES = [
    BASE_DIR / "runs" / "classify" / "trash_cls" / "weights" / "best.pt",
    BASE_DIR / "runs" / "classify" / "trash_cls_v2" / "weights" / "best.pt",
]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
PATCH_GRID_SIZE = 3
PATCH_OVERRIDE_RATIO = 0.67
LOW_CONFIDENCE_THRESHOLD = 0.55
LOW_MARGIN_THRESHOLD = 0.12
ZERO_SHOT_MODEL_NAME = "ViT-B-32"
ZERO_SHOT_PRETRAINED = "laion2b_s34b_b79k"
ZERO_SHOT_PROMPTS = {
    "plastic": [
        "plastic waste such as plastic bags, plastic bottles, and plastic containers",
        "household plastic containers, detergent bottles, shampoo bottles, spray bottles made of plastic",
        "recyclable plastic packaging and plastic waste",
    ],
    "metal": [
        "metal waste such as aluminum cans and metal containers",
        "pile of metal cans and scrap metal waste",
    ],
    "paper": [
        "paper waste such as paper sheets and printed paper",
        "stack of used paper and paper waste",
    ],
    "cardboard": [
        "cardboard waste such as corrugated boxes and cartons",
        "pile of cardboard boxes and carton waste",
    ],
    "glass": [
        "glass waste such as glass bottles and jars",
        "broken glass and glass bottle waste",
    ],
    "trash": [
        "mixed trash waste",
        "general garbage waste with mixed materials",
    ],
}
DETECTOR_MODEL_SOURCE = str(BASE_DIR / "yolov8n.pt") if (BASE_DIR / "yolov8n.pt").exists() else "yolov8n.pt"
DETECTOR_CONF_THRESHOLD = 0.2
DETECTOR_MIN_AREA_RATIO = 0.03
LOCALIZATION_GRID_SIZE = 6
LOCALIZATION_WINDOW_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
REGION_MIN_SCORE = 0.24
REGION_SCORE_RATIO = 0.72
BOX_IOU_THRESHOLD = 0.32
BOX_CONTAINMENT_THRESHOLD = 0.78
MAX_DISPLAY_BOXES = 4
BOX_COLOR = (15, 159, 110, 255)
BOX_FILL = (15, 159, 110, 235)

for folder in (UPLOAD_DIR, RESULTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

CUSTOM_MODEL_PATH = next((path for path in CUSTOM_MODEL_CANDIDATES if path.exists()), None)
MODEL_SOURCE = str(CUSTOM_MODEL_PATH) if CUSTOM_MODEL_PATH else "yolov8n-cls.pt"
INFERENCE_IMAGE_SIZE = 320 if CUSTOM_MODEL_PATH and "trash_cls_v2" in str(CUSTOM_MODEL_PATH) else 224

@lru_cache(maxsize=1)
def get_model():
    return YOLO(MODEL_SOURCE)

@lru_cache(maxsize=1)
def get_detector_model():
    return YOLO(DETECTOR_MODEL_SOURCE)

IS_CUSTOM_MODEL = CUSTOM_MODEL_PATH is not None

def get_class_id_map():
    m = get_model()
    return {str(name).lower(): int(class_id) for class_id, name in m.names.items()}

def get_id_class_map():
    m = get_model()
    return {int(class_id): str(name).lower() for class_id, name in m.names.items()}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_font(size: int, bold: bool = False):
    font_candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def get_dataset_overview() -> tuple[int, int]:
    if not DATASET_DIR.exists():
        return 0, 0

    class_dirs = [path for path in DATASET_DIR.iterdir() if path.is_dir()]
    image_count = 0
    for class_dir in class_dirs:
        image_count += sum(1 for image_path in class_dir.iterdir() if allowed_file(image_path.name))

    return len(class_dirs), image_count


def format_class_label(label: str) -> str:
    return label.replace("_", " ").title()


def probabilities_to_predictions(probabilities, limit: int = 3) -> list[dict[str, float | str]]:
    values = probabilities.tolist()
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:limit]
    predictions = []
    for class_id, confidence in ranked:
        label = format_class_label(get_model().names.get(int(class_id), str(class_id)))
        predictions.append({"label": label, "confidence": round(float(confidence) * 100, 2)})
    return predictions


def split_image_into_patches(image: Image.Image, grid_size: int = PATCH_GRID_SIZE):
    width, height = image.size
    patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = int(col * width / grid_size)
            upper = int(row * height / grid_size)
            right = int((col + 1) * width / grid_size)
            lower = int((row + 1) * height / grid_size)
            patches.append((f"patch-{row + 1}-{col + 1}", image.crop((left, upper, right, lower))))
    return patches


def split_image_into_patch_entries(image: Image.Image, grid_size: int = PATCH_GRID_SIZE):
    width, height = image.size
    entries = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = int(col * width / grid_size)
            upper = int(row * height / grid_size)
            right = int((col + 1) * width / grid_size)
            lower = int((row + 1) * height / grid_size)
            entries.append(
                {
                    "row": row,
                    "col": col,
                    "box": (left, upper, right, lower),
                    "image": image.crop((left, upper, right, lower)),
                }
            )
    return entries


@lru_cache(maxsize=1)
def get_zero_shot_engine():
    if not ZERO_SHOT_AVAILABLE:
        return None

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        ZERO_SHOT_MODEL_NAME,
        pretrained=ZERO_SHOT_PRETRAINED,
        device="cpu",
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(ZERO_SHOT_MODEL_NAME)

    prompt_texts = []
    prompt_owners = []
    for class_label, prompts in ZERO_SHOT_PROMPTS.items():
        for prompt in prompts:
            prompt_texts.append(prompt)
            prompt_owners.append(class_label)

    with torch.no_grad():
        text_tokens = tokenizer(prompt_texts)
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return clip_model, preprocess, text_features.cpu(), prompt_owners


def compute_custom_probabilities_batch(images: list[Image.Image]):
    if not images:
        return []
    results = get_model()(images, imgsz=INFERENCE_IMAGE_SIZE, verbose=False)
    return [result.probs.data.float().cpu() for result in results]


def compute_zero_shot_probabilities_batch(images: list[Image.Image]):
    if not images:
        return [], []

    engine = get_zero_shot_engine()
    if engine is None:
        return [None] * len(images), [None] * len(images)

    clip_model, preprocess, text_features, prompt_owners = engine
    image_tensors = torch.stack([preprocess(image) for image in images])

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prompt_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu()

    probability_list = []
    diagnostics_list = []
    for row in prompt_scores:
        class_scores = {}
        for class_label in ZERO_SHOT_PROMPTS:
            prompt_values = [row[index].item() for index, owner in enumerate(prompt_owners) if owner == class_label]
            class_scores[class_label] = sum(prompt_values) / len(prompt_values)

        total_score = sum(class_scores.values()) or 1.0
        probabilities = torch.zeros(len(get_class_id_map()), dtype=torch.float32)
        for class_label, score in class_scores.items():
            class_id = get_class_id_map().get(class_label)
            if class_id is not None:
                probabilities[class_id] = score / total_score

        top_class_id = int(torch.argmax(probabilities).item())
        diagnostics = {
            "zero_shot_label": format_class_label(get_id_class_map()[top_class_id]),
            "zero_shot_confidence": round(float(probabilities[top_class_id].item()) * 100, 2),
        }
        probability_list.append(probabilities)
        diagnostics_list.append(diagnostics)

    return probability_list, diagnostics_list


def fuse_probability_vectors(custom_probabilities, zero_shot_probabilities, prefer_custom: bool = False):
    if zero_shot_probabilities is None:
        return custom_probabilities, "Custom Only"

    custom_top_class_id = int(torch.argmax(custom_probabilities).item())
    zero_shot_top_class_id = int(torch.argmax(zero_shot_probabilities).item())
    custom_top_confidence = float(custom_probabilities[custom_top_class_id].item())
    zero_shot_top_confidence = float(zero_shot_probabilities[zero_shot_top_class_id].item())

    zero_shot_weight = 0.55
    if custom_top_class_id != zero_shot_top_class_id and zero_shot_top_confidence >= 0.9:
        zero_shot_weight = 0.7
    elif prefer_custom and custom_top_confidence >= 0.75:
        zero_shot_weight = 0.4

    final_probabilities = custom_probabilities * (1 - zero_shot_weight) + zero_shot_probabilities * zero_shot_weight
    return final_probabilities, "Hybrid"


def compute_custom_probabilities(image: Image.Image):
    full_image = image.copy()
    patches = split_image_into_patches(image)

    full_result = get_model()(full_image, imgsz=INFERENCE_IMAGE_SIZE, verbose=False)[0]
    patch_results = get_model()([patch for _, patch in patches], imgsz=INFERENCE_IMAGE_SIZE, verbose=False)

    full_probs = full_result.probs.data.float().cpu()
    patch_probabilities = [result.probs.data.float().cpu() for result in patch_results]
    patch_average = sum(patch_probabilities) / len(patch_probabilities)

    patch_top_ids = [int(result.probs.top1) for result in patch_results]
    patch_vote_counts = Counter(patch_top_ids)
    patch_vote_class_id, patch_vote_count = patch_vote_counts.most_common(1)[0]
    patch_vote_ratio = patch_vote_count / len(patch_results)

    full_top_class_id = int(full_result.probs.top1)
    full_top_class_label = format_class_label(get_model().names.get(full_top_class_id, str(full_top_class_id)))
    patch_vote_label = format_class_label(get_model().names.get(patch_vote_class_id, str(patch_vote_class_id)))

    if patch_vote_ratio >= PATCH_OVERRIDE_RATIO and patch_vote_class_id != full_top_class_id:
        custom_probabilities = patch_average
        analysis_note = (
            f"Pile-aware patch voting was used because {patch_vote_count} of {len(patch_results)} patches agreed on "
            f"{patch_vote_label} while the whole-image guess leaned toward {full_top_class_label}."
        )
        analysis_mode = "Patch Voting"
    else:
        custom_probabilities = full_probs * 0.7 + patch_average * 0.3
        analysis_note = "Combined full-image and local patch analysis was used for a more stable prediction."
        analysis_mode = "Blended View"

    diagnostics = {
        "custom_label": full_top_class_label,
        "full_label": full_top_class_label,
        "patch_vote_label": patch_vote_label,
        "patch_vote_ratio": round(patch_vote_ratio * 100, 2),
        "analysis_mode": analysis_mode,
    }
    return custom_probabilities, analysis_note, diagnostics


def compute_zero_shot_probabilities(image: Image.Image):
    probabilities_list, diagnostics_list = compute_zero_shot_probabilities_batch([image])
    return probabilities_list[0], diagnostics_list[0]


def combine_probabilities(custom_probabilities, zero_shot_probabilities, custom_note: str, diagnostics: dict[str, float | str]):
    if zero_shot_probabilities is None:
        final_probabilities = custom_probabilities
        diagnostics["fusion_mode"] = "Custom Only"
        note = custom_note
    else:
        final_probabilities, fusion_mode = fuse_probability_vectors(
            custom_probabilities,
            zero_shot_probabilities,
            prefer_custom=diagnostics["analysis_mode"] == "Patch Voting",
        )
        diagnostics["fusion_mode"] = fusion_mode
        note = custom_note
        if diagnostics["fusion_mode"] == "Hybrid":
            custom_top_class_id = int(torch.argmax(custom_probabilities).item())
            zero_shot_top_class_id = int(torch.argmax(zero_shot_probabilities).item())
            zero_shot_top_confidence = float(zero_shot_probabilities[zero_shot_top_class_id].item())
        else:
            custom_top_class_id = zero_shot_top_class_id = 0
            zero_shot_top_confidence = 0.0

        if custom_top_class_id != zero_shot_top_class_id and zero_shot_top_confidence >= 0.9:
            note += " Zero-shot fallback was weighted more heavily because the uploaded image looked outside the training distribution."
        else:
            note += " Zero-shot fallback was blended in to improve external-image generalization."

    predictions = probabilities_to_predictions(final_probabilities)
    top_prediction = predictions[0] if predictions else None
    confidence_margin = 0.0
    if len(predictions) > 1:
        confidence_margin = (float(predictions[0]["confidence"]) - float(predictions[1]["confidence"])) / 100

    if top_prediction and (
        float(top_prediction["confidence"]) / 100 < LOW_CONFIDENCE_THRESHOLD or confidence_margin < LOW_MARGIN_THRESHOLD
    ):
        note += " The scene looks mixed or visually complex, so treat the top label as a best estimate."

    return predictions, top_prediction, note, diagnostics


def analyze_image(input_path: Path):
    with Image.open(input_path).convert("RGB") as image:
        custom_probabilities, custom_note, diagnostics = compute_custom_probabilities(image)
        zero_shot_probabilities, zero_shot_diagnostics = compute_zero_shot_probabilities(image)

    if zero_shot_diagnostics:
        diagnostics.update(zero_shot_diagnostics)

    return combine_probabilities(custom_probabilities, zero_shot_probabilities, custom_note, diagnostics)


def build_connected_patch_components(selected_entries: list[dict]):
    lookup = {(entry["row"], entry["col"]): entry for entry in selected_entries}
    visited = set()
    components = []

    for position in lookup:
        if position in visited:
            continue

        stack = [position]
        component = []
        visited.add(position)

        while stack:
            row, col = stack.pop()
            component.append(lookup[(row, col)])
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor = (row + row_offset, col + col_offset)
                    if neighbor in lookup and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        components.append(component)

    return components


def generate_region_scan_entries(
    image: Image.Image,
    grid_size: int = LOCALIZATION_GRID_SIZE,
    window_shapes: tuple[tuple[int, int], ...] = LOCALIZATION_WINDOW_SHAPES,
):
    width, height = image.size
    seen_boxes = set()
    entries = []

    for row_span, col_span in window_shapes:
        max_row = grid_size - row_span
        max_col = grid_size - col_span
        if max_row < 0 or max_col < 0:
            continue

        for row in range(max_row + 1):
            for col in range(max_col + 1):
                left = int(col * width / grid_size)
                top = int(row * height / grid_size)
                right = int((col + col_span) * width / grid_size)
                bottom = int((row + row_span) * height / grid_size)
                box = (left, top, right, bottom)

                if box in seen_boxes or right <= left or bottom <= top:
                    continue

                seen_boxes.add(box)
                entries.append(
                    {
                        "box": box,
                        "image": image.crop(box),
                    }
                )

    return entries


def get_localization_probabilities(
    images: list[Image.Image],
    use_zero_shot: bool,
):
    custom_probabilities = compute_custom_probabilities_batch(images)
    zero_shot_probabilities = [None] * len(images)
    if use_zero_shot:
        zero_shot_probabilities, _ = compute_zero_shot_probabilities_batch(images)

    fused_probabilities = []
    for custom_probs, zero_probs in zip(custom_probabilities, zero_shot_probabilities):
        if zero_probs is None:
            fused_probabilities.append(custom_probs)
        else:
            fused_probs, _ = fuse_probability_vectors(custom_probs, zero_probs, prefer_custom=True)
            fused_probabilities.append(fused_probs)

    return fused_probabilities


def box_iou(first_box, second_box) -> float:
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    first_area = max(1, (first_box[2] - first_box[0]) * (first_box[3] - first_box[1]))
    second_area = max(1, (second_box[2] - second_box[0]) * (second_box[3] - second_box[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def box_containment(first_box, second_box) -> float:
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    first_area = max(1, (first_box[2] - first_box[0]) * (first_box[3] - first_box[1]))
    second_area = max(1, (second_box[2] - second_box[0]) * (second_box[3] - second_box[1]))
    return max(intersection / first_area, intersection / second_area)


def suppress_overlapping_candidates(candidates: list[dict], limit: int = MAX_DISPLAY_BOXES):
    ranked_candidates = sorted(candidates, key=lambda item: item["ranking_score"], reverse=True)
    selected = []

    for candidate in ranked_candidates:
        if any(
            box_iou(candidate["box"], kept["box"]) >= BOX_IOU_THRESHOLD
            or box_containment(candidate["box"], kept["box"]) >= BOX_CONTAINMENT_THRESHOLD
            for kept in selected
        ):
            continue

        cleaned_candidate = dict(candidate)
        cleaned_candidate.pop("ranking_score", None)
        cleaned_candidate.pop("top_label", None)
        selected.append(cleaned_candidate)

        if len(selected) >= limit:
            break

    if not selected and ranked_candidates:
        fallback = dict(ranked_candidates[0])
        fallback.pop("ranking_score", None)
        fallback.pop("top_label", None)
        selected.append(fallback)

    return selected


def get_region_box_candidates(image: Image.Image, target_label: str, diagnostics: dict[str, float | str] | None = None):
    target_class_id = get_class_id_map()[target_label.lower()]
    region_entries = generate_region_scan_entries(image)
    if not region_entries:
        return []

    zero_shot_confidence = float(diagnostics.get("zero_shot_confidence", 0.0)) if diagnostics else 0.0
    use_zero_shot = bool(
        diagnostics
        and diagnostics.get("fusion_mode") == "Hybrid"
        and diagnostics.get("zero_shot_label") == target_label
        and zero_shot_confidence >= 70
    )
    fused_probabilities = get_localization_probabilities(
        [entry["image"] for entry in region_entries],
        use_zero_shot=use_zero_shot,
    )

    image_area = image.width * image.height
    raw_candidates = []
    max_target_score = 0.0
    for entry, fused_probs in zip(region_entries, fused_probabilities):
        target_score = float(fused_probs[target_class_id].item())
        top_class_id = int(torch.argmax(fused_probs).item())
        top_label = get_id_class_map()[top_class_id]
        area_ratio = ((entry["box"][2] - entry["box"][0]) * (entry["box"][3] - entry["box"][1])) / image_area
        max_target_score = max(max_target_score, target_score)
        ranking_score = target_score - area_ratio * 0.16
        if top_label == target_label.lower():
            ranking_score += 0.04

        raw_candidates.append(
            {
                "box": entry["box"],
                "confidence": round(target_score * 100, 2),
                "area_ratio": area_ratio,
                "source": "Region Scan",
                "top_label": top_label,
                "ranking_score": ranking_score,
            }
        )

    selection_threshold = max(REGION_MIN_SCORE, max_target_score * REGION_SCORE_RATIO)
    selected_candidates = [
        candidate
        for candidate in raw_candidates
        if (candidate["confidence"] / 100) >= selection_threshold
        and (
            candidate["top_label"] == target_label.lower()
            or (candidate["confidence"] / 100) >= max(selection_threshold + 0.08, 0.42)
        )
    ]

    if not selected_candidates and raw_candidates:
        selected_candidates = sorted(raw_candidates, key=lambda item: item["ranking_score"], reverse=True)[:1]

    return suppress_overlapping_candidates(selected_candidates)


def get_detector_box_candidates(image: Image.Image, target_label: str):
    target_class_id = get_class_id_map()[target_label.lower()]
    detection_result = get_detector_model()(image, verbose=False)[0]
    if detection_result.boxes is None or len(detection_result.boxes) == 0:
        return []

    raw_candidates = []
    image_area = image.width * image.height
    for box in detection_result.boxes:
        detector_confidence = float(box.conf.item())
        if detector_confidence < DETECTOR_CONF_THRESHOLD:
            continue

        left, top, right, bottom = [int(value) for value in box.xyxy[0].tolist()]
        if right <= left or bottom <= top:
            continue

        area_ratio = ((right - left) * (bottom - top)) / image_area
        if area_ratio < DETECTOR_MIN_AREA_RATIO:
            continue

        raw_candidates.append(
            {
                "box": (left, top, right, bottom),
                "crop": image.crop((left, top, right, bottom)),
                "area_ratio": area_ratio,
                "detector_confidence": detector_confidence,
            }
        )

    if not raw_candidates:
        return []

    custom_probabilities = compute_custom_probabilities_batch([candidate["crop"] for candidate in raw_candidates])
    zero_shot_probabilities, _ = compute_zero_shot_probabilities_batch([candidate["crop"] for candidate in raw_candidates])

    candidates = []
    for candidate, custom_probs, zero_probs in zip(raw_candidates, custom_probabilities, zero_shot_probabilities):
        fused_probabilities, _ = fuse_probability_vectors(custom_probs, zero_probs)
        target_score = float(fused_probabilities[target_class_id].item())
        top_class_id = int(torch.argmax(fused_probabilities).item())
        top_label = get_id_class_map()[top_class_id]

        if top_label != target_label.lower() and target_score < 0.45:
            continue

        candidates.append(
            {
                "box": candidate["box"],
                "confidence": round(max(target_score, float(fused_probabilities[top_class_id].item())) * 100, 2),
                "area_ratio": candidate["area_ratio"],
                "source": "Detector",
                "top_label": top_label,
                "ranking_score": max(target_score, float(fused_probabilities[top_class_id].item()))
                + 0.08
                - candidate["area_ratio"] * 0.08,
            }
        )

    return sorted(candidates, key=lambda item: item["ranking_score"], reverse=True)


def select_bounding_boxes(input_path: Path, target_label: str, diagnostics: dict[str, float | str] | None = None):
    with Image.open(input_path).convert("RGB") as image:
        detector_candidates = get_detector_box_candidates(image, target_label)
        region_candidates = get_region_box_candidates(image, target_label, diagnostics)

    combined_candidates = detector_candidates + region_candidates
    if not combined_candidates:
        return []

    for candidate in combined_candidates:
        candidate.setdefault(
            "ranking_score",
            (candidate["confidence"] / 100)
            - candidate.get("area_ratio", 0.0) * 0.16
            + (0.06 if candidate.get("source") == "Detector" else 0.0),
        )

    selected_candidates = suppress_overlapping_candidates(combined_candidates)
    if not selected_candidates:
        return []

    return selected_candidates[:MAX_DISPLAY_BOXES]


def create_result_image(
    input_path: Path,
    output_path: Path,
    top_prediction: dict[str, float | str],
    diagnostics: dict[str, float | str],
    box_candidates: list[dict],
):
    with Image.open(input_path).convert("RGB") as image:
        width, height = image.size
        draw = ImageDraw.Draw(image, "RGBA")

        box_font = load_font(max(16, width // 42), bold=True)
        line_width = max(3, min(width, height) // 130)
        for candidate in box_candidates:
            left, top, right, bottom = candidate["box"]
            draw.rounded_rectangle((left, top, right, bottom), radius=18, outline=BOX_COLOR, width=line_width)

            label_text = f'{top_prediction["label"]} {float(candidate["confidence"]):.0f}%'
            label_box = draw.textbbox((0, 0), label_text, font=box_font)
            label_width = (label_box[2] - label_box[0]) + 18
            label_height = (label_box[3] - label_box[1]) + 12
            label_left = left
            label_top = top - label_height - 8 if top > label_height + 12 else top + 8

            draw.rounded_rectangle(
                (label_left, label_top, label_left + label_width, label_top + label_height),
                radius=12,
                fill=BOX_FILL,
            )
            draw.text((label_left + 9, label_top + 6), label_text, font=box_font, fill=(255, 255, 255, 255))

        image.save(output_path)


def build_context(**overrides):
    class_count, image_count = get_dataset_overview()
    context = {
        "error": None,
        "original_image": None,
        "result_image": None,
        "predictions": [],
        "top_prediction": None,
        "model_ready": IS_CUSTOM_MODEL,
        "class_count": class_count,
        "image_count": image_count,
        "analysis_note": None,
        "diagnostics": None,
        "box_count": 0,
    }
    context.update(overrides)
    return context


def render_classification(input_path: Path, original_image_url: str, result_filename: str):
    result_path = RESULTS_DIR / result_filename

    try:
        predictions, top_prediction, analysis_note, diagnostics = analyze_image(input_path)
        if not predictions:
            raise ValueError("The model did not return any classification scores.")
        box_candidates = select_bounding_boxes(input_path, str(top_prediction["label"]), diagnostics)
        if box_candidates:
            diagnostics["box_source"] = " + ".join(sorted({candidate["source"] for candidate in box_candidates}))
        create_result_image(input_path, result_path, top_prediction, diagnostics, box_candidates)
    except Exception as exc:
        return render_template(
            "index.html",
            **build_context(error=f"Prediction failed: {exc}"),
        )

    return render_template(
        "index.html",
        **build_context(
            original_image=original_image_url,
            result_image=url_for("static", filename=f"results/{result_filename}"),
            predictions=predictions,
            top_prediction=top_prediction,
            analysis_note=analysis_note,
            diagnostics=diagnostics,
            box_count=len(box_candidates),
        ),
    )


@app.route("/")
def home():
    return render_template("index.html", **build_context())


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return render_template(
            "index.html",
            **build_context(error="Please choose an image before running prediction."),
        )

    safe_name = secure_filename(image_file.filename)
    extension = Path(safe_name).suffix.lower() or ".jpg"
    stored_name = f"{uuid4().hex}{extension}"
    upload_path = UPLOAD_DIR / stored_name
    image_file.save(upload_path)

    return render_classification(
        input_path=upload_path,
        original_image_url=url_for("uploaded_file", filename=stored_name),
        result_filename=stored_name,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
