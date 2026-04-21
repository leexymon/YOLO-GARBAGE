from pathlib import Path
import random
import shutil

from PIL import Image, ImageOps
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATASET = BASE_DIR / "TrashType_Image_Dataset"
OUTPUT_DATASET = BASE_DIR / "trash_cls_dataset"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
VAL_RATIO = 0.2
PILE_AUG_RATIO = 0.35
PILE_IMAGE_SIZE = 640
BASE_MODEL = BASE_DIR / "runs" / "classify" / "trash_cls" / "weights" / "best.pt"
TRAIN_RUN_NAME = "trash_cls_v2"
IMAGE_SIZE = 320
EPOCHS = 12


def iter_classes():
    return sorted(path for path in SOURCE_DATASET.iterdir() if path.is_dir())


def iter_images(class_dir: Path):
    return [path for path in sorted(class_dir.iterdir()) if path.suffix.lower() in ALLOWED_EXTENSIONS]


def build_same_class_pile(image_paths: list[Path], output_path: Path):
    canvas = Image.new("RGB", (PILE_IMAGE_SIZE, PILE_IMAGE_SIZE), color=(245, 245, 245))
    tile_size = PILE_IMAGE_SIZE // 2

    for index, image_path in enumerate(image_paths):
        row = index // 2
        col = index % 2
        with Image.open(image_path).convert("RGB") as image:
            fitted = ImageOps.fit(
                image,
                (tile_size, tile_size),
                method=Image.Resampling.LANCZOS,
                bleed=0.02,
            )
            canvas.paste(fitted, (col * tile_size, row * tile_size))

    canvas.save(output_path, quality=95)


def create_pile_augmentations(class_name: str, train_images: list[Path], target_dir: Path):
    synthetic_count = max(12, round(len(train_images) * PILE_AUG_RATIO))
    if len(train_images) < 4:
        return 0

    for index in range(synthetic_count):
        sampled_images = random.choices(train_images, k=4)
        output_path = target_dir / f"{class_name}_pile_{index + 1:03d}.jpg"
        build_same_class_pile(sampled_images, output_path)

    return synthetic_count


def prepare_dataset():
    random.seed(SEED)

    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)

    for split in ("train", "val"):
        (OUTPUT_DATASET / split).mkdir(parents=True, exist_ok=True)

    summary = []
    for class_dir in iter_classes():
        images = iter_images(class_dir)
        random.shuffle(images)

        val_count = max(1, round(len(images) * VAL_RATIO))
        val_images = images[:val_count]
        train_images = images[val_count:]

        for split_name, split_images in (("train", train_images), ("val", val_images)):
            target_dir = OUTPUT_DATASET / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, target_dir / image_path.name)

        synthetic_count = create_pile_augmentations(class_dir.name, train_images, OUTPUT_DATASET / "train" / class_dir.name)
        summary.append((class_dir.name, len(train_images), len(val_images), synthetic_count))

    print("Prepared stratified dataset split:")
    for class_name, train_count, val_count, synthetic_count in summary:
        print(f"  {class_name}: train={train_count}, val={val_count}, pile_aug={synthetic_count}")


def train_model():
    base_model = str(BASE_MODEL) if BASE_MODEL.exists() else "yolov8n-cls.pt"
    model = YOLO(base_model)
    model.train(
        data=str(OUTPUT_DATASET),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=32,
        workers=0,
        device="cpu",
        patience=5,
        project=str(BASE_DIR / "runs" / "classify"),
        name=TRAIN_RUN_NAME,
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    prepare_dataset()
    train_model()
