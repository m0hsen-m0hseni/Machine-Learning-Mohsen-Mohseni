import os
import json
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask


IMAGE_DIR = "images"
RESULTS_DIR = "results"
JSON_PATH = "imagenet_class_index.json"

MODEL_NAME = "resnet50"
DEVICE = torch.device("cpu")

TARGET_CLASSES = {
    "cat": [
        "tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat"
    ],
    "dog": [
        "golden retriever", "Labrador retriever", "German shepherd", "beagle", "pug", "Samoyed", "Eskimo dog", "husky"
    ],
    "car": [
        "sport car", "convertible", "jeep", "limousine", "Model T", "racer", "car wheel", "minivan"
    ]
}


EXPERIMENTS = [
    {
        "class_group": "cat",
        "image_name": "cat_positive.jpg",
        "kind": "positive"
    },
    {
        "class_group": "cat",
        "image_name": "cat_negative.jpg",
        "kind": "negative"
    },
    {
        "class_group": "dog",
        "image_name": "dog_positive.jpg",
        "kind": "positive"
    },
    {
        "class_group": "dog",
        "image_name": "dog_negative.jpg",
        "kind": "negative"
    },
    {
        "class_group": "car",
        "image_name": "car_positive.jpg",
        "kind": "positive"
    },
    {
        "class_group": "car",
        "image_name": "car_negative.jpg",
        "kind": "negative"
    },
    {
        "class_group": "unknown",
        "image_name": "unknown_object.jpg",
        "kind": "unknown"
    }
]

VG_TARGET_LAYERS = [
    "layer1",
    "layer2",
    "layer3",
    "layer4"
]


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_imagenet_index(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r") as f:
        return json.load(f)


def build_model(model_name: str = "resnet50") -> torch.nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    model.to(DEVICE)
    return model


def get_preprocess():
    weights = models.ResNet50_Weights.DEFAULT
    return weights.transforms()


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def preprocess_image(img: Image.Image, preprocess) -> torch.Tensor:
    tensor = preprocess(img).unsqueeze(0)
    return tensor.to(DEVICE)


def predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: Dict[str, List[str]],
    topk: int = 5
) -> Tuple[torch.Tensor, List[Tuple[int, str, float]]]:

    with torch.no_grad():
        output = model(input_tensor)

    probs = F.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probs, topk)

    results = []
    for i in range(topk):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        label = class_idx[str(idx)][1]
        results.append((idx, label, prob))

    return output, results


def find_target_class_index(
    class_idx: Dict[str, List[str]],
    candidate_names: List[str]
) -> int:
    """
    Find one ImageNet class index matching any of the given candidate name.
    """
    lowercase_candidates = [x.lower() for x in candidate_names]

    for idx_str, value in class_idx.items():
        human_label = value[1].lower()
        for c in lowercase_candidates:
            if c == human_label:
                return int(idx_str)

    # fallback: partial match
    for idx_str, value in class_idx.items():
        human_label = value[1].lower()
        for c in lowercase_candidates:
            if c in human_label:
                return int(idx_str)

    raise ValueError(
        f"Could not find class index for candidates: {candidate_names}")


def save_top5_text(
    filepath: str,
    top5: List[Tuple[int, str, float]],
    image_name: str,
    group_name: str
) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Target group: {group_name}\n")
        f.write("Top-5 predictions:\n\n")

        for rank, (idx, label, prob) in enumerate(top5, start=1):
            f.write(
                f"{rank}. class_id={idx}, label={label}, probability={prob:.4f}\n")


def generate_cam_for_target(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_img: Image.Image,
    target_class_idx: int,
    target_layer: str = "layer4"
):
    with LayerCAM(model, target_layer=target_layer) as cam_extractor:
        scores = model(input_tensor)
        activation_map = cam_extractor(target_class_idx, scores)[0]

    activation_map = activation_map.detach().cpu()
    cam_tensor = activation_map.squeeze()
    heatmap_pil = transforms.functional.to_pil_image(cam_tensor, mode="F")
    overlay = overlay_mask(original_img, heatmap_pil,
                           alpha=0.5, colormap="jet")

    return scores, activation_map, overlay


def save_cam_figure(
    original_img: Image.Image,
    overlay_img: Image.Image,
    save_path: str,
    title: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_multilayer_cam_figure(
    original_img: Image.Image,
    overlays: List[Tuple[str, Image.Image]],
    save_path: str,
    main_title: str
) -> None:
    n = len(overlays) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (layer_name, overlay_img) in enumerate(overlays, start=1):
        axes[i].imshow(overlay_img)
        axes[i].set_title(layer_name)
        axes[i].axis("off")

    fig.suptitle(main_title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def class_group_hit(top5: List[Tuple[int, str, float]], target_group: str) -> bool:
    if target_group == "unknown":
        return False

    acceptable = [x.lower() for x in TARGET_CLASSES[target_group]]
    predicted_labels = [label.lower() for (_, label, _) in top5]

    for pred in predicted_labels:
        for acc in acceptable:
            if acc in pred or pred in acc:
                return True

    return False


def write_summary_report(report_path: str, summary_rows: List[Dict]) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Lab 2 - CNN Interpretability Summary\n")
        f.write("=" * 50 + "\n\n")

        for row in summary_rows:
            f.write(f"Image: {row['image_name']}\n")
            f.write(f"Target class group: {row['class_group']}\n")
            f.write(f"Type: {row['kind']}\n")
            f.write(
                f"Top-1 prediction: {row['top1_label']} ({row['top1_prob']:.4f})\n")
            f.write(f"Group found in top-5: {row['group_found']}\n")
            f.write(f"CAM saved: {row['cam_path']}\n")
            if row.get("multilayer_path"):
                f.write(f"Multi-layer CAM saved: {row['multilayer_path']}\n")
            f.write("\nTop-5:\n")
            for rank, (idx, label, prob) in enumerate(row["top5"], start=1):
                f.write(f" {rank}, {label} ({prob:.4f}) [class_id={idx}]\n")
            f.write("\n" + "-" * 50 + "\n\n")


def main():
    ensure_dirs()

    print(f"Using devise: {DEVICE}")
    print("Loading ImageNet class index...")
    class_idx = load_imagenet_index(JSON_PATH)

    print("Loading model...")
    model = build_model(MODEL_NAME)
    model.eval()
    preprocess = get_preprocess()

    summary_row = []

    # choose one positive image for VG multi-layer comparison
    vg_multilayer_done = False

    for exp in EXPERIMENTS:
        class_group = exp["class_group"]
        image_name = exp["image_name"]
        kind = exp["kind"]

        image_path = os.path.join(IMAGE_DIR, image_name)
        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            continue

        print(f"\nProcessing: {image_name}")
        img = load_image(image_path)
        input_tensor = preprocess_image(img, preprocess)
        with torch.no_grad():
            output_for_pred = model(input_tensor)
        probs = F.softmax(output_for_pred, dim=1)
        top_probs, top_indices = torch.topk(probs, 5)

        top5 = []
        for i in range(5):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            label = class_idx[str(idx)][1]
            top5.append((idx, label, prob))

        top1_idx, top1_label, top1_prob = top5[0]

        target_class_idx = top1_idx
        cam_title = f"LayerCAM: {top1_label} ({top1_prob:.2%})"

        # generate CAM from default last conv block
        _, _, overlay = generate_cam_for_target(
            model=model,
            input_tensor=input_tensor,
            original_img=img,
            target_class_idx=target_class_idx,
            target_layer="layer4"
        )

        cam_filename = f"{os.path.splitext(image_name)[0]}_cam_layer4.png"
        cam_path = os.path.join(RESULTS_DIR, cam_filename)
        save_cam_figure(
            original_img=img,
            overlay_img=overlay,
            save_path=cam_path,
            title=cam_title
        )

        # save top5 text
        txt_filename = f"{os.path.splitext(image_name)[0]}_top5.txt"
        txt_path = os.path.join(RESULTS_DIR, txt_filename)
        save_top5_text(
            filepath=txt_path,
            top5=top5,
            image_name=image_name,
            group_name=class_group
        )

        multilayer_path = None

        # VG: compare multiple layer for at least two positive examples
        if kind == "positive":
            overlays = []

            for layer_name in VG_TARGET_LAYERS:
                _, _, layer_overlay = generate_cam_for_target(
                    model=model,
                    input_tensor=input_tensor,
                    original_img=img,
                    target_class_idx=target_class_idx,
                    target_layer=layer_name
                )
                overlays.append((layer_name, layer_overlay))

            multilayer_filename = f"{os.path.splitext(image_name)[0]}_multilayer_cam.png"
            multilayer_path = os.path.join(RESULTS_DIR, multilayer_filename)

            save_multilayer_cam_figure(
                original_img=img,
                overlays=overlays,
                save_path=multilayer_path,
                main_title=f"Multi-layer Grad-CAM comparison for {image_name}"
            )

            vg_multilayer_done = True

        summary_row.append({
            "image_name": image_name,
            "class_group": class_group,
            "kind": kind,
            "top1_label": top1_label,
            "top1_prob": top1_prob,
            "group_found": class_group_hit(top5, class_group),
            "cam_path": cam_path,
            "multilayer_path": multilayer_path,
            "top5": top5
        })

    # save summary report
    report_path = os.path.join(RESULTS_DIR, "summary_report.txt")
    write_summary_report(report_path, summary_row)

    print("\nDone.")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
