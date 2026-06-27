import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.pidnet import PIDNet


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PIDNet inference on val patches from list/val.txt or on a folder."
    )
    parser.add_argument(
        "--weight-path",
        default="output/custom/pidnet_small_local_5_lse_v1/best.pt",
        help="Checkpoint path.",
    )
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument(
        "--input-mode",
        choices=["val_list", "folder"],
        default="val_list",
        help="Use val list for exact validation-set inference, or scan a folder.",
    )
    parser.add_argument(
        "--root",
        default="data/PIDNet_Power_Dataset",
        help="Dataset root used with --input-mode val_list.",
    )
    parser.add_argument(
        "--list-path",
        default="list/val.txt",
        help="Relative val list path used with --input-mode val_list.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/PIDNet_Power_Dataset/images/val",
        help="Image folder used with --input-mode folder.",
    )
    parser.add_argument(
        "--output-mask-dir",
        default="output/inference/current_val512_masks",
        help="Directory for predicted single-channel masks.",
    )
    parser.add_argument(
        "--output-vis-dir",
        default="output/inference/current_val512_vis",
        help="Directory for overlay visualizations.",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--save-512",
        action="store_true",
        help="Save predicted masks at 512x512 to match training validation protocol.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_final_output(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[1]
    return outputs


def build_model(num_classes, checkpoint_path, device):
    model = PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_state = model.state_dict()
    usable = {}
    for key, value in checkpoint.items():
        clean_key = key.replace("module.", "")
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            usable[clean_key] = value

    missing, unexpected = model.load_state_dict(usable, strict=False)
    model.to(device)
    model.eval()

    total_model_tensors = len(model_state)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Total model tensors:", total_model_tensors)
    print("Usable tensors:", len(usable))
    print("Missing tensors:", len(missing))
    print("Unexpected tensors:", len(unexpected))
    if len(usable) < 0.95 * total_model_tensors:
        raise RuntimeError(
            "Too few tensors loaded: {}/{}. Checkpoint and model architecture may not match.".format(
                len(usable), total_model_tensors
            )
        )
    return model


def read_val_items(root, list_path):
    items = []
    seen_names = set()
    with open(os.path.join(root, list_path), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_rel, label_rel = parts[0], parts[1]
            name = Path(label_rel).stem
            if name in seen_names:
                raise ValueError("Duplicate sample name in list: {}".format(name))
            seen_names.add(name)
            items.append(
                {
                    "name": name,
                    "image_path": os.path.join(root, image_rel),
                }
            )
    return items


def read_folder_items(input_dir):
    items = []
    seen_names = set()
    for path in sorted(Path(input_dir).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = path.stem
        if name in seen_names:
            raise ValueError("Duplicate file stem in folder: {}".format(name))
        seen_names.add(name)
        items.append({"name": name, "image_path": str(path)})
    return items


def preprocess_image(image_bgr):
    image = image_bgr.astype(np.float32) / 255.0
    image -= np.asarray(MEAN, dtype=np.float32)
    image /= np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def infer_mask(model, image_tensor, height, width, crop_size, stride, save_512=False):
    if save_512:
        image_resized = F.interpolate(
            image_tensor, size=(crop_size, crop_size), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            logits = select_final_output(model(image_resized))
        logits = F.interpolate(
            logits, size=(crop_size, crop_size), mode="bilinear", align_corners=True
        )
        return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    if height <= crop_size and width <= crop_size:
        image_resized = F.interpolate(
            image_tensor, size=(crop_size, crop_size), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            logits = select_final_output(model(image_resized))
        logits = F.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=True
        )
        return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    count_preds = torch.zeros((1, 1, height, width), device=image_tensor.device)
    preds = torch.zeros((1, model.final_layer.conv2.out_channels, height, width), device=image_tensor.device)

    y_positions = list(range(0, max(height - crop_size, 0) + 1, stride))
    x_positions = list(range(0, max(width - crop_size, 0) + 1, stride))
    if y_positions[-1] != height - crop_size:
        y_positions.append(height - crop_size)
    if x_positions[-1] != width - crop_size:
        x_positions.append(width - crop_size)

    for y in y_positions:
        for x in x_positions:
            crop = image_tensor[:, :, y : y + crop_size, x : x + crop_size]
            with torch.no_grad():
                logits = select_final_output(model(crop))
            logits = F.interpolate(
                logits, size=(crop_size, crop_size), mode="bilinear", align_corners=True
            )
            preds[:, :, y : y + crop_size, x : x + crop_size] += logits
            count_preds[:, :, y : y + crop_size, x : x + crop_size] += 1

    preds = preds / torch.clamp(count_preds, min=1.0)
    return torch.argmax(preds, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def build_vis(raw_image, pred_mask):
    color_map = np.zeros_like(raw_image)
    color_map[pred_mask == 1] = [0, 0, 255]
    color_map[pred_mask == 2] = [0, 255, 0]
    color_map[pred_mask == 3] = [255, 0, 0]
    color_map[pred_mask == 4] = [255, 255, 0]

    if (pred_mask > 0).any():
        blended = cv2.addWeighted(raw_image, 0.4, color_map, 0.6, 0)
        fg = (pred_mask > 0)[:, :, None]
        return np.where(fg, blended, raw_image)
    return raw_image.copy()


def main():
    args = parse_args()
    os.makedirs(args.output_mask_dir, exist_ok=True)
    os.makedirs(args.output_vis_dir, exist_ok=True)

    if args.input_mode == "val_list":
        items = read_val_items(args.root, args.list_path)
        print("Input mode: val_list")
        print("Dataset root:", args.root)
        print("List path:", args.list_path)
    else:
        items = read_folder_items(args.input_dir)
        print("Input mode: folder")
        print("Input dir:", args.input_dir)

    print("Samples to infer:", len(items))
    print("Output mask dir:", args.output_mask_dir)
    print("Output vis dir:", args.output_vis_dir)
    if args.save_512:
        print("Evaluation protocol: val512")
        print("Save mask size: {}x{}".format(args.crop_size, args.crop_size))
    else:
        print("Evaluation protocol: original-size")
        print("Save mask size: original image size")

    model = build_model(args.num_classes, args.weight_path, args.device)

    written = 0
    all_values = set()
    shape_counts = {}
    for item in tqdm(items, desc="infer"):
        raw_image = cv2.imread(item["image_path"], cv2.IMREAD_COLOR)
        if raw_image is None:
            print("Skip unreadable image:", item["image_path"])
            continue

        if args.save_512:
            infer_image = cv2.resize(
                raw_image, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR
            )
        else:
            infer_image = raw_image

        height, width = infer_image.shape[:2]
        image_tensor = preprocess_image(infer_image).to(args.device)
        pred_mask = infer_mask(
            model,
            image_tensor,
            height,
            width,
            args.crop_size,
            args.stride,
            save_512=args.save_512,
        )

        if args.save_512:
            vis_image = infer_image
        else:
            vis_image = raw_image

        mask_path = os.path.join(args.output_mask_dir, "{}.png".format(item["name"]))
        vis_path = os.path.join(args.output_vis_dir, "{}_vis.jpg".format(item["name"]))
        cv2.imwrite(mask_path, pred_mask)
        cv2.imwrite(vis_path, build_vis(vis_image, pred_mask))
        all_values.update(np.unique(pred_mask).astype(int).tolist())
        shape_counts[pred_mask.shape] = shape_counts.get(pred_mask.shape, 0) + 1
        written += 1

    print("Total samples:", len(items))
    print("Written masks:", written)
    print("Mask unique values:", sorted(all_values))
    print("Mask shape counts:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: (x[0][0], x[0][1])):
        print("  {}x{}: {}".format(shape[0], shape[1], count))
    print("Done.")


if __name__ == "__main__":
    main()

#python quanchicuntuili.py --weight-path output\custom\pidnet_small_local_5_v1_cw\best.pt --input-mode val_list --root data\PIDNet_Power_Dataset --list-path list\val.txt --output-mask-dir output\inference\v1_cw_val512_masks --output-vis-dir output\inference\v1_cw_val512_vis
