import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".bmp", ".tif", ".tiff"]


def list_images(image_dir: Path):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def find_matching_mask(mask_dir: Path, image_path: Path):
    for ext in MASK_EXTS:
        p = mask_dir / f"{image_path.stem}{ext}"
        if p.exists():
            return p
    return None


def clip_xy(x, y, image_w, image_h, patch_size):
    max_x = max(0, image_w - patch_size)
    max_y = max(0, image_h - patch_size)
    x = max(0, min(int(round(x)), max_x))
    y = max(0, min(int(round(y)), max_y))
    return x, y


def unique_positions(start, end, step):
    if start > end:
        start, end = end, start

    vals = set()
    cur = start
    while cur <= end:
        vals.add(int(round(cur)))
        cur += step

    vals.add(int(round(start)))
    vals.add(int(round(end)))
    vals.add(int(round((start + end) / 2)))
    return sorted(vals)


def get_components(mask: np.ndarray, target_classes, min_component_area: int):
    """
    对 ID2/ID4 分别做连通域分析。
    返回每个杆塔连通域的 bbox、面积、像素坐标、类别。
    """
    components = []

    for cls in target_classes:
        binary = (mask == cls).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_component_area:
                continue

            ys, xs = np.where(labels == i)
            cx, cy = centroids[i]

            components.append(
                {
                    "class_id": int(cls),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "area": int(area),
                    "centroid": (float(cx), float(cy)),
                    "xs": xs.astype(np.int32),
                    "ys": ys.astype(np.int32),
                }
            )

    return components


def generate_candidate_windows(component, image_w, image_h, patch_size, stride, context_margin):
    """
    只在杆塔附近生成候选 512×512 patch。

    包含三类候选：
    1. 尽量完整覆盖 bbox 的 crop；
    2. bbox center / centroid crop；
    3. bbox 周围扩展 context_margin 后的局部滑窗。
    """
    x, y, bw, bh = component["bbox"]
    cx, cy = component["centroid"]

    windows = set()

    # 1. 如果 bbox 尺寸小于 patch，生成“完整覆盖 bbox”的候选窗口
    if bw <= patch_size and bh <= patch_size:
        full_x_min = x + bw - patch_size
        full_x_max = x
        full_y_min = y + bh - patch_size
        full_y_max = y

        full_x_min, _ = clip_xy(full_x_min, 0, image_w, image_h, patch_size)
        full_x_max, _ = clip_xy(full_x_max, 0, image_w, image_h, patch_size)
        _, full_y_min = clip_xy(0, full_y_min, image_w, image_h, patch_size)
        _, full_y_max = clip_xy(0, full_y_max, image_w, image_h, patch_size)

        xs = unique_positions(full_x_min, full_x_max, stride)
        ys = unique_positions(full_y_min, full_y_max, stride)

        for px in xs:
            for py in ys:
                windows.add(clip_xy(px, py, image_w, image_h, patch_size))

    # 2. bbox center crop
    bbox_cx = x + bw / 2.0
    bbox_cy = y + bh / 2.0
    windows.add(
        clip_xy(
            bbox_cx - patch_size / 2.0,
            bbox_cy - patch_size / 2.0,
            image_w,
            image_h,
            patch_size,
        )
    )

    # 3. centroid crop
    windows.add(
        clip_xy(
            cx - patch_size / 2.0,
            cy - patch_size / 2.0,
            image_w,
            image_h,
            patch_size,
        )
    )

    # 4. 杆塔 bbox 附近局部滑窗，不做全图滑窗
    local_x1 = max(0, x - context_margin)
    local_y1 = max(0, y - context_margin)
    local_x2 = min(image_w, x + bw + context_margin)
    local_y2 = min(image_h, y + bh + context_margin)

    # 用局部区域中心点生成 patch 左上角
    center_xs = unique_positions(local_x1, local_x2, stride)
    center_ys = unique_positions(local_y1, local_y2, stride)

    for cxx in center_xs:
        for cyy in center_ys:
            px = cxx - patch_size / 2.0
            py = cyy - patch_size / 2.0
            windows.add(clip_xy(px, py, image_w, image_h, patch_size))

    return windows


def component_coverage(component, x1, y1, patch_size):
    """
    计算当前 patch 覆盖了该杆塔连通域多少像素。
    coverage 越接近 1，说明越完整包含该杆塔。
    """
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    xs = component["xs"]
    ys = component["ys"]

    inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    covered = int(inside.sum())

    if component["area"] <= 0:
        return 0.0

    return covered / float(component["area"])


def target_center_is_reasonable(mask_patch, target_classes, center_ratio):
    """
    判断 ID2/ID4 的像素重心是否处于 patch 中央区域。
    避免目标贴边。
    """
    target = np.isin(mask_patch, target_classes)
    ys, xs = np.where(target)

    if len(xs) == 0:
        return False

    cx = float(xs.mean())
    cy = float(ys.mean())

    h, w = mask_patch.shape[:2]

    margin_x = (1.0 - center_ratio) * w / 2.0
    margin_y = (1.0 - center_ratio) * h / 2.0

    return (
        margin_x <= cx <= w - margin_x
        and margin_y <= cy <= h - margin_y
    )


def check_mask_values(mask_patch, allowed_values):
    vals = set(np.unique(mask_patch).tolist())
    invalid = vals - set(allowed_values)
    return invalid


def light_augment_pil(img: Image.Image):
    """
    一次温和光照增强：
    - brightness
    - contrast
    - saturation
    - gamma

    只处理 image，不处理 mask。
    """
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.30))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.30))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.20))

    if random.random() < 0.7:
        gamma = random.uniform(0.70, 1.50)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def make_list_line(image_name, mask_name, image_prefix, mask_prefix):
    image_item = f"{image_prefix.rstrip('/')}/{image_name}"
    mask_item = f"{mask_prefix.rstrip('/')}/{mask_name}"
    return f"{image_item} {mask_item}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", required=True, help="母图 images/train 文件夹")
    parser.add_argument("--mask-dir", required=True, help="母图 masks/train 文件夹")

    parser.add_argument("--out-image-dir", required=True, help="新增 image 输出文件夹")
    parser.add_argument("--out-mask-dir", required=True, help="新增 mask 输出文件夹")
    parser.add_argument("--out-list", required=True, help="新增样本 txt 文件")

    parser.add_argument("--image-list-prefix", default="images/train")
    parser.add_argument("--mask-list-prefix", default="masks/train")

    parser.add_argument("--target-classes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--allowed-mask-values", nargs="+", type=int, default=[0, 1, 2, 3, 4, 255])

    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--context-margin", type=int, default=256)

    parser.add_argument("--min-component-area", type=int, default=50)
    parser.add_argument("--min-target-pixels", type=int, default=100)
    parser.add_argument("--min-component-coverage", type=float, default=0.85)
    parser.add_argument("--center-ratio", type=float, default=0.75)

    parser.add_argument("--max-patches-per-image", type=int, default=20)

    parser.add_argument("--save-original-crop", action="store_true")
    parser.add_argument("--light-aug-times", type=int, default=1)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_image_dir = Path(args.out_image_dir)
    out_mask_dir = Path(args.out_mask_dir)
    out_list = Path(args.out_list)

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_list.parent.mkdir(parents=True, exist_ok=True)

    image_files = list_images(image_dir)

    new_lines = []

    total_images = 0
    matched_pairs = 0
    target_images = 0
    saved_pairs = 0
    skipped_invalid = 0
    skipped_existing = 0

    for image_path in image_files:
        total_images += 1

        mask_path = find_matching_mask(mask_dir, image_path)
        if mask_path is None:
            print(f"[WARN] mask not found: {image_path.name}")
            continue

        matched_pairs += 1

        img = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        image_w, image_h = img.size

        if image_w < args.patch_size or image_h < args.patch_size:
            print(f"[WARN] skip small image: {image_path.name}, size={image_w}x{image_h}")
            continue

        if not np.isin(mask, args.target_classes).any():
            continue

        target_images += 1

        components = get_components(
            mask=mask,
            target_classes=args.target_classes,
            min_component_area=args.min_component_area,
        )

        if len(components) == 0:
            continue

        scored = {}

        for comp in components:
            windows = generate_candidate_windows(
                component=comp,
                image_w=image_w,
                image_h=image_h,
                patch_size=args.patch_size,
                stride=args.stride,
                context_margin=args.context_margin,
            )

            x, y, bw, bh = comp["bbox"]
            bbox_fits_patch = (bw <= args.patch_size and bh <= args.patch_size)

            for x1, y1 in windows:
                x2 = x1 + args.patch_size
                y2 = y1 + args.patch_size

                mask_patch = mask[y1:y2, x1:x2]
                target_pixels = int(np.isin(mask_patch, args.target_classes).sum())

                if target_pixels < args.min_target_pixels:
                    continue

                coverage = component_coverage(comp, x1, y1, args.patch_size)

                # 如果该杆塔理论上能被 512 patch 完整覆盖，则优先要求 coverage 足够高
                # 如果该杆塔本身超过 512，则不强制 coverage=0.85，只保留覆盖较多的候选
                if bbox_fits_patch and coverage < args.min_component_coverage:
                    continue

                if not target_center_is_reasonable(
                    mask_patch=mask_patch,
                    target_classes=args.target_classes,
                    center_ratio=args.center_ratio,
                ):
                    continue

                invalid = check_mask_values(mask_patch, args.allowed_mask_values)
                if invalid:
                    skipped_invalid += 1
                    print(
                        f"[WARN] invalid mask values {invalid} in {mask_path.name}, "
                        f"x={x1}, y={y1}"
                    )
                    continue

                key = (x1, y1)
                old = scored.get(key)

                # score: coverage 优先，其次 target_pixels
                score_tuple = (coverage, target_pixels, comp["class_id"])

                if old is None or score_tuple > old["score_tuple"]:
                    scored[key] = {
                        "x1": x1,
                        "y1": y1,
                        "coverage": coverage,
                        "target_pixels": target_pixels,
                        "major_cls": comp["class_id"],
                        "score_tuple": score_tuple,
                    }

        candidates = list(scored.values())
        candidates.sort(
            key=lambda d: (d["coverage"], d["target_pixels"]),
            reverse=True,
        )
        candidates = candidates[: args.max_patches_per_image]

        for item in candidates:
            x1 = item["x1"]
            y1 = item["y1"]
            x2 = x1 + args.patch_size
            y2 = y1 + args.patch_size

            img_patch = img.crop((x1, y1, x2, y2))
            mask_patch = mask[y1:y2, x1:x2].astype(np.uint8)

            target_pixels = item["target_pixels"]
            coverage_int = int(round(item["coverage"] * 100))
            major_cls = item["major_cls"]

            base = (
                f"{image_path.stem}_id{major_cls}_"
                f"x{x1:04d}_y{y1:04d}_"
                f"cov{coverage_int:03d}_pix{target_pixels:06d}"
            )

            if args.save_original_crop:
                image_name = f"{base}_orig.png"
                mask_name = f"{base}_orig.png"

                out_img = out_image_dir / image_name
                out_mask = out_mask_dir / mask_name

                if (out_img.exists() or out_mask.exists()) and not args.overwrite:
                    skipped_existing += 1
                else:
                    img_patch.save(out_img)
                    Image.fromarray(mask_patch).save(out_mask)

                    new_lines.append(
                        make_list_line(
                            image_name=image_name,
                            mask_name=mask_name,
                            image_prefix=args.image_list_prefix,
                            mask_prefix=args.mask_list_prefix,
                        )
                    )
                    saved_pairs += 1

            for aug_idx in range(args.light_aug_times):
                image_name = f"{base}_light{aug_idx + 1}.png"
                mask_name = f"{base}_light{aug_idx + 1}.png"

                out_img = out_image_dir / image_name
                out_mask = out_mask_dir / mask_name

                if (out_img.exists() or out_mask.exists()) and not args.overwrite:
                    skipped_existing += 1
                    continue

                aug_img = light_augment_pil(img_patch)
                aug_img.save(out_img)
                Image.fromarray(mask_patch).save(out_mask)

                new_lines.append(
                    make_list_line(
                        image_name=image_name,
                        mask_name=mask_name,
                        image_prefix=args.image_list_prefix,
                        mask_prefix=args.mask_list_prefix,
                    )
                )
                saved_pairs += 1

    with open(out_list, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    print("Done.")
    print(f"Total images scanned: {total_images}")
    print(f"Matched image-mask pairs: {matched_pairs}")
    print(f"Images containing ID2/ID4: {target_images}")
    print(f"Saved image-mask pairs: {saved_pairs}")
    print(f"Skipped invalid mask crops: {skipped_invalid}")
    print(f"Skipped existing files: {skipped_existing}")
    print(f"New list written to: {out_list}")


if __name__ == "__main__":
    main()