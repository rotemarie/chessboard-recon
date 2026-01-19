"""
Inference pipeline for chessboard reconstruction:
1) Detect board and warp to top-down view
2) Extract 64 square crops
3) Classify each crop with a trained model
4) Convert predictions to FEN
5) Render chess.com-style board from FEN with unknown squares marked
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

import chess
import chess.svg

try:
    from preprocessing.board_detector import BoardDetector
    from preprocessing.square_extractor import SquareExtractor, FENParser
except ImportError:
    from ..preprocessing.board_detector import BoardDetector
    from ..preprocessing.square_extractor import SquareExtractor, FENParser


CHESS_COM_COLORS = {
    "square light": "#f0d9b5",
    "square dark": "#b58863",
    "square light lastmove": "#f7ec75",
    "square dark lastmove": "#dac94a",
    "square light check": "#f7786b",
    "square dark check": "#d95c4d",
}


def load_class_names(class_dir: Optional[str], classes_file: Optional[str]) -> List[str]:
    if classes_file:
        path = Path(classes_file)
        if path.exists():
            return [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    if class_dir:
        path = Path(class_dir)
        if path.exists():
            return sorted([p.name for p in path.iterdir() if p.is_dir()])
    raise FileNotFoundError(
        "Class list not found. Provide a valid --class-dir or --classes-file."
    )


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_transform() -> transforms.Compose:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )


def labels_to_fen_with_unknown(labels: List[str]) -> str:
    if len(labels) != 64:
        raise ValueError(f"Expected 64 labels, got {len(labels)}")

    reverse_map = {v: k for k, v in FENParser.PIECE_MAP.items()}

    fen_ranks = []
    for rank_idx in range(8):
        rank_labels = labels[rank_idx * 8 : (rank_idx + 1) * 8]
        fen_rank = ""
        empty_count = 0
        for label in rank_labels:
            if label == "empty":
                empty_count += 1
                continue

            if empty_count > 0:
                fen_rank += str(empty_count)
                empty_count = 0

            if label == "unknown":
                fen_rank += "?"
            else:
                fen_rank += reverse_map.get(label, "?")

        if empty_count > 0:
            fen_rank += str(empty_count)

        fen_ranks.append(fen_rank)

    return "/".join(fen_ranks)


def labels_to_fen(labels: List[str], include_unknowns: bool) -> str:
    if include_unknowns:
        return labels_to_fen_with_unknown(labels)
    normalized = ["empty" if label == "unknown" else label for label in labels]
    return labels_to_fen_with_unknown(normalized)


def _idx_to_square(idx: int) -> chess.Square:
    row = idx // 8
    col = idx % 8
    file_idx = col
    rank_idx = 7 - row
    return chess.square(file_idx, rank_idx)


def _square_xy(square: chess.Square, square_size: int, orientation: chess.Color) -> Tuple[float, float]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if orientation == chess.WHITE:
        x = file_idx * square_size
        y = (7 - rank_idx) * square_size
    else:
        x = (7 - file_idx) * square_size
        y = rank_idx * square_size
    return x, y


_SVG_SQUARE_RE = re.compile(
    r'<rect x="(?P<x>[\d.]+)" y="(?P<y>[\d.]+)" width="(?P<w>[\d.]+)" '
    r'height="(?P<h>[\d.]+)" class="[^"]* (?P<sq>[a-h][1-8])"'
)


def _extract_svg_square_geometry(
    svg: str, orientation: chess.Color
) -> Optional[Tuple[float, float, float]]:
    match = _SVG_SQUARE_RE.search(svg)
    if not match:
        return None
    x = float(match.group("x"))
    y = float(match.group("y"))
    square_size = float(match.group("w"))
    square_name = match.group("sq")
    file_idx = ord(square_name[0]) - ord("a")
    rank_idx = int(square_name[1]) - 1
    if orientation == chess.WHITE:
        origin_x = x - file_idx * square_size
        origin_y = y - (7 - rank_idx) * square_size
    else:
        origin_x = x - (7 - file_idx) * square_size
        origin_y = y - rank_idx * square_size
    return origin_x, origin_y, square_size


def _normalize_fen_rank(rank: str) -> str:
    normalized = []
    empty = 0
    for char in rank:
        if char.isdigit():
            empty += int(char)
        elif char == "?":
            empty += 1
        else:
            if empty:
                normalized.append(str(empty))
                empty = 0
            normalized.append(char)
    if empty:
        normalized.append(str(empty))
    return "".join(normalized)


def _board_fen_from_string(fen: str) -> str:
    board_fen = fen.split()[0]
    ranks = board_fen.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN: expected 8 ranks, got {len(ranks)}")
    normalized_ranks = [_normalize_fen_rank(rank) for rank in ranks]
    for idx, rank in enumerate(normalized_ranks):
        total = 0
        for char in rank:
            total += int(char) if char.isdigit() else 1
        if total != 8:
            raise ValueError(f"Invalid FEN rank length at {idx}: {rank}")
    return "/".join(normalized_ranks)


def _unknown_indices_from_fen(fen: str) -> List[int]:
    board_fen = fen.split()[0]
    ranks = board_fen.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN: expected 8 ranks, got {len(ranks)}")

    unknown_indices: List[int] = []
    idx = 0
    for rank in ranks:
        for char in rank:
            if char.isdigit():
                idx += int(char)
            else:
                if char == "?":
                    unknown_indices.append(idx)
                idx += 1

    if idx != 64:
        raise ValueError(f"Invalid FEN: expected 64 squares, got {idx}")

    return unknown_indices


def render_board_svg(
    fen: str,
    size: int = 512,
    orientation: chess.Color = chess.WHITE,
    show_unknowns: bool = True,
) -> str:
    board = chess.Board(None)
    board.set_board_fen(_board_fen_from_string(fen))

    svg = chess.svg.board(board=board, size=size, orientation=orientation, colors=CHESS_COM_COLORS)

    if not show_unknowns:
        return svg

    unknown_indices = _unknown_indices_from_fen(fen)
    if not unknown_indices:
        return svg

    origin = _extract_svg_square_geometry(svg, orientation)
    if origin is None:
        origin_x, origin_y, square_size = 0.0, 0.0, size / 8.0
    else:
        origin_x, origin_y, square_size = origin

    pad = square_size * 0.15
    stroke = max(1, int(round(square_size * 0.08)))

    x_marks = []
    for idx in unknown_indices:
        square = _idx_to_square(idx)
        x0, y0 = _square_xy(square, square_size, orientation)
        x0 += origin_x
        y0 += origin_y
        x1 = x0 + pad
        y1 = y0 + pad
        x2 = x0 + square_size - pad
        y2 = y0 + square_size - pad
        x_marks.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#d32f2f" stroke-width="{stroke}" stroke-linecap="round" />'
        )
        x_marks.append(
            f'<line x1="{x2}" y1="{y1}" x2="{x1}" y2="{y2}" '
            f'stroke="#d32f2f" stroke-width="{stroke}" stroke-linecap="round" />'
        )

    return svg.replace("</svg>", "\n" + "\n".join(x_marks) + "\n</svg>")


def classify_squares(
    model: nn.Module,
    squares: List[np.ndarray],
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
    threshold: float,
) -> Tuple[List[str], List[float], List[int]]:
    images = []
    for square in squares:
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        images.append(transform(img))

    if not images:
        return [], [], []

    batch = torch.stack(images).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

    confs_list = confs.cpu().tolist()
    preds_list = preds.cpu().tolist()

    labels = []
    unknown_indices = []
    for idx, (pred_idx, conf) in enumerate(zip(preds_list, confs_list)):
        if conf < threshold:
            labels.append("unknown")
            unknown_indices.append(idx)
        else:
            labels.append(class_names[pred_idx])

    return labels, confs_list, unknown_indices


def draw_unknown_x(image: np.ndarray) -> np.ndarray:
    marked = image.copy()
    h, w = marked.shape[:2]
    pad = int(min(h, w) * 0.15)
    cv2.line(marked, (pad, pad), (w - pad, h - pad), (0, 0, 255), 2)
    cv2.line(marked, (w - pad, pad), (pad, h - pad), (0, 0, 255), 2)
    return marked


def save_crops(
    squares: List[np.ndarray],
    labels: List[str],
    output_dir: Path,
    mark_unknowns: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = SquareExtractor(board_size=512)

    for idx, square in enumerate(squares):
        pos = extractor.get_square_position(idx)
        label = labels[idx]
        filename = f"{idx:02d}_{pos}_{label}.jpg"
        if label == "unknown" and mark_unknowns:
            image = draw_unknown_x(square)
        else:
            image = square
        cv2.imwrite(str(output_dir / filename), image)


def save_crops_grid(
    squares: List[np.ndarray],
    labels: List[str],
    output_path: Path,
    mark_unknowns: bool = True,
) -> None:
    if not squares:
        return
    square_h, square_w = squares[0].shape[:2]
    grid = np.zeros((square_h * 8, square_w * 8, 3), dtype=np.uint8)
    for idx, square in enumerate(squares):
        row = idx // 8
        col = idx % 8
        if labels[idx] == "unknown" and mark_unknowns:
            tile = draw_unknown_x(square)
        else:
            tile = square
        y1 = row * square_h
        y2 = y1 + square_h
        x1 = col * square_w
        x2 = x1 + square_w
        grid[y1:y2, x1:x2] = tile
    cv2.imwrite(str(output_path), grid)


def run_pipeline(
    image_path: str,
    model_path: str,
    output_dir: str,
    class_dir: Optional[str],
    classes_file: Optional[str],
    threshold: float,
    board_size: int,
    render_size: int,
    save_square_crops: bool,
    print_squares: bool,
    crops_dir: Optional[str],
    save_grid: bool,
    include_unknowns: bool = True,
    save_clean_board: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    detector = BoardDetector(board_size=board_size)
    extractor = SquareExtractor(board_size=board_size)

    warped = detector.detect_board(image, debug=False)
    if warped is None:
        raise RuntimeError("Board detection failed.")

    cv2.imwrite(str(output_path / "warped_board.jpg"), warped)

    squares = extractor.extract_squares(warped)
    if print_squares:
        print("Squares (index, position, shape):")
        for idx, square in enumerate(squares):
            pos = extractor.get_square_position(idx)
            print(f"{idx:02d} {pos} {square.shape}")

    class_names = load_class_names(class_dir, classes_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, len(class_names), device)
    transform = make_transform()

    labels, confs, unknown_indices = classify_squares(
        model=model,
        squares=squares,
        transform=transform,
        class_names=class_names,
        device=device,
        threshold=threshold,
    )

    fen = labels_to_fen(labels, include_unknowns)
    (output_path / "fen.txt").write_text(fen, encoding="utf-8")

    svg = render_board_svg(
        fen,
        size=render_size,
        orientation=chess.WHITE,
        show_unknowns=include_unknowns,
    )
    (output_path / "board.svg").write_text(svg, encoding="utf-8")

    if save_clean_board:
        clean_fen = labels_to_fen(labels, include_unknowns=False)
        (output_path / "fen_clean.txt").write_text(clean_fen, encoding="utf-8")
        clean_svg = render_board_svg(
            clean_fen,
            size=render_size,
            orientation=chess.WHITE,
            show_unknowns=False,
        )
        (output_path / "fen.svg").write_text(clean_svg, encoding="utf-8")

    if save_square_crops:
        target_dir = Path(crops_dir) if crops_dir else output_path / "crops"
        save_crops(squares, labels, target_dir, mark_unknowns=include_unknowns)

    if save_grid:
        save_crops_grid(
            squares,
            labels,
            output_path / "crops_grid.jpg",
            mark_unknowns=include_unknowns,
        )

    preds = []
    for idx, (label, conf) in enumerate(zip(labels, confs)):
        preds.append(
            {
                "index": idx,
                "square": extractor.get_square_position(idx),
                "label": label,
                "confidence": float(conf),
            }
        )
    (output_path / "predictions.json").write_text(
        json.dumps(preds, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chessboard inference pipeline.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--model",
        default=str(Path("model") / "resnet18_ft.pth"),
        help="Path to model .pth file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write outputs (fen.txt, board.svg, predictions.json).",
    )
    parser.add_argument(
        "--class-dir",
        default="dataset/train",
        help="Directory with class subfolders (ImageFolder order).",
    )
    parser.add_argument(
        "--classes-file",
        default=str(Path("model") / "classes.txt"),
        help="Text file with class names, one per line.",
    )
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold.")
    parser.add_argument("--board-size", type=int, default=512, help="Warped board size.")
    parser.add_argument("--render-size", type=int, default=512, help="Output render size.")
    parser.add_argument("--save-crops", action="store_true", help="Save per-square crops.")
    parser.add_argument(
        "--print-squares",
        action="store_true",
        help="Print index/position/shape for each square crop.",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Optional directory to save crops (defaults to output-dir/crops).",
    )
    parser.add_argument(
        "--save-grid",
        action="store_true",
        help="Save an 8x8 grid image of crops as outputs/crops_grid.jpg.",
    )
    parser.add_argument(
        "--save-clean-board",
        action="store_true",
        help="Save a clean board SVG (no X markers) and standard FEN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        class_dir=args.class_dir,
        classes_file=args.classes_file,
        threshold=args.threshold,
        board_size=args.board_size,
        render_size=args.render_size,
        save_square_crops=args.save_crops,
        print_squares=args.print_squares,
        crops_dir=args.crops_dir,
        save_grid=args.save_grid,
        save_clean_board=args.save_clean_board,
    )


if __name__ == "__main__":
    main()
