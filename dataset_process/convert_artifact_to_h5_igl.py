#!/usr/bin/env python3
"""
Convert artifact_compressed to HDF5 using libigl (official decompress.py pipeline).

Dependencies: numpy, scipy, h5py, igl (libigl python bindings), tqdm

    pip install numpy scipy h5py libigl tqdm
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import h5py
import igl
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Artifact2H5")


def _compute_vertex_normals(vertices, faces):
    """Compute area-weighted vertex normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(vertices)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals /= norms
    return normals.astype(np.float32)


def decompress_object(obj_dir):
    """Decompress all fracture events for one object using igl.

    Matches the official Breaking Bad decompress.py exactly.
    """
    obj_dir = Path(obj_dir)
    obj_id = obj_dir.name

    mesh_path = obj_dir / "compressed_mesh.obj"
    npz_path = obj_dir / "compressed_data.npz"
    if not mesh_path.exists() or not npz_path.exists():
        return []

    fine_vertices, fine_triangles = igl.read_triangle_mesh(str(mesh_path))
    piece_to_fine_vertices_matrix = load_npz(npz_path)

    frac_dirs = sorted(
        [d for d in obj_dir.iterdir()
         if d.is_dir() and (d.name.startswith("fractured_") or d.name.startswith("mode_"))],
        key=lambda d: d.name,
    )

    results = []
    for frac_dir in frac_dirs:
        npy_path = frac_dir / "compressed_fracture.npy"
        if not npy_path.exists():
            continue

        piece_labels = np.load(npy_path)
        fine_vertex_labels = piece_to_fine_vertices_matrix @ piece_labels
        n_pieces = int(np.max(piece_labels) + 1)

        pieces = []
        for i in range(n_pieces):
            tri_labels = fine_vertex_labels[fine_triangles[:, 0]]
            if not np.any(tri_labels == i):
                continue

            # Official decompress.py pipeline (igl calls)
            vi, fi = igl.remove_unreferenced(
                fine_vertices, fine_triangles[tri_labels == i, :])[:2]
            ui, _I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
            gi = J[fi]
            ffi, _ = igl.resolve_duplicated_faces(gi)
            nv, nf = igl.remove_unreferenced(ui, ffi)[:2]

            if len(nf) == 0 or len(nv) < 3:
                continue

            vertices = nv.astype(np.float32)
            faces = nf.astype(np.int32)
            normals = _compute_vertex_normals(vertices, faces)
            pieces.append((f"piece_{i}", vertices, faces, normals))

        frac_key = f"{obj_id}/{frac_dir.name}"
        if len(pieces) >= 2:
            results.append((frac_key, pieces))

    return results


def _decompress_wrapper(obj_dir):
    try:
        return decompress_object(obj_dir)
    except Exception as e:
        logger.warning(f"Failed {obj_dir}: {e}")
        return []


def _load_split_file(split_path):
    """Load official split file. Format: 'artifact/73400_sf' -> '73400_sf'."""
    obj_ids = set()
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj_ids.add(line.split("/")[-1])
    return obj_ids


def convert(data_root, output_path, dataset_name, split_dir, num_workers=None):
    data_root = Path(data_root)
    objects = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and (d / "compressed_mesh.obj").exists() and (d / "compressed_data.npz").exists()
    )
    logger.info(f"Found {len(objects)} objects")
    if not objects:
        return False

    # Load official splits
    split_dir = Path(split_dir)
    train_file = split_dir / f"{dataset_name}.train.txt"
    val_file = split_dir / f"{dataset_name}.val.txt"
    if not train_file.exists() or not val_file.exists():
        logger.error(f"Split files not found: {train_file}, {val_file}")
        return False

    train_objs = _load_split_file(train_file)
    val_objs = _load_split_file(val_file)
    logger.info(f"Official splits: {len(train_objs)} train, {len(val_objs)} val objects")

    # Parallel decompression
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(objects))
    logger.info(f"Decompressing with {num_workers} workers...")

    all_entries = []
    with mp.Pool(num_workers) as pool:
        for obj_results in tqdm(
            pool.imap_unordered(_decompress_wrapper, [str(o) for o in objects]),
            total=len(objects), desc="Objects",
        ):
            all_entries.extend(obj_results)

    logger.info(f"Decompressed {len(all_entries)} fracture events with >= 2 pieces")
    if not all_entries:
        return False

    # Assign splits
    train_keys, val_keys = [], []
    skipped = 0
    for frac_key, _ in all_entries:
        obj_id = frac_key.split("/")[0]
        if obj_id in train_objs:
            train_keys.append(frac_key)
        elif obj_id in val_objs:
            val_keys.append(frac_key)
        else:
            skipped += 1

    if skipped:
        logger.warning(f"{skipped} events from objects not in any split")
    logger.info(f"Split: {len(train_keys)} train, {len(val_keys)} val")

    # Write HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        split_grp = h5.create_group("data_split").create_group(dataset_name)
        split_grp.create_dataset("train", data=[k.encode() for k in train_keys])
        split_grp.create_dataset("val", data=[k.encode() for k in val_keys])

        for idx, (frac_key, pieces) in enumerate(all_entries):
            frag_grp = h5.create_group(frac_key)
            pieces_grp = frag_grp.create_group("pieces")
            piece_names = []

            for pidx, (part_name, vertices, faces, normals) in enumerate(pieces):
                part_grp = pieces_grp.create_group(str(pidx))
                part_grp.create_dataset("vertices", data=vertices, compression="gzip")
                if faces is not None and len(faces) > 0:
                    part_grp.create_dataset("faces", data=faces, compression="gzip")
                    part_grp.create_dataset("shared_faces",
                                            data=-np.ones(len(faces), dtype=np.int64),
                                            compression="gzip")
                if normals is not None and len(normals) > 0:
                    part_grp.create_dataset("normals", data=normals, compression="gzip")
                piece_names.append(part_name)

            frag_grp.create_dataset("pieces_names", data=[n.encode() for n in piece_names])

            if (idx + 1) % 500 == 0:
                logger.info(f"[{idx + 1}/{len(all_entries)}] {frac_key}: {len(piece_names)} pieces")

    logger.info(f"Created {output_path} with {len(all_entries)} fracture events")
    return True


def verify(h5_path, dataset_name):
    logger.info("Verifying...")
    with h5py.File(h5_path, "r") as h5:
        splits = h5["data_split"][dataset_name]
        train_keys = [k.decode() for k in splits["train"][:]]
        val_keys = [k.decode() for k in splits["val"][:]]
        logger.info(f"Splits: {len(train_keys)} train, {len(val_keys)} val")

        part_counts = []

        def _visit(name, obj):
            if isinstance(obj, h5py.Group) and "pieces" in obj:
                parts = sorted(obj["pieces"].keys())
                assert len(parts) >= 2, f"{name}: {len(parts)} parts"
                for p in parts:
                    assert "vertices" in obj["pieces"][p]
                part_counts.append(len(parts))

        h5.visititems(_visit)
        logger.info(
            f"Verified {len(part_counts)} events. "
            f"Parts: min={min(part_counts)}, max={max(part_counts)}, mean={np.mean(part_counts):.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert artifact_compressed to HDF5 (igl)")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset_name", default="artifact")
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    success = convert(args.data_root, args.output_path, args.dataset_name,
                      args.split_dir, args.num_workers)
    if success and args.verify:
        verify(args.output_path, args.dataset_name)
