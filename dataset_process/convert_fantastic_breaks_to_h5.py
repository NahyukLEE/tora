#!/usr/bin/env python3
"""
Convert the Fantastic Breaks dataset to HDF5 format compatible with both RPF and GARF.

Fantastic Breaks layout:
    fantastic_breaks/
    ├── 00/
    │   ├── 00002/
    │   │   ├── model_b_0.ply   (broken piece)
    │   │   ├── model_r_0.ply   (restoration piece)
    │   │   ├── model_c.ply     (complete model, ignored)
    │   │   └── meta_0.npz      (ignored)
    │   └── ...
    └── ...

Only model_b_0.ply and model_r_0.ply are kept as assembly parts.

Output HDF5 structure (Breaking Bad format, compatible with both RPF and GARF):
    <fragment_key>/
        pieces/
            0/  (model_b_0)
                vertices, faces, normals, shared_faces
            1/  (model_r_0)
                vertices, faces, normals, shared_faces
        pieces_names  (["model_b_0", "model_r_0"])

Usage:
    python convert_fantastic_breaks_to_h5.py \
        --data_root /path/to/fantastic_breaks \
        --output_path fantastic_breaks.hdf5 \
        --val_ratio 0.2 \
        --seed 42 \
        --verify
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FantasticBreaks2H5")

PART_FILES = ["model_b_0.ply", "model_r_0.ply"]


def load_ply_file(ply_path):
    """Load a PLY file and return vertices, faces, and normals."""
    mesh = trimesh.load(str(ply_path), process=False)

    vertices = np.array(mesh.vertices, dtype=np.float32)

    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        faces = np.array(mesh.faces, dtype=np.int32)
    else:
        faces = np.array([], dtype=np.int32).reshape(0, 3)

    normals = None
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        normals = np.array(mesh.vertex_normals, dtype=np.float32)
    elif len(faces) > 0 and hasattr(mesh, "face_normals"):
        normals = np.zeros_like(vertices, dtype=np.float32)
        for i, face in enumerate(faces):
            fn = mesh.face_normals[i]
            for vi in face:
                normals[vi] += fn
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norms > 0, normals / norms, normals)
    else:
        normals = np.zeros_like(vertices, dtype=np.float32)
        normals[:, 2] = 1.0

    return vertices, faces, normals


def find_fragments(data_root):
    """Walk the two-level directory and return a sorted list of (fragment_key, fragment_path) tuples."""
    data_root = Path(data_root)
    fragments = []
    for category_dir in sorted(data_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for frag_dir in sorted(category_dir.iterdir()):
            if not frag_dir.is_dir():
                continue
            # Check that the required part files exist
            if all((frag_dir / f).exists() for f in PART_FILES):
                key = f"{category_dir.name}_{frag_dir.name}"
                fragments.append((key, frag_dir))
            else:
                logger.warning(f"Skipping {frag_dir}: missing part files")
    return fragments


def convert(data_root, output_path, dataset_name, val_ratio, seed):
    fragments = find_fragments(data_root)
    logger.info(f"Found {len(fragments)} fragments")
    if not fragments:
        logger.error("No fragments found")
        return False

    # Train/val split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(fragments))
    n_val = int(len(fragments) * val_ratio)
    if 0 < val_ratio < 1:
        n_val = max(1, n_val)
    val_indices = set(indices[:n_val].tolist())

    train_keys, val_keys = [], []
    for i, (key, _) in enumerate(fragments):
        if i in val_indices:
            val_keys.append(key)
        else:
            train_keys.append(key)

    logger.info(f"Split: {len(train_keys)} train, {len(val_keys)} val")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        # data_split
        split_grp = h5.create_group("data_split").create_group(dataset_name)
        split_grp.create_dataset("train", data=[k.encode() for k in train_keys])
        split_grp.create_dataset("val", data=[k.encode() for k in val_keys])

        # Fragment data (Breaking Bad format: pieces/0, pieces/1, ...)
        for key, frag_dir in fragments:
            frag_grp = h5.create_group(key)
            pieces_grp = frag_grp.create_group("pieces")
            piece_names = []

            for idx, part_file in enumerate(PART_FILES):
                part_stem = Path(part_file).stem  # model_b_0, model_r_0
                ply_path = frag_dir / part_file
                vertices, faces, normals = load_ply_file(ply_path)
                if vertices is None or len(vertices) == 0:
                    logger.warning(f"Empty mesh: {ply_path}")
                    continue

                part_grp = pieces_grp.create_group(str(idx))
                part_grp.create_dataset("vertices", data=vertices, compression="gzip")
                if faces is not None and len(faces) > 0:
                    part_grp.create_dataset("faces", data=faces, compression="gzip")
                    # shared_faces: -1 for all faces (no adjacency info)
                    shared_faces = -np.ones(len(faces), dtype=np.int64)
                    part_grp.create_dataset("shared_faces", data=shared_faces, compression="gzip")
                if normals is not None and len(normals) > 0:
                    part_grp.create_dataset("normals", data=normals, compression="gzip")
                piece_names.append(part_stem)

            frag_grp.create_dataset(
                "pieces_names",
                data=[n.encode() for n in piece_names],
            )
            logger.info(f"Converted {key}")

    logger.info(f"Created {output_path}")
    return True


def verify(h5_path, dataset_name):
    logger.info("Verifying HDF5 structure...")
    with h5py.File(h5_path, "r") as h5:
        assert "data_split" in h5, "Missing data_split"
        assert dataset_name in h5["data_split"], f"Missing {dataset_name} in data_split"

        splits = h5["data_split"][dataset_name]
        train_keys = [k.decode() for k in splits["train"][:]]
        val_keys = [k.decode() for k in splits["val"][:]]
        logger.info(f"Splits: {len(train_keys)} train, {len(val_keys)} val")

        fragment_count = 0
        for key in h5.keys():
            if key == "data_split":
                continue
            grp = h5[key]
            assert "pieces" in grp, f"Fragment {key} missing 'pieces' group"
            assert "pieces_names" in grp, f"Fragment {key} missing 'pieces_names'"
            pieces = grp["pieces"]
            parts = sorted(pieces.keys())
            assert len(parts) >= 2, f"Fragment {key} has {len(parts)} parts, expected >= 2"
            for p in parts:
                assert "vertices" in pieces[p], f"Missing vertices in {key}/pieces/{p}"
                assert "faces" in pieces[p], f"Missing faces in {key}/pieces/{p}"
                assert "shared_faces" in pieces[p], f"Missing shared_faces in {key}/pieces/{p}"
                verts = pieces[p]["vertices"]
                assert verts.shape[1] == 3, f"Bad vertex shape in {key}/pieces/{p}: {verts.shape}"
            fragment_count += 1

        logger.info(f"Verified {fragment_count} fragments. All OK.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert Fantastic Breaks to RPF HDF5")
    parser.add_argument("--data_root", required=True, help="Path to fantastic_breaks directory")
    parser.add_argument("--output_path", default="fantastic_breaks.hdf5", help="Output HDF5 path")
    parser.add_argument("--dataset_name", default="fantastic_breaks", help="Dataset name in HDF5")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--verify", action="store_true", help="Verify output after conversion")
    args = parser.parse_args()

    success = convert(args.data_root, args.output_path, args.dataset_name, args.val_ratio, args.seed)
    if success and args.verify:
        verify(args.output_path, args.dataset_name)


if __name__ == "__main__":
    main()
