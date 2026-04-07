#!/bin/bash
# Create per-dataset symlinks for bone_synthetic.hdf5
#
# The bone_synthetic HDF5 contains multiple datasets (hip, leg, pig, rib,
# vertebra) under data_split/<name>/..., but the datamodule derives the
# dataset name from the filename. Symlinks make each name resolve to the
# same file.

DIR="${1:?"Usage: bash setup_fractura.sh <data_root>"}"
HDF5_FILE="$DIR/bone_synthetic.hdf5"

if [ ! -f "$HDF5_FILE" ]; then
    echo "ERROR: $HDF5_FILE not found."
    exit 1
fi

for name in hip leg pig rib vertebra; do
    ln -sfv "$HDF5_FILE" "$DIR/${name}.hdf5"
done

echo "Done."
