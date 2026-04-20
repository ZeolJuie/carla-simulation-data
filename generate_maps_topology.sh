#!/bin/bash

MAPS=("Town02_Opt")

for MAP in "${MAPS[@]}"; do
  echo "--------------------------------------------"
  echo "Processing: $MAP"
  
  TARGET_DIR="./topology/$MAP"
  mkdir -p "$TARGET_DIR"
  
  conda run -n carla python utils/generate_walkable_cells.py \
    --map-name "$MAP" \
    --output-dir "$TARGET_DIR"
done

echo "All Done!"