#!/bin/bash
source ~/lce_env/bin/activate
cd /mnt/c/Projects/PhD/NAS-BNN/WakeVision
for key in 3 4 5 6; do
  echo ""
  echo "===== KEY $key ====="
  python eval_lce_keras.py --key "$key" --batch 64 2>&1 | grep -Ev "tensorflow|TF-TRT|oneDNN|cudart|NUMA|dlopen|rebuild|AVX|FMA|XNNPACK|cpu_feature|Skipping|GPU"
done
