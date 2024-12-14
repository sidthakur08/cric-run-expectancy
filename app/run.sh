#!/usr/bin/env bash
set -e  # Stop the script if any command fails

echo "Generating data..."
python data_gen.py

echo "Training model..."
python model.py

echo "Predicting for Ireland overs..."

python pred.py \
    --model models/linear_model.pkl \
    --csv data/delivery.csv \
    --team "Ireland" \
    --start_over 1 \
    --end_over 5