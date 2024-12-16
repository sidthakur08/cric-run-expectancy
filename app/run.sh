#!/usr/bin/env bash

set -e  # Stop the script if any command fails

# Step 1: Generate data
echo "Generating data..."
python data_gen.py

# Step 2: Train the model
echo "Training model..."
python model.py

# Step 3: Make predictions
echo "Predicting for Ireland overs..."
python pred.py \
    --model models/model.pkl \
    --csv data/delivery.csv \
    --team "Ireland" \
    --start_over 1 \
    --end_over 5

# Step 4: Run unit tests
echo "Running tests..."
pytest tests --maxfail=5 --disable-warnings -v