#!/bin/bash

output_dir="model_output_"`date +"%Y%m%d%H%M%S"`
phase="train"
# training_set="./datasets/apple2orange" # Otherwise, you can use Queue or numpy ndarray to load image.
# tfrecord file
training_set="./datasets/tfrecords"
batch_size=2
training_steps=10100
summary_steps=100
save_steps=100
checkpoint_steps=500

# new training
python main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --training_set="$training_set" \
               --batch_size="$batch_size" \
               --training_steps="$training_steps" \
               --summary_steps="$summary_steps" \
               --save_steps="$save_steps" \
               --checkpoint_steps="$checkpoint_steps" 