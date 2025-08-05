#!/bin/bash

drivers=('16' '81' '55' '4' '63' '1' '44' '22' '23' '10' '14' '3' '77' '18' '2' '24' '31' '11' '27' '20')

# Run in batches of 5
for ((i=0; i<${#drivers[@]}; i+=5)); do
   # Start up to 5 processes in parallel
   for ((j=i; j<i+5 && j<${#drivers[@]}; j++)); do
       uv run python train_pca.py --driver ${drivers[j]} &
   done
   # Wait for this batch to complete before starting the next
   wait
done