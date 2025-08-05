#!/bin/bash

# drivers=('16' '81' '55' '4' '63' '1' '44' '22' '23' '10' '14' '3' '77' '18' '2' '24' '31' '11' '27' '20')
drivers=('1' '11' '16' '55' '63' '44' '14' '18' '4' '81' '23' '2' '77' '24' '27' '20' '22' '31' '10' '3' '30' '43')

# Run in batches of 5
for ((i=0; i<${#drivers[@]}; i+=5)); do
   # Start up to 5 processes in parallel
   for ((j=i; j<i+5 && j<${#drivers[@]}; j++)); do
       uv run python train.py --driver ${drivers[j]} --prediction-horizon 10 --window-size 100 &
   done
   # Wait for this batch to complete before starting the next
   wait
done