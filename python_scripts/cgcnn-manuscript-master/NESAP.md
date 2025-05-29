# Benchmark Test 

## Measurement on Edison

```C
sbatch run_benchmark.sh 
```

## Benchmark Results 
Current device:cpu

Time to load 2999 documents: 3.011513 seconds

Time to convert 9 documents into SDT list: 58.292460 seconds

Time to train the model using 625 training examples: 334.683130 seconds

Training error: 0.475118 ev

Validation error: 0.578228 ev

Test error: 0.344367 ev

Time to predict energy for 9 documents: 0.001414 seconds
