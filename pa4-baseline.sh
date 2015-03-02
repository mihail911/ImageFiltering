#!/bin/sh

# Set number of OpenMP threads
export OMP_NUM_THREADS=8

# Clean up the directory
make clean

# Compile the program
make DEBUG=0

# Run the program
./CpuReference images/noisy_01.nsy
./CpuReference images/noisy_02.nsy
./CpuReference images/noisy_03.nsy
