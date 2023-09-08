

# Overview

This repository is the implementation code of Binary Code Module Partitioning Based on Graph Embedding (GEBCM for short).

# Build and run

Following the instructions below to build and run the product.

Requirements:

- PyTorch-GPU 1.7 and above
- munkres 1.1 and above
- networkx
- numpy
- scipy 1.7 and above
- sklearn

Running GEBCM ( vcperf for example):

1. Clone the repository on your machine.

2. Go to the samples directory.

   ```
   cd samples
   ```

3. Execute the first phase of the model to obtain a pkl file (We have provided a pkl and bak file, so this step can be skipped).

   ```
   python ..\PreGebcm.py --name vcperf
   ```

4. Execute the second phase to obtain the results of the decomposition and the evaluation results.

   ```
   python ..\Gebcm.py --name vcperf
   ```



