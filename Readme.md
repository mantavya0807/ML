
---

# NdLinear CNN Benchmark

## Overview

This project benchmarks the performance of **NdLinear** models compared to traditional CNNs using the **MNIST** dataset. The goal is to evaluate the efficiency of **NdLinear** in reducing model size and accelerating inference without compromising accuracy.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mantavya0807/ML.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install **NdLinear**:
   ```bash
   pip install git+https://github.com/your-username/ndlinear.git
   ```

4. Create the `results` directory:
   ```bash
   mkdir ./results
   ```

## Running the Benchmark

Run the benchmark script to train and evaluate the models:
```bash
python benchmark.py
```

This will train the following models:
- **StandardCNN**: Traditional CNN architecture
- **BasicNdLinearCNN**: CNN with **NdLinear** layers
- **OptimizedNdLinearCNN**: Enhanced **NdLinear** CNN
- **StructurePreservingCNN**: CNN with **NdLinear** for structure preservation

## Results

### Summary:
- **Standard CNN** achieved **96.03% accuracy** with **1,200,074 parameters**, taking **5.73s** for inference.
- **Basic NdLinear CNN** had **95.76% accuracy** with **21,298 parameters**, offering a significant reduction in model size and maintaining **5.69s** inference time.
- **Structure-Preserving CNN** showed **93.75% accuracy** with **21,236 parameters**, performing well with a **5.56s** inference time.
- **Optimized NdLinear CNN** achieved the highest accuracy of **97.63%** with **26,424 parameters**, and inference time of **6.13s**.

The **NdLinear** models show reduced parameter count while maintaining high accuracy, demonstrating better efficiency in terms of model size and performance.

All results are saved in the `./results` folder, including:
- Confusion matrices
- Learning curves (training and validation losses)
- Summary tables with model comparison

## Conclusion

This project shows how **NdLinear** improves CNN efficiency and performance. It’s an open-source contribution to enhancing model optimization using advanced layer replacement techniques.

--- 
