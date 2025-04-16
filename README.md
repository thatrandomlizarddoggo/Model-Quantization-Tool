# Model Quantization Tool

A Python script for quantizing Hugging Face models with support for 1-bit to 16-bit precision, including experimental binary quantization and FP8 support.

![Quantization](https://img.shields.io/badge/Quantization-1bit%20to%2016bit-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40%2B-orange)

## Features

- Supports 8 quantization modes:
  - 1-bit (experimental binary)
  - 2-bit integer
  - 4-bit NormalFloat (NF4)
  - 4-bit Floating Point (FP4)
  - 8-bit integer (INT8)
  - 8-bit Floating Point (FP8)
  - 16-bit half precision (FP16)
  - 16-bit brain float (BF16)
- Automatic device placement (CPU/GPU)
- Progress tracking and timing metrics
- Hugging Face model hub integration

## Installation

```bash
pip install -U bitsandbytes>=0.42.0 transformers>=4.40.0 torch>=2.3.0
```
## Usage
```bash
python quant.py --model [MODEL_NAME_OR_PATH] --quant [QUANT_TYPE]--output [OUTPUT_DIR]
```

## Hardware requirements 
### (based on 7B model)

| Quantization  | VRAM Required | Recommended GPU |
| ------------- | ------------- |-------------|
| 1-bit  | ~2GB  | A100/A6000  |
| 2-bit  | ~3GB  | A100/A6000  |
| 4-bit  | ~5GB  | RTX 3060+   |
| 8-bit  | ~8GB  | RTX 3060+   |
| FP16/BF16  | ~14GB  | rtx 3090+   |
note that cpu can be used recommended gpu is based of acceptable speed