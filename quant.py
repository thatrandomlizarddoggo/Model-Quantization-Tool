import argparse
import time
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from enum import Enum


class QuantType(Enum):
    INT1 = "int1"  # 1-bit
    INT2 = "int2"  # 2-bit integer
    NF4 = "nf4"    # 4-bit NormalFloat
    FP4 = "fp4"    # 4-bit Floating Point
    INT8 = "int8"  # 8-bit integer
    FP8 = "fp8"    # 8-bit Floating Point
    FP16 = "fp16"  # 16-bit half precision
    BF16 = "bf16"  # 16-bit brain float


def get_quantization_config(quant_type, compute_dtype):
    """Return the appropriate quantization config"""
    if quant_type == QuantType.INT1:
        return BitsAndBytesConfig(
            load_in_4bit=True,  # Using 4bit container for 1-bit weights
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=torch.uint8
        )
    elif quant_type == QuantType.INT2:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
    elif quant_type == QuantType.NF4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
    elif quant_type == QuantType.FP4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=compute_dtype
        )
    elif quant_type == QuantType.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant_type == QuantType.FP8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True
        )
    return None


def parse_quant_type(quant_str):
    try:
        return QuantType(quant_str.lower())
    except ValueError:
        raise ValueError(f"Invalid quant type: {quant_str}, possible types: {[e.value for e in QuantType]}")


def format_time(seconds):
    return str(timedelta(seconds=seconds))


def main():
    total_start = time.time()

    parser = argparse.ArgumentParser(description="Model Quant Tool")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--quant", required=True, choices=[e.value for e in QuantType],
                        help=f"Quantization options ({', '.join([e.value for e in QuantType])})")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()

    quant_type = parse_quant_type(args.quant)

    compute_dtype = torch.float16
    if quant_type in [QuantType.FP16, QuantType.FP8]:
        compute_dtype = torch.float16
    elif quant_type == QuantType.BF16:
        compute_dtype = torch.bfloat16

    # Load tokenizer
    print(f"\n[1/3] Loading tokenizer...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Tokenizer loaded in {format_time(time.time() - tokenizer_start)}")

    # Model config
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": compute_dtype
    }

    # Apply quantization
    if quant_config := get_quantization_config(quant_type, compute_dtype):
        print(f"\n[2/3] Configuring {quant_type.value} quantization...")
        model_kwargs["quantization_config"] = quant_config
        model_kwargs.pop("torch_dtype", None)

    # Load model
    print(f"\n[3/3] Loading model with {quant_type.value}...")
    model_start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        print(f"Model loaded in {format_time(time.time() - model_start)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        if quant_type == QuantType.INT1:
            print("Note: 1-bit quant is experimental and can fail")
        return

    # Save model
    print(f"\nSaving to {args.output}...")
    save_start = time.time()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved in {format_time(time.time() - save_start)}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {format_time(total_time)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()