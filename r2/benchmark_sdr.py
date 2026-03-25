import torch
import time
from src.numenta_sdr_tokenizer import create_numenta_sdr_tokenizer

def benchmark():
    tokenizer = create_numenta_sdr_tokenizer(input_dim=512, sdr_dim=2044, sdr_sparsity=0.02)
    x = torch.randn(32, 128, 512)
    
    # Warmup
    for _ in range(10):
        tokenizer(x)
        
    # Benchmark
    start = time.time()
    for _ in range(100):
        tokenizer(x)
    end = time.time()
    
    print(f"Average latency: {(end - start) / 100 * 1000:.3f} ms")

if __name__ == "__main__":
    benchmark()
