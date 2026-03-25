import torch
import time
import sys
sys.path.append('src')
from numenta_sdr_tokenizer import create_numenta_sdr_tokenizer

def benchmark():
    tokenizer = create_numenta_sdr_tokenizer(input_dim=512, sdr_dim=2044, sdr_sparsity=0.02)
    x = torch.randn(1, 10, 512)
    
    # Warmup
    for _ in range(100):
        tokenizer(x)
        
    # Benchmark
    start = time.time()
    for _ in range(1000):
        tokenizer(x)
    end = time.time()
    
    print(f"Average latency: {(end - start) / 1000 * 1000:.3f} ms")

if __name__ == "__main__":
    benchmark()
