"""Sampler module for token sampling."""

import torch
from torch import nn

from haste.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling

torch.manual_seed(0) 

class Sampler(nn.Module): 
    """Sampler module for token sampling.
    
    This class implements token sampling from logits with temperature control.
    """
    def __init__(self, sampler_x: float | None = None, async_fan_out: int = 3):
        """Initialize the Sampler module.
        
        Args:
            sampler_x (float | None, optional): Sampler X parameter for rescaling. Defaults to None.
            async_fan_out (int, optional): Async fan-out parameter. Defaults to 3.
        """
        super().__init__()
        self.sampler_x = sampler_x
        self.F = async_fan_out # will need to accomodate lists for hit/miss eventually 
    
    @torch.inference_mode() # what shape are logits during tree decode? MQ_LEN, 
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, is_tree: bool = False):
        """Forward pass of the Sampler module.
        
        Args:
            logits (torch.Tensor): Logits tensor of shape [B, V]
            temperatures (torch.Tensor): Temperatures tensor of shape [B]
            is_tree (bool, optional): Whether this is a tree decode. Defaults to False.
            
        Returns:
            torch.Tensor: Sampled tokens of shape [B]
        """
        # logits: [B, V], temperatures: [B]
        
        logits_cpy = logits.to(torch.float) 
        greedy_tokens = logits_cpy.argmax(dim=-1)

        # Fast path: any zero temperature rows are greedy
        temps = temperatures.to(logits.device)
        zero_mask = temps == 0
        safe_temps = torch.where(zero_mask, torch.ones_like(temps), temps)
        
        # Note: keep inplace ops for speed
        logits_cpy.div_(safe_temps.unsqueeze(dim=1))
        probs = torch.softmax(logits_cpy, dim=-1, dtype=torch.float)
        
        # Apply sampler_x rescaling when conditions are met
        if self.sampler_x is not None and is_tree:
            probs = apply_sampler_x_rescaling(probs, self.sampler_x, self.F)
        
        epsilon = 1e-10
        scores = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon)
        sample_tokens = scores.argmax(dim=-1)
        return torch.where(zero_mask, greedy_tokens, sample_tokens)


def profile_sampler():
    """Profile the sampler on [b, v] logits for b=128, v=150_000"""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nProfiling Sampler on {device}")
    
    # Test parameters
    b = 128
    v = 150_000
    
    # Create test data
    logits = torch.randn(b, v, device=device)
    temperatures = torch.rand(b, device=device) * 1.5  # temperatures in [0, 1.5]
    
    sampler = Sampler().to(device)
    
    print(f"Testing with batch_size={b}, vocab_size={v}")
    
    # Warm up
    print("Warming up sampler")
    for _ in range(10):
        _ = sampler(logits, temperatures)
    
    # Profile
    num_runs = 100
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        _ = sampler(logits, temperatures)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    sampler_time_ms = (end_time - start_time) * 1000 / num_runs
    
    print(f"Sampler time: {sampler_time_ms:.3f}ms")

# takes 0.5ms, negligible 
if __name__ == "__main__":
    profile_sampler()
