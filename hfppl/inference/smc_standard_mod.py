import copy
from ..util import logsumexp
import numpy as np
import asyncio

async def smc_standard_mod(model, n_particles, ess_threshold=0.5):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling.
    
    Args:
        model (hfppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.
    
    Returns:
        particles (list[hfppl.modeling.Model]): The completed particles after inference.
    """
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    # weights, in log space
    weights = [0.0 for _ in range(n_particles)]
    
    step_num = 1
    prev_avg_weight = 0

    while any(map(lambda p: not p.done_stepping(), particles)):
        # Step each particle
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])
        for i, p in enumerate(particles):
            print(f"├ Particle {i} (weight {p.weight:.4f}): {p}")
        # Normalize weights
        weights = np.array([p.weight for p in particles])
        total_weight = logsumexp(weights)
        weights_normalized = weights - total_weight

        # Compute log average weight (used if resampling, else only for printing)
        avg_weight = total_weight - np.log(n_particles)
        print(f"│ Average weight: {avg_weight:.4f}")
        
        # Resample if necessary
        if -logsumexp(weights_normalized * 2) < np.log(ess_threshold) + np.log(n_particles):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(weights_normalized)
            particles = [copy.deepcopy(particles[np.random.choice(range(len(particles)), p=probs)]) for _ in range(n_particles)]

            for p in particles:
                p.weight = avg_weight
            print(f"└╼  Resampled. Weights now each = {avg_weight:.4f}.")
        else:
            print("└╼")
        # Print the diff avg weight (biased (over)estimator of marginalizing constant)
        if step_num > 1:
            print(f"{avg_weight - prev_avg_weight =:.4f}")
            prev_avg_weight = avg_weight
        step_num += 1
    return particles