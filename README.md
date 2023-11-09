Fork for experimentation.

# LLaMPPL + HuggingFace

[![docs](https://github.com/probcomp/hfppl/actions/workflows/docs.yml/badge.svg)](https://probcomp.github.io/hfppl)

LLaMPPL is a research prototype for language model probabilistic programming: specifying language generation tasks by writing probabilistic programs that combine calls to LLMs, symbolic program logic, and probabilistic conditioning. To solve these tasks, LLaMPPL uses a specialized sequential Monte Carlo inference algorithm. This technique, SMC steering, is described in [our recent workshop abstract](https://arxiv.org/abs/2306.03081).

This repository implements LLaMPPL for use with HuggingFace Transformers.

## Installation

If you just want to try out LLaMPPL, check out our [demo notebook on Colab](https://colab.research.google.com/drive/1uJEC-U8dcwsTWccCDGVexpgXexzZ642n?usp=sharing), which performs a simple constrained generation task using GPT-2. (Larger models may require more RAM or GPU resources than Colab's free version provides.)

To get started on your own machine, clone this repository and run `pip install .` (or `pip install -e .` if you plan to modify the library).

```
git clone https://github.com/probcomp/hfppl
cd hfppl
pip install .
```

Then, try running an example. Note that this will cause the weights for Vicuna-7b-v1.5 to be downloaded.

```
python examples/hard_constraints.py
```

If everything is working, you should see the model generate political news using words that are at most five letters long (e.g., "Dr. Jill Biden may still be a year away from the White House but she is set to make her first trip to the U.N. today.").

## Modeling with LLaMPPL

A LLaMPPL program is a subclass of the `hfppl.Model` class.

```python
from hfppl import Model, LMContext, TokenCategorical, CachedCausalLM

# A LLaMPPL model subclasses the Model class
class MyModel(Model):

    # The __init__ method is used to process arguments
    # and initialize instance variables.
    def __init__(self, lm, prompt, forbidden_letter):
        super().__init__()

        # A stateful context object for the LLM, initialized with the prompt
        self.context = LMContext(lm, prompt)
        self.lm = lm
        
        # The forbidden letter
        self.forbidden_tokens = [i for (i, v) in enumerate(lm.vocab)
                                   if forbidden_letter in v]
    
    # The step method is used to perform a single 'step' of generation.
    # This might be a single token, a single phrase, or any other division.
    # Here, we generate one token at a time.
    async def step(self):
        # Sample a token from the LLM -- automatically extends `self.context`.
        # We use `await` so that LLaMPPL can automatically batch language model calls.
        token = await self.sample(self.context.next_token(), 
                                  proposal=self.proposal())

        # Condition on the token not having the forbidden letter
        self.condition(token.token_id not in self.forbidden_tokens)

        # Check for EOS or end of sentence
        if token.token_id == self.lm.tokenizer.eos_token_id or str(token) in ['.', '!', '?']:
            # Finish generation
            self.finish()
    
    # Helper method to define a custom proposal
    def proposal(self):
        logits = self.context.next_token_logprobs.copy()
        logits[self.forbidden_tokens] = -float('inf')
        return TokenCategorical(self.lm, logits)

    # To improve performance, a hint that `self.forbidden_tokens` is immutable
    def immutable_properties(self):
        return set(['forbidden_tokens'])
```

The Model class provides a number of useful methods for specifying a LLaMPPL program:

* `self.sample(dist[, proposal])` samples from the given distribution. Providing a proposal does not modify the task description, but can improve inference. Here, for example, we use a proposal that pre-emptively avoids the forbidden letter.
* `self.condition(cond)` conditions on the given Boolean expression.
* `self.finish()` indicates that generation is complete.
* `self.observe(dist, obs)` performs a form of 'soft conditioning' on the given distribution. It is equivalent to (but more efficient than) sampling a value `v` from `dist` and then immediately running `condition(v == obs)`.

To run inference, we use the `smc_steer` or `smc_standard` methods:

```python
import asyncio
from hfppl import smc_steer

# Initialize the HuggingFace model
lm = CachedCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", auth_token=<YOUR_HUGGINGFACE_API_TOKEN_HERE>)

# Create a model instance
model = MyModel(lm, "The weather today is expected to be", "e")

# Run inference
particles = asyncio.run(smc_steer(model, 5, 3)) # number of particles N, and beam factor K
```

Sample output:

```
sunny.
sunny and cool.
34° (81°F) in Chicago with winds at 5mph.
34° (81°F) in Chicago with winds at 2-9 mph.
```

Further documentation can be found at https://probcomp.github.io/hfppl.