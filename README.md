# Reppl

Using [APPL](https://github.com/appl-team/appl) to reimplement popular algorithms for Large Language Models (LLMs) and prompts.

Contributions are welcome!

## Setup

You can save your API keys in `.env` file, example is in `.env.example`.
APPL will automatically read the environment variables in the `.env` file.
Read more on [APPL's documentation](https://appl-team.github.io/appl/setup/).

## List of algorithms

- [x] [Tree-of-thought](./tree-of-thoughts/): [[paper]](https://arxiv.org/abs/2305.10601), [[official code]](https://github.com/princeton-nlp/tree-of-thought-llm)
- [x] [Reflexion](./reflexion/): [[paper]](https://arxiv.org/abs/2303.11366), [[official code]](https://github.com/noahshinn/reflexion)
- [ ] [Prompt Optimization](./prompt-optimization/):
  - [ ] Large Language Models as Optimizers (OPRO): [[paper]](https://arxiv.org/abs/2309.03409), [[official code]](https://github.com/google-deepmind/opro/tree/main)
  - [x] [ExemplarGuided Reflection with Memory mechanism (ERM)](./prompt-optimization/erm_blog.md): [[paper]](https://arxiv.org/abs/2411.07446)
