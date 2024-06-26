# llaminate

> Optimized version of [llama3][github-llama3], using [tokun][github-tokun].

<img src="../.github/header.png" alt="Neural tokenization" title="Source: Image by Author and generated with MidJourney" width="100%" style="margin: auto;"/>

This project is a showcase for a neural tokenization technique.
Since the inputs are compressed and have a smaller shape, the LLM is downsized accordingly.

For example, llama3-8b is brought down to 34 million parameters instead of 8 billion.

## Installation

## Usage

## Resources

### Models

### Notebooks

Final model:

- pretraining: [file][notebook-github-pretrain] / [Google Colab][notebook-colab-pretrain]
- fine-tuning: file / Google Colab

## TODO

See [TODO](TODO.md).

## Credits

This project winks at [llama3 from Meta][github-llama3], but doesn't actually its weights nor code.

## License

Licensed under the [aGPLv3](LICENSE.md).

[github-llama3]: https://github.com/meta-llama/llama3
[github-tokun]: https://github.com/apehex/tokun

[notebook-colab-pretrain]: https://colab.research.google.com/github/apehex/llaminate/blob/main/notebooks/llaminate.student.pretrain.ipynb
[notebook-github-pretrain]: ../notebooks/llaminate.student.pretrain.ipynb
