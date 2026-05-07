# Targeted Pooled Latent-Space Steganalysis Applied to Generative Steganography, with a Fix

Etienne Levecque, Aurélien Noirault, Tomáš Pevný, Jan Butora, Patrick Bas, Rémi Cogranne

## Installation

Download the repository:

```bash
git clone
```

Install the required python packages in you environment:

```bash
pip install -r requirements.txt
```

## Reproduce the results

To get the plot from the paper, you can run the main script:

```bash
python statistical_test.py
```

If you want to apply this test on your own dataset, you can modify the end of the script and in particular the `paths`
variable:

```python
paths = [("path/to/cover.npy", 
          "path/to/stego.npy", 
          "label", 
          "color", 
          "linestyle")]
```
For each element in the `paths` list, the code will plot a new curve on the final graph.

The `.npy` files need to arrays of size `(N,M)` with `N` the number of samples and `M` the dimension of the latent
space. By default, the `read_norm()` function will crop the dataset to make them of equal length.

You can also try different values of `tau`.

## Citation

If you use our work, please consider citing us:

```
@misc{levecque2026targetedpooledlatentspacesteganalysis,
      title={Targeted Pooled Latent-Space Steganalysis Applied to Generative Steganography, with a Fix}, 
      author={Etienne Levecque and Aurélien Noirault and Tomáš Pevný and Jan Butora and Patrick Bas and Rémi Cogranne},
      year={2026},
      eprint={2510.12414},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2510.12414}, 
}
```