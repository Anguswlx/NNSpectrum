# Reconstructing spectral functions via automatic differentiation

Cite this work as,

L. Wang, S. Shi, and K. Zhou, *Reconstructing Spectral Functions via Automatic Differentiation*, ArXiv:2111.14760 [Hep-Lat, Physics:Hep-Ph] (2021).


## Getting Started

The code requires Python >= 3.8 and PyTorch >= 1.2. You can configure on CPU machine and accelerate with a recent Nvidia GPU card.

## Running the tests

Run juputer notebook to generate mock data. Using Index and noise to specify propagator data.

```python
python NNspectrum1202.py  --Index 4 --noise 5
```

## Authors

* **Lingxiao Wang** - *Construct codes and write the preprint paper* - [Homepage](https://sites.google.com/view/lingxiao)
* **Shuzhe Shi** - *Check results and provide physics guidance*
* **Kai Zhou** - *Lead the project and complete the article.*

## License

This project is licensed under the MIT License - see the LICENSE file for details
