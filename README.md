# ModifiedKneserNey

As part of an independent research project, focusing on natural language processing, I implemented a modified, interpolated [Kneser-Ney smoothing](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing) algorithm.  Looking online, I could not find a Kneser-Ney smoothing algorithm that met my exact needs, so I created my own.  

### What's special about my version:

1)	It has a correction for out-of-vocabulary words, necessary for scoring probabilities with unseen n-grams
2)	It estimates discount values based on training data instead of setting them to a fixed value of the typically used .75
3)	It is super easy to use

### Example

```python3
# let corpus represent a large string of training data
# let scentence represent a string that you wish to score

kn = KneserNey()
kn.train(corpus)
kn.log_score_per_ngram(scentence)

# Done!:)
```

**Note:** New features and versions, which may include bug fixes, to come.

---

## Requirements
* **Python 3**, including:
  * **nltk**
  * **numpy**

## References: 
* Stanley F. Chen, Joshua Goodman (1999), ”An empirical study of smoothing techniques for language modeling,” in Computer Speech and Language, vol. 13, Issue 4, pp. 359-394.

* P. Taraba (2007), ”Kneser-Ney Smoothing With a Correcting Transformation for Small Data Sets,” in IEEE Transactions on Audio, Speech, and Language Processing, vol. 15, no. 6, pp. 1912-1921.

* Heafield, Kenneth and Pouzyrevsky, Ivan and H Clark, Jonathan and Koehn, Philipp. (2013). ”Scalable Modified Kneser-Ney Language Model Estimation” in Proceedings of the 51st Annual Meeting of the Association for Computational Linguistic, vol. 2, pp. 690-696.

* Kneser, Reinhard and Hermann Ney (1995), ”Improved backing-off for M-gram language modeling.” ICASSP. D. Jurafsky and J. H. Martin (2017), ”Speech and Language Processing,” (Third Edition draft)
