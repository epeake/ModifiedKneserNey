# ModifiedKneserNey

As part of an independent study in natural language processing, I implemented a modified, interpolated Kneser-Ney smoothing algorithm needed for a research project.  Looking online, I could not find a Kneser-Ney smoothing algorithm that met my exact needs, so I created my own.  Here is what is special about this version:

1)	It has a correction for out-of-vocabulary words, necessary for scoring probabilities with unseen n-grams
2)	It estimates discount values based on training data instead of setting them to a fixed value of the typically used .75
3)	It is super easy to use

Note: This has not been optimized for very large data sets, but appears to train quickly in   
      practice.  New features and versions, which may include bug fixes, to come.
