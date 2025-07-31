.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

Introduction
------------

Rare samples in datasets often represent novel patterns, edge cases, or important
subpopulations, and can therefore be crucial for understanding the true complexity of
the data. In classification tasks, a special form of rarity emerges: instances that are rare relative
to their assigned class label.
These samples may not be rare in a global sense, but they represent an uncommon
subconcept within their class. Just as globally rare samples can offer new insights into a
dataset as a whole, these intra-class rare samples can reveal valuable information about
the internal structure of their respective class.
However, in traditional machine learning algorithms these samples usually
have little influence on model training due to their rare nature.
This package provides a set of sklearn-compatible estimators and methods that are built to
measure and handle intra-class rarity, allowing these rare samples to have a more significant
impact on model training and prediction.

ICRRandomForestClassifier
-------------------------

The icrlearn package provides a new algorithm called Intra-Class Rare Random Forest.
This algorithm is a modification of the Random Forest algorithm that is designed to
handle data with intra-class imbalance in the form of rare samples within a class.
It uses a specified intra-class rarity measure to quantify the intra-class rarity of samples
in a dataset and adjust the bootstrap sampling of the training data or to apply corresponding
sample weights accordingly.

.. code-block:: python

   from icrlearn import ICRRandomForestClassifier

   clf = ICRRandomForestClassifier(rarity_measure="l2class")
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)

Rarity Measures
---------------

The rarity measures in this package are designed to quantify how rare a sample is within its class.
They are used by the ICRRandomForestClassifier to adjust the sampling of the training data or to apply
sample weights. The measures can also be used independently to analyze the rarity of samples in a dataset.

| They are based on two existing methods:
| 1. Local Outlier Probability (LoOP) [1]
| LoOP is a local density based outlier detection method by Kriegel et al. that is based on Local Outlier Factor (LOF), but provides outlier scores in the range of [0,1]. It is implemented in PyNomaly (:ref:`https://github.com/vc1492a/PyNomaly`).

| 2. L²-min [2]
| L²-min is a method by Błaszczyński and Stefanowski that calculates the rarity of a sample based on the distance to its k-nearest neighbors within its class.

.. code-block:: python

   from icrlearn.rarity import calculate_cb_loop, calculate_l2class

   # Class-Based Local Outlier Probability
   cb_scores = calculate_cb_loop(X, y)

   # Adapted L²-min rarity across classes
   l2_scores = calculate_l2class(X, y)

| For more information on using the classifier and the rarity measures, refer to the :ref:`API reference <api>`.

| [1] H.-P. Kriegel, P. Kröger, E. Schubert, and A. Zimek, “LoOP: local outlier probabilities,” in Proceedings of the 18th ACM conference on Information and knowledge management, Hong Kong China: ACM, Nov. 2009, pp. 1649–1652. doi: 10.1145/1645953.1646195.
| [2] J. Błaszczyński and J. Stefanowski, “Neighbourhood sampling in bagging for imbalanced data,” Neurocomputing, vol. 150, pp. 529–542, Feb. 2015, doi: 10.1016/j.neucom.2014.07.064.
