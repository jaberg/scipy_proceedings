:author: Brent Komer
:email: brent.komer@uwaterloo.ca
:institution: Senate House, S.P.Q.R.

:author: James Bergstra
:email: james.bergstra@uwaterloo.ca
:institution: Egyptian Embassy, S.P.Q.R.

.. :
    :author: Jarrod Millman
    :email: millman@rome.it
    :institution: Egyptian Embassy, S.P.Q.R.

.. :video: http://www.youtube.com/watch?v=dhRUe-gz690

------------------------------------------------
Hyperopt-Sklearn: Some subtitle
------------------------------------------------

.. class:: abstract

    Hyperopt is awesome, sklearn is awesome, we put them together.


.. class:: keywords

   machine learning, algorithm configuration, Python

Introduction
------------

Machine learning has become increasingly popular in both academic and
industrial circles over the last two decades as both the size of data sets and
the speed of computers have increased to the point where it is often easier to
fit complex functions to data using statistical estimation techniques than it
is to design them by hand.

Unfortunately, the fitting of such functions (training machine learning
algorithms) remains a relatively arcane art, typically requiring a graduate
degree and years of experience.

Recently, it has been shown that techniques for automatic algorithm configuration based on
Regression Trees [SMAC]_,
Gaussian Processes ([Mockus78]_,[Brochu]_,[Snoek]_),
and density-estimation techniques [TPE]_ can be viable alternatives to the
employment of a domain specialist for the fitting of statistical models.
Algorithm configuration approaches use function optimization and graph search techniques
to search more-or-less-efficiently through a large space of possible algorithm
configurations.

This paper introduces Hyperopt-sklearn, a software package that makes this
approach available to Python users by combining the Hyperopt algorithm
configuration package with the Scikit-learn machine learning package.
Hyperopt-sklearn provides a wrapper around Scikit-learn that describes how the
various estimators and preprocessing components can be created and chained
together.

Conceptually, Hyperopt-sklearn provides a single very high-level
estimator (HyperoptAutoEstimator??) whose "fit" method searches through *all*
of Scikit-learn's classifiers to find the one that gets the best score on the
given data. In actual fact, we have chosen a good palette of algorithms
that tends to work well for a range of classification problems for text and
small images (XXX). We imagine the exact search space will evolve over time as
it is refined in the course of use, but we want to share our encouraging
progress so far with the Python community.


Scikit-learn
------------

What is it, what algorithms and pre-processors are we going to be talking about?


Describing Scikit-Learn with Hyperopt
-------------------------------------

Brief review of Hyperopt, mainly citing [Hyperopt-scipy2013].
Reminder how do you describe search spaces in general?

Main content: how did we set up the search spaces around specific Sklearn
classifiers?  Pick a couple of representative ones, no need to go through them
all.

How many configuration parameters are there? What does the search tree look like?
Consider emphasizing the huge number of conditional parameters.



Of course, no paper would be complete without some source code.  Without
highlighting, it would look like this::

   def sum(a, b):
       """Sum two numbers."""

       return a + b

With code-highlighting:

.. code-block:: python

   def sum(a, b):
       """Sum two numbers."""

       return a + b

.. math::

   g(x) = \int_0^\infty f(x) dx

or on multiple, aligned lines:

.. math::
   :type: eqnarray

   g(x) &=& \int_0^\infty f(x) dx \\
        &=& \ldots


The area of a circle and volume of a sphere are given as

.. math::
   :label: circarea

   A(r) = \pi r^2.

.. math::
   :label: spherevol

   V(r) = \frac{4}{3} \pi r^3

We can then refer back to Equation (:ref:`circarea`) or
(:ref:`spherevol`) later.


Experiments that Show Hyperopt-sklearn is worth using
-----------------------------------------------------

Describe the data sets that you used. Where are they from, what are they
called, what's in them. How big are they. What have people used on them in the
past.

What accuracy does Hyperopt-sklearn get on each dataset? How does the curve of
best-accuracy-to-date look for each data set? How much computation time was
spent on each one?

How good is search?
What if you search only linearSGD or only the decision forest instead of the
whole space?

Understanding the model space:
What pre-processing ends up getting used?
What classifiers end up getting used?


.. figure:: figure1.png

   This is the caption. :label:`egfig`

.. figure:: figure1.png
   :align: center
   :figclass: w

   This is a wide figure, specified by adding "w" to the figclass.  It is also
   center aligned, by setting the align keyword (can be left, right or center).

.. figure:: figure1.png
   :scale: 20%
   :figclass: bht

   This is the caption on a smaller figure that will be placed by default at the
   bottom of the page, and failing that it will be placed inline or at the top.
   Note that for now, scale is relative to a completely arbitrary original
   reference size which might be the original size of your image - you probably
   have to play with it. :label:`egfig2`

As you can see in Figures :ref:`egfig` and :ref:`egfig2`, this is how you reference auto-numbered
figures.

.. table:: This is the caption for the materials table. :label:`mtable`

   +------------+----------------+
   | Material   | Units          |
   +------------+----------------+
   | Stone      | 3              |
   +------------+----------------+
   | Water      | 12             |
   +------------+----------------+
   | Cement     | :math:`\alpha` |
   +------------+----------------+


We show the different quantities of materials required in Table
:ref:`mtable`.


.. The statement below shows how to adjust the width of a table.

.. raw:: latex

   \setlength{\tablewidth}{0.8\linewidth}


.. table:: This is the caption for the wide table.
   :class: w

   +--------+----+------+------+------+------+--------+
   | This   | is |  a   | very | very | wide | table  |
   +--------+----+------+------+------+------+--------+


Future Work
-----------

Lots of areas for future work:

* Speed: Aborting points early when using K-fold validation
* Applications: Extending search space for e.g. regression problems
* Input domains: Including more pre-processing to handle different kinds of data


Conclusions
-----------

Hyperopt-sklearn automates the process of model search within the space
of algorithms provided by Scikit-learn. It finds pretty good models in
a pretty big space. Try it out!


Acknowledgements
----------------


References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

.. [SMAC] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

.. [Mockus78] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

.. [Brochu] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

.. [Snoek] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

.. [TPE] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

.. [Hyperopt] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.

