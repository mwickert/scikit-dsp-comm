PLL and AFC Components: NCOs, Loop Filter Design & Implementation
=================================================================

The module :doc:`../../synchronization` includes many functions and classes focused on:

#. Component classes for implementing NCO's (32-bit and 48-bit), DSP loop filters (two types), accumulator (used in the loop filters), :math:`K_p` and :math:`K_i` calculation functions, and several digital PLL analysis functions.
#. Two implementations examples (one PLL and one AFC).
#. To implement the AFC a quadricorrelator/frequency discriminator class for sample-by-sample processing, as is needed for tracking loop simulation/implementation.

.. toctree::

   ../../nb_examples/synchronization/PLL
   ../../nb_examples/synchronization/AFC