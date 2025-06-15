# PLL and AFC Components: NCOs, Loop Filter Design & Implementation

1. Component classes for implementing NCO's (32-bit and 48-bit), DSP loop filters (two types), accumulator (used in the loop filters), $K_p$ and $K_i$ calculation functions, and several digital PLL analysis functions.
1. Two implementations examples (one PLL and one AFC).
1. To implement the AFC a quadricorrelator/frequency discriminator class for sample-by-sample processing, as is needed for tracking loop simulation/implementation.