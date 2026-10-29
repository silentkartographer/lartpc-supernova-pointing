"""Head-tail directional reconstruction of low-energy electrons from raw LArTPC
wire-plane images, for supernova neutrino pointing.

The pure-NumPy detector geometry (3D <-> 2D wire-plane transforms, the U/V/Z consistency
condition, and the voting system) lives in :mod:`main.geometry` and has no deep-learning
dependencies.
"""

