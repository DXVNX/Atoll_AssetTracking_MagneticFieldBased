# Methodology

Raw trajectories are parsed into a canonical schema, assigned direction-invariant corridor labels, divided into segment-pure timestamp windows, transformed into feature or sequence representations, and evaluated with trajectory-group isolation.

Magnetic magnitude: `B = sqrt(Bx^2 + By^2 + Bz^2)`.

Macro-F1 averages per-label F1 scores. Balanced accuracy averages per-label recall. PCA replaces correlated feature dimensions with orthogonal components ordered by retained variance.
