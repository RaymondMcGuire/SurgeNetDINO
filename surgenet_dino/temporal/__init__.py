"""Offline temporal phase recognition on top of the frozen per-frame encoder.

Frozen tokens ([final CLS | mean patch token]) -> per-video label-free standardisation ->
leave-one-video-out cross-fitted linear heads -> a small dilated temporal network with one global
self-attention layer -> exact decoding under order rules read from the training videos. Only the
labeled videos are used; every score is a grouped cross-validation development estimate.
"""
