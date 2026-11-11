"""The part of the study's analysis that post 1 reads: section 9's one-policy estimator (`estimator`), the integrated crash
rate (`boundary.integrated_rate`) and S6's re-score under other crash limits (`sensitivities.s6_rescore`, `S6_LIMITS`).

The study's full analysis (every test, interval, verdict word and sensitivity that turns stored rows into results; spec
sections 7 to 10 and 20.3) reads retrained arms, which do not exist in post 1, and is part of the study's second post. It is
not in this public repository.

Units, everywhere: a crash rate is a fraction (0.10 is ten points); RMSE is metres; speed is m/s.
"""
