# 2.0.0

Implements scheduler contract v2: remote jobs run from explicit experiment bundles.
Declares its `packaging` dependency. Builds the job script's environment block and
Python command itself (`shell.py`); the IDE no longer provides those helpers
(Theta SDK 3.1).
