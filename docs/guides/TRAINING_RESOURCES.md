# Local training resources and launch stages

Settings → Training Pipeline controls Theta's local workload in the selected workspace:

- **Maximum training workers** is a shared cap across experiments, methods, training seeds, and trials. `0` means automatic admission from the CPU/RAM budgets; `1` means one worker at a time. Evaluation inside a worker occupies its slot.
- **CPU threads per worker** sets Torch and supported native numerical-library thread limits for newly started workers. The scheduler retains at least this CPU reservation while the worker runs, including quiet startup and evaluation phases.
- **CPU and RAM budgets** regulate admission using Theta's worker reservations and observed usage. They are soft targets with five percentage points of tolerance. Running tasks finish if a target is lowered or usage grows; new tasks wait until their reservations fit.

These are workspace admission and thread controls, not operating-system CPU/RAM quotas. Other applications' CPU use and free system RAM are displayed as telemetry; they do not impose another admission gate or cause Theta to cancel research. The operating system remains responsible for host memory pressure. Actual usage can exceed estimates or thread controls in third-party components.

An experiment's `variability.parallel_runs` independently caps simultaneous training runs. `1` means sequential execution, including under resource management. Trial `n_jobs` and sequential method settings also remain effective. The shared worker cap and resource budgets can reduce these limits further. Seeds that have not entered the executor do not reserve FIFO positions ahead of another experiment's ready work.

The queue follows runner stage events:

- **Waiting / Starting**: resource admission, interpreter/model setup, or initial diagnostic setup.
- **Testing**: policy evaluation, including evaluation at training step zero.
- **Diagnostics**: optional monitor collection after an update.
- **Training**: training steps can advance.

Testing and diagnostics retain the last training step count and clear speed/ETA. Training speed is estimated again from fresh training progress after resuming, so an old steps-per-second value does not suggest that training is advancing during evaluation. Runner components should bracket synchronous work with `run.phase("testing", ...)` or `run.phase("diagnostics", ...)`, and use `run.set_stage("training", ...)` when training begins. Older runners without stage events keep their existing display behavior; the editable `sb3-dev` runner and core environment evaluator emit these phases.

On macOS, new provider and task workers request user-initiated activity for App Nap protection and user-initiated QoS on their execution thread. No administrator access, real-time policy, or privileged negative nice value is required. The telemetry start record stores which scheduling requests succeeded. Child numerical libraries retain their own scheduling behavior; QoS requests do not isolate a worker from contention, thermal changes, or OS scheduling, and cannot guarantee identical wall-clock research timings.

Apple references: [Prioritize Work at the App Level](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheAppLevel.html), [User-initiated QoS](https://developer.apple.com/documentation/Dispatch/DispatchQoS/userInitiated).
