# MountainCar monitoring expert

The rule text is copied from the installed NSFR MountainCar/default rules. This
local selection supplies its own valuation because the installed simulator
package does not include a MountainCar valuation file. It changes neither the
simulator nor managed packages.

The input converter duplicates raw `[position, velocity]` into the `obj1` agent
and `obj2` environment objects. Negative velocity recommends left, positive
velocity recommends right, and zero velocity recommends noop. Predicates use
0.99 for true and 0.01 for false, with a zero velocity threshold. No synthetic
state grid or visited-state claim is involved.
