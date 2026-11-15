# ConceptGuard-RL

ConceptGuard-RL is a reinforcement-learning policy for network intrusion response that acts through a
concept bottleneck. In a partially observable five-action environment (classify benign, classify attack,
abstain, request telemetry at a cost, escalate to an analyst), the policy sees a cheap, always-visible block
of each flow. When it pays for telemetry, a concept predictor maps the revealed telemetry to eight
human-meaningful concepts, and the policy decides from these predicted concepts only. An analyst can replace
the predicted concepts with verified values at the same decision state, and the model reports how often this
changes, corrects or harms the decision.

The policy is trained with squared-error value regression, full-feedback targets (the reward of every
non-telemetry action is known once the label is known), and a telemetry price calibrated by dual ascent so
that the policy respects the per-window telemetry budget.

The whole model is in one file, `conceptguard_rl.py`: concept predictor, policy training with price
calibration and validation-based checkpoint selection, window-level scoring, and intervention scores.

## Install

The model needs three packages: `numpy`, `torch` (PyTorch) and `scikit-learn`. It was developed and tested with Python 3.13 and PyTorch 2.12 on CPU.

```bash
pip install numpy torch scikit-learn
```

Copy `conceptguard_rl.py` into your project, or clone this repository and import it from the repository root.

## Usage

The model takes numeric arrays that are already scaled and encoded. For each flow it needs the cheap block,
the telemetry block, the standardized concept values, and the label (1 = attack, 0 = benign). The label is
used only to form reward targets during training and is never part of an observation.

```python
from conceptguard_rl import ConceptGuardRL, Flows, Rewards, ACTIONS

train = Flows(cheap=cheap_train, telemetry=tel_train, concepts=conc_train, labels=y_train)
val   = Flows(cheap=cheap_val,   telemetry=tel_val,   concepts=conc_val,   labels=y_val)
test  = Flows(cheap=cheap_test,  telemetry=tel_test,  concepts=conc_test,  labels=y_test)

model = ConceptGuardRL(rewards=Rewards(escalate=0.2), budget=32, window=256, seed=1)
model.fit(train, val, episodes=300, eps_decay=150)

scores = model.score(test, n_windows=50, interventions=True)
print(scores["mean_reward_per_flow"], scores["coverage"], scores["classified_accuracy"])
print(scores["intervention"]["concept_visible_states"])   # action-change, error-correction, harm rates

print(model.telemetry_price_)                               # calibrated telemetry price (lambda)

# decision for one flow of a window, before and after a telemetry request
a = model.decide(cheap_row, budget_left=32, position=0)
a = model.decide(cheap_row, budget_left=31, position=0, previous_action=3, telemetry=tel_row)
print(ACTIONS[a])
```

`Rewards` sets the environment rewards (true positive, false positive, true negative, false negative,
abstain, telemetry, escalate). `score` rolls the greedy policy out on windows drawn from the given flows and
returns the mean reward per flow, coverage, accuracy and macro-F1 on classified flows, the macro-F1 over all
flows when escalations are resolved correctly by the analyst, and the telemetry and escalation rates.
`model.state_dict()` and `model.load_state_dict(...)` save and restore a fitted model.

## Result

On UNSW-NB15 (official training and testing partitions, ten seeds), with telemetry for at most 12.5% of the
flows in a window, the policy reaches a mean reward per flow of 0.518 when an analyst escalation is cheap
and 0.384 when it is costly, against 0.439 and 0.160 for the same policy trained with a standard DQN recipe
(Huber loss, chosen-action targets, no budget price). This is above a causal supervised cascade with the same telemetry budget in the cheap
regime (0.503) and within 0.013 of it in the costly regime (0.397). In the cheap regime the policy classifies
59.1% of the flows with an accuracy of 0.909 and defers the rest to the analyst; with telemetry for half of
the flows (budget 128, five seeds) the accuracy on the classified flows is 0.954 at a coverage of 0.708. Accuracy over
all flows does not reach 95%, because the official test partition limits full-feature tabular models to
about 0.90, so accuracy should always be read together with coverage.

## Citation

```bibtex
@unpublished{massaoudi2026conceptguardrl,
  author = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  title  = {{ConceptGuard-RL}: Intervenable Concept Bottlenecks for Explainable Intrusion Response},
  year   = {2026},
  note   = {Manuscript submitted to the Journal of Information Security and Applications}
}
```

## License

MIT License. See the `LICENSE` file.

## Contact

Mohamed Massaoudi, Department of Electrical and Computer Engineering, Texas A&M University,
mohamed.massaoudi@tamu.edu
