# I shipped a 97% earthquake detector, then proved it was mostly a trick

*A short, honest debugging story — the kind that's more useful than the metric.*

I built a small 1-D CNN to spot the first tremor of an earthquake (the P-wave) in a
10-second, 3-channel seismogram. On a balanced test set it hit **97.4% accuracy, 0.994
ROC-AUC**. That's the number most people would put on a résumé and move on.

I didn't, because the plots looked *too* clean. Here's the whole arc.

## 1. The tell

Every earthquake example had the same silhouette: flat and quiet, then a sudden spike.
Real P-waves do emerge from background noise — but a shape that consistent across every
positive is exactly what a model exploits as a **shortcut**.

First hypothesis from a reviewer: maybe the flat part is literal zero-padding from how
windows are cut. I checked the raw data: across 800 earthquake traces, **0%** had an
all-zero lead-in and only ~0.1% of pre-P samples were exactly zero. The quiet part is
*genuine ambient noise*, about the same amplitude as the dedicated noise traces. Myth
busted — but the concern was still right, for a subtler reason.

## 2. The mechanism

Each window is z-scored (subtract mean, divide by standard deviation). The earthquake
coda is ~27× larger than its own pre-P noise, so z-scoring squashes that real pre-P noise
until it *looks* flat, while pure-noise windows stay uniformly "loud." After normalization:

| First 1.5 s energy (normalized) | Noise | Earthquake |
|---|---|---|
| std | 0.99 | **0.23** |

So the classes are trivially separable on **amplitude envelope alone** — no seismology
required.

## 3. The proof

If the network rides the envelope, a dumb baseline should match it. It does. A single
feature — the amplitude of the first 1.5 seconds — plus a threshold scores **95.4%
(0.979 AUC)**; a 5-feature logistic regression hits **95.3% / 0.991**. The 110k-parameter
CNN bought about two points over `if the window starts quiet: earthquake`.

## 4. The reckoning

Balanced accuracy is the wrong test for a detector that will see 99%+ noise in the field.
So I ran it over a real continuous hour (the 2019 Ridgecrest M7.1). On the ~10 quiet
minutes *before the quake existed*, the model called it "earthquake" more than half the
time — roughly **1,440 false alarms per hour**. On a real stream it would cry wolf every
few seconds.

## 5. The fixes — and the twist

Two obvious culprits, both addressed and measured:

- **Leakage:** the train/test split was at the window level, so ~25% of test windows shared
  an event with training. I switched to an **event-grouped split**. Accuracy moved 97.4% →
  **96.9%**. Real, but minor.
- **The shortcut:** I replaced z-scoring with **AGC (automatic gain control)**, which
  divides by local RMS and flattens the envelope. It worked — the shortcut feature collapsed
  from **0.986 → 0.613 AUC**, the two classes now have equal early-window energy, and honest
  accuracy fell to **95.0%**. The model can no longer cheat.

Then the twist: I re-ran the continuous test. The shortcut-free model still fired **~1,500
false alarms/hour**. Meanwhile a classical **STA/LTA** trigger — the decades-old baseline —
managed **~126/hour** and still caught the M7.1.

**Fixing the shortcut did not fix the detector.**

## 6. The real diagnosis

The remaining problem isn't normalization — it's **domain shift**. STEAD's "noise" class is
curated, isolated windows; real continuous ambient noise is different, so a model trained to
separate STEAD-noise from STEAD-earthquake doesn't transfer to a live stream even once the
amplitude crutch is gone. The right next step is to **train on realistic continuous noise**
(mine hard negatives from long, quiet, multi-station streams), not to keep tuning the
preprocessing. Until then, STA/LTA wins, and the repo says so out loud.

## Why I'm sharing the messy version

Anyone can post a 97% number. The skills worth hiring for are: distrusting your own metric,
building the ablation that exposes it (`early_std` baseline), measuring the metric that
actually matters (false alarms/hour vs a classical baseline), and being honest when the fix
falls short and naming what's next. That's the story here.

*Everything is reproducible: `scripts/diagnose_shortcut.py` regenerates the audit, and
`scripts/evaluate_baselines.py` runs CNN vs STA/LTA on the same continuous data. Full analysis
in [`shortcut_and_leakage_analysis.md`](shortcut_and_leakage_analysis.md).*
