---
name: clever-hans-effect
description: When a model achieves high accuracy by learning a spurious correlation (a shortcut) instead of the intended concept — chromatic aberration that gives away patch position, scanner type that correlates with disease prevalence, watermarks that flag class membership. The model passes training and validation but fails when the shortcut isn't there.
type: concept
sources:
  - raw/week-06/w06-slides.pdf
  - raw/week-06/w06-problem-set.pdf
  - raw/week-06/w6-notion.zip
status: stable
updated: 2026-04-26
---

*Clever Hans was a horse in 1900s Berlin who appeared to do arithmetic — tap his hoof the right number of times in response to "what is 5 + 3?" After a careful investigation, Carl Stumpf discovered Hans wasn't doing maths at all. He was reading micro-expressions on his trainer's face: the trainer relaxed when Hans's tap count reached the right answer, and Hans had learned to stop tapping at that signal. Hans solved the supervised task perfectly via a shortcut that had nothing to do with arithmetic. Modern neural networks do this constantly.*

## The pattern

A network achieves high accuracy on the training set and validation set, then fails when deployed. Investigation reveals: the model was using a **spurious feature** — a low-level cue that correlated with the label in the training distribution but doesn't carry the intended meaning. When the cue is absent (different equipment, clean data, distribution shift), the model breaks.

The defining symptom: high in-distribution metrics, poor out-of-distribution generalisation, and a saliency map / attribution analysis that points at something nonsensical (the bottom-left corner of every horse photo, the colour of an ultrasound border, etc.).

## Why it happens

Three ingredients:

1. **A spurious feature is in the data.** Some artefact, watermark, or systematic difference between classes that the network can detect.
2. **The feature is easier to learn than the real one.** Easier in the sense that it's lower-level / requires less depth / has cleaner gradients. Networks are lazy: SGD finds the easiest path to low loss.
3. **The feature correlates with the label well enough on training/validation.** If the validation set is sampled from the same distribution as training, validation will not catch the shortcut.

The result: the network achieves high accuracy via the shortcut, the gradient never has reason to learn the real feature, and the validation curve shows everything is fine.

## Three case studies from week 6

### Chromatic aberration in context prediction

Doersch et al.'s [[pretext-task|context-prediction]] network was supposed to learn that "cat ears go above cat faces." It actually learned to detect **chromatic aberration** — the colour-fringe artefact at lens edges that lets you tell, from a small patch, where it sat in the original image (centre = no fringing, corners = lots of fringing). The pretext task was solvable from optics alone, with zero scene understanding. Detected by careful analysis; fixed by randomly dropping colour channels during training, forcing the network to ignore colour and look at content.

### The "horse" classifier that learned watermarks

Lapuschkin et al. (2019) examined a network trained on PASCAL VOC. It classified horses with high accuracy. Saliency analysis showed the network was looking at the *bottom-left corner of every horse photo* — where the photographer's watermark appeared (`pferdefotoarchiv.de`). Synthesise an image of a car, paste the watermark in the corner: the network classifies it as a horse. The "horse-ness" feature learned was "presence of this specific watermark." Useless for any image that doesn't come from that photographer's archive.

### Hospital identity in medical imaging (Week 6 problem set Q2)

A diagnostic classifier is trained on three combined datasets:

- Hospital A: 5% disease prevalence, scanner X.
- Hospital B: 95% disease prevalence, scanner Y.
- Hospital C: 50% disease prevalence, scanner Z.

The data is shuffled and the network is trained to predict disease vs healthy. It achieves great training and validation accuracy.

What it has actually learned: "scanner Y → say diseased; scanner X → say healthy." The scanners' image artefacts (sensor noise, contrast, ring patterns, intensity range) are easy to detect; they correlate strongly with the label *because of class-imbalance differences between hospitals*. The network has learned to identify the hospital, not the disease.

When deployed at a fourth hospital with a fourth scanner: catastrophic failure. The shortcut isn't there.

**Defences (from the problem set solution):**

- **Balance the labels within each source.** Down-sample healthy in B, down-sample diseased in A — equalise positive/negative within each scanner so "scanner = label" no longer holds.
- **Validate on a balanced held-out set,** ideally from a single source (hospital C), to detect scanner-confounding.
- **Preprocess to remove scanner artefacts.** Normalise resolution, intensity, contrast across sources; strip metadata that leaks the source.
- **Use only one balanced source** (hospital C alone), trading dataset size for cleanliness.

The same problem applies to *segmentation*, not just classification — the network can use scanner-specific artefacts to infer probable liver shapes for that hospital's distribution, getting good in-distribution Dice scores while failing out-of-distribution.

## How to detect Clever Hans

- **Out-of-distribution evaluation.** A test set from a different source / scanner / photographer / acquisition setup. If accuracy plummets, you have a shortcut.
- **Saliency / attribution maps.** Visualise where the network is "looking." If the most important pixels are in places semantically irrelevant to the class (corners, borders, watermarks), suspect a shortcut.
- **Counterfactual probes.** Strip the suspected shortcut feature (mask the watermark, normalise the scanner artefact, drop the colour channel) and re-evaluate. If accuracy drops, the feature was load-bearing.
- **Class-balance analysis per source.** If your dataset has source labels (hospital, photographer, camera), check whether class is imbalanced *within* each source. If so, "source" is a candidate confounder.

## How to defend against it

The general principles:

- **Make the spurious feature uninformative.** Augment to break low-level shortcuts (random colour drop in context prediction). Balance labels within each data source. Normalise away artefacts.
- **Make the data distribution wider.** More sources, more conditions, more variation — the more diverse the dataset, the harder it is for a single shortcut to correlate with the label.
- **Force the network to learn deeper features.** Architectural / regularisation choices (e.g. heavy data augmentation, dropout) push the network away from low-capacity shortcuts.
- **Validate honestly.** Out-of-distribution validation, cross-source validation, deliberate stress tests. The standard "random 80/20 split" fails to catch shortcuts because the shortcut is in both halves.

> [!question]- A friend trains a skin-cancer classifier and gets 95% accuracy. They want to deploy it. What three checks should you make first?
> First, check whether **rulers, scale bars, or surgical markings** appear disproportionately in the malignant images. Dermoscopy datasets often have these as artefacts of clinical practice (suspicious lesions get measured / circled), and networks reliably learn "ruler present → malignant" instead of looking at the lesion. Second, **evaluate on data from a different source** — a different hospital, a different camera, a different country. If accuracy crashes from 95% to 60%, you have a Clever Hans shortcut. Third, **inspect saliency maps** on a sample of correct predictions. If the network is looking at the lesion, great; if it's looking at the ruler, the patient's skin around the lesion, or the clinic's logo, you have a problem. Without these checks, deployment risks a model that works for whoever's data it was trained on and silently mis-diagnoses anyone else.

## Connections

- [[pretext-task]] — chromatic aberration was the canonical pretext-task Clever Hans.
- [[overfitting-nc|overfitting]] — Clever Hans is overfitting to the *wrong feature*. The network fits the training distribution perfectly via a shortcut, then fails when the shortcut is absent. The standard defences against overfitting (regularisation, more data) help, but the deeper defence is dataset design — make sure the shortcut isn't there to be exploited.
- [[data-augmentation]] — augmentations that break the shortcut (random colour drop, intensity perturbation, geometric jitter) are the most direct defence.
- [[self-supervised-learning]] — particularly vulnerable, because pretext targets are easy to design carelessly and easy to "solve" by shortcut.
- [[contrastive-learning]] — SimCLR's heavy colour-distortion augmentation is partly motivated by killing colour-statistics shortcuts in the contrastive objective.
