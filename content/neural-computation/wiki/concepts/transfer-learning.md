---
name: transfer-learning
description: Reusing the weights of a network pre-trained on a large source dataset (e.g. ImageNet) as the starting point for training on a smaller target task. Standard practice in modern computer vision; lets deep models work in domains where collecting millions of labelled examples is impossible.
type: concept
sources:
  - raw/week-05/w05-l13-transcript.txt
  - raw/week-05/w05-slides.pdf
  - raw/week-05/w5-notion.zip
status: stable
updated: 2026-04-26
---

*Deep CNNs need millions of labelled images to learn from scratch. Most real problems — rare diseases, specialised industrial inspection, niche scientific imagery — have hundreds, not millions. Transfer learning sidesteps this by starting from the weights of a network already trained on a large general dataset (typically ImageNet) and adapting them to the new task. The lower convolution layers, which learn universal visual features like edges and textures, work just as well on the new domain.*

> [!tip] TIP — Piano to organ
> The intuitive picture: imagine learning to play the electric organ. Starting from scratch, you'd have to learn what a musical note is, how to read rhythm, how to coordinate fingers — years of work. But if you already know the **piano**, almost all of that transfers: muscle memory, sight-reading, chord theory. You only need to learn what's *new* — the pedals and stops. Transfer learning is the same: the pre-trained network is your piano-playing self; your target task is the organ. Most of the skills carry over; you only retrain the parts that genuinely differ.

## The motivation: data hunger

A network like VGG16 or ResNet-50 has tens of millions of parameters. Training one from scratch requires:

- A large dataset — millions of labelled examples is the rough requirement for ImageNet-scale models.
- Significant compute — days or weeks on multiple GPUs.
- Stable training — careful initialisation, schedule, and regularisation to avoid overfitting.

If you have 500 photos of a rare skin lesion, training a deep CNN from scratch will overfit catastrophically — the network will memorise the training set in a few epochs and fail on new patients. The model has the capacity to memorise; what it lacks is the *evidence* needed to learn general features.

Transfer learning bypasses this by handing the network a generic visual feature extractor that someone else has already trained, on millions of examples, at significant compute cost. You skip the hardest part of training and only adapt the final task-specific layers.

## The mechanics

![[transfer-learning-pipeline.png]]

The pipeline:

1. **Source training (someone else, once).** Train a deep CNN on a large general dataset like ImageNet (~1.2M images, 1000 classes). The network learns hierarchical visual features — early layers detect edges, gradients, dots; middle layers detect parts and textures; late layers compose them into objects.

2. **Take the pre-trained model.** Strip off the final fully-connected layer (or layers) — those were trained for ImageNet's 1000-class output, which has nothing to do with your task.

3. **Attach a new head.** Replace the removed layers with a fresh classifier sized for *your* task — e.g., a single FC layer with 2 outputs for benign-vs-malignant skin lesion.

4. **Decide what to train.** Two main strategies:
   - **Feature extraction:** *freeze* the convolution layers (don't update their weights), train only the new head. Fastest and most data-efficient; treats the pre-trained CNN as a fixed feature extractor. Good when target data is very limited.
   - **Fine-tuning:** unfreeze some or all of the pre-trained layers and let them update too, usually with a smaller learning rate than the new head. Slower, needs more data, but can adapt the features to the target domain. Standard practice when you have at least a few thousand target examples.

5. **Train.** Run gradient descent as usual on the target dataset. Convergence is typically *much* faster than from-scratch training because most of the work is already done.

## Why it works: universal early features

The empirical observation that makes transfer learning effective: **early conv layers learn features that are useful for almost any visual task.** Gabor-like edge detectors, oriented gradients, simple textures, dot detectors. These are properties of natural images in general, not of ImageNet's 1000 categories specifically.

A curve is a curve whether it's the edge of a dog (ImageNet) or the boundary of a tumour (medical imaging). The layers that detect curves don't need to be retrained. Only the *task-specific composition* of these features into class predictions — which lives in the final FC layers — needs to change.

The deeper into the network you go, the more task-specific the features become. So the standard practice is:

- **Freeze early layers always** — they're transferable and don't need adjusting.
- **Unfreeze later layers selectively** — they're more task-specific and may benefit from adaptation.
- **Always replace the very last classifier layer** — it's hardwired to the source task's output format.

> [!tip] TIP — When fine-tuning helps versus when feature-extraction is enough
> Rough rule: the *more similar* the target task is to the source task, the less fine-tuning you need. Classifying a new species of bird from ImageNet weights → feature extraction is often sufficient. Classifying medical microscopy from ImageNet weights → fine-tuning helps because the input statistics are quite different from natural photographs. Classifying X-rays → fine-tune more layers, since the visual features really do differ.

## When transfer learning fails

It works when the source and target domains share visual structure. It fails or underperforms when:

- **Target images are radically different** from natural photographs — pure scientific imagery (electron microscopy, satellite imagery in non-visible bands, audio spectrograms) may not benefit much from ImageNet pre-training.
- **Target task is non-visual** — transferring an image classifier's weights to a tabular regression problem makes no sense.
- **Source pre-training was on a too-narrow domain** — a network pre-trained only on cat photos would transfer poorly to general image classification.

Pre-training on the largest, most diverse dataset available (ImageNet, or larger ones like JFT-300M) gives the broadest transferable features.

## Beyond computer vision

The pattern generalises:

- **Language models** — fine-tuning a pre-trained LLM (BERT, GPT, etc.) on a specific task is the dominant paradigm in NLP. Same idea: the source model has learned general linguistic features; fine-tuning teaches it the task-specific mapping.
- **Speech recognition, protein folding, etc.** — anywhere a large general dataset exists and a smaller task-specific dataset is the bottleneck.

The phrase "standing on the shoulders of giants" is widely used; you download a model that has already seen far more of the world than you could afford to show it, and teach it your specific trick.

## Related

- [[convolutional-neural-network]] — ImageNet-trained CNNs are the canonical pre-trained models
- [[overfitting]] — small datasets cause severe overfitting; transfer learning is one of the strongest mitigations
- [[data-augmentation]] — usually combined with transfer learning when target data is limited
- [[regularization]] — fine-tuning with a small learning rate is a form of implicit regularisation against catastrophic forgetting
- [[u-net]] — segmentation networks often use ImageNet-pretrained encoders as the contracting path

## Active Recall

> [!question]- Why is training a deep CNN from scratch usually a bad idea when you have only ~500 labelled images, even with regularisation and data augmentation?
> The model has tens of millions of parameters and the dataset has 500 examples. Even with strong regularisation, the ratio is wrong by orders of magnitude — the network either underfits (regularisation cripples it) or overfits (regularisation is too weak). The learnable capacity simply can't be calibrated by 500 examples to find general features rather than memorise specifics. Transfer learning addresses the root cause: import the feature extractor from a model trained on millions of examples, so only the small task-specific head has to be calibrated by your 500.

> [!question]- What does "freezing" a layer mean in transfer learning, and why might you do it?
> Freezing means setting that layer's weights as non-trainable for the optimizer — gradients are still computed and passed through the layer (so deeper trainable layers can update), but the frozen layer's own weights stay at their pre-trained values. You freeze early layers because (a) their features (edges, textures) are universal and don't need adapting, and (b) preventing them from changing reduces the number of trainable parameters, which mitigates overfitting on small target datasets. As target dataset size grows, you progressively unfreeze more layers.

> [!question]- Why do you always replace the *final* classifier layer of a pre-trained model when transferring to a new task, even if you don't change anything else?
> The final layer's output dimension is fixed to the source task's number of classes (1000 for ImageNet) and its weights are calibrated for that exact mapping. Your target task almost certainly has a different number of classes (or is a regression problem). The final layer is the most task-specific part of the network — keeping its weights would be meaningless. The standard pattern: drop the final FC, attach a new one with the right output size, train at minimum that new layer.

> [!question]- A friend pre-trains a CNN on cat-vs-dog classification (50K images), then transfers to a 1000-class natural-image classification task with ImageNet. Will this work? Why or why not?
> Probably poorly. The cat-vs-dog source task is too narrow — the network only needed to learn features that distinguish those two categories, not the broader range of visual features that 1000-class classification requires. Pre-training is most useful when the source task is *broader* than the target, exposing the network to more diverse features than it would otherwise need. ImageNet → cat-vs-dog works (broad → narrow); cat-vs-dog → ImageNet doesn't (narrow → broad). The general rule: pre-train on the largest, most diverse dataset available, then transfer to the more specific task.

> [!question]- Compare "feature extraction" and "fine-tuning" as transfer learning strategies. What's the difference, and how do you choose between them?
> Feature extraction: freeze the entire pre-trained network, train only the new head on top. Fastest, fewest parameters to update, most resistant to overfitting on small target sets. Treats the pre-trained CNN as a fixed feature mapping. Fine-tuning: unfreeze some or all pre-trained layers and let them update during training, usually with a smaller learning rate than the new head. More flexible, lets the network adapt features to the target domain, but needs more data and more compute. Choose feature extraction when target data is very small (~hundreds) or very similar to source; choose fine-tuning when target data is moderate (~thousands+) or differs substantially from source.
