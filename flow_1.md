# Flow 1: Tracing Language-Specific Neuron Influence Through a Multi-Layer Sparse Autoencoder in Pythia

## Overview

Large language models handle many languages without being trained on curated parallel corpora, and recent work has located part of this ability in a small set of neurons that fire differently depending on the language of the input. Tang et al. named this phenomenon language-specific neurons and showed, using a detection method called language activation probability entropy, that these neurons cluster in the bottom and top layers of models such as LLaMA-2 and BLOOM. A neuron-level account leaves an important question unanswered, however, since once a language-specific neuron fires, it is not clear how that signal travels through the hundreds of dimensions that make up the rest of the network before it reaches the layers where language is expressed in the output. **Sparse autoencoders promise a more structured basis for answering this question, because they decompose a layer's activity into a small number of interpretable latent directions, but no published method yet traces a specific, previously identified neuron through those latent directions across layers.** Answering this question would turn a static list of language-specific neurons into a dynamic account of how language identity actually propagates through a model.

This flow addresses that gap by combining two tools that have not previously been combined for this purpose. It starts from Lawson et al.'s multi-layer sparse autoencoder, a single dictionary trained jointly on the residual stream activations from every layer of a transformer, so that a language-specific neuron's downstream footprint can be tracked as one coherent latent space rather than as several disconnected per-layer dictionaries. For each neuron that language activation probability entropy flags in Pythia, we attribute its causal influence onto the sparse autoencoder's latents in two independent ways. A closed-form path attribution projects the neuron's output weight vector through the intervening layers' weight matrices onto each latent's encoder direction, while a gradient-based attribution backpropagates from latent activations to the neuron, in the same spirit as Shu et al.'s Gradient Sparse Autoencoder backpropagates from model output to latents. **Where the two methods agree on a small subset of latents for a given neuron, we treat that subset as the neuron's likely causal footprint in latent space, and we confirm the footprint causally by patching those latents and checking that the resulting shift in output language matches the shift produced by patching the neuron directly.**

The experiments run on Pythia-70M, which lets the flow reuse Lawson et al.'s released multi-layer sparse autoencoder checkpoint instead of training one from scratch, with Pythia-410M added as a scale check. Because Pythia is trained overwhelmingly on English text, the first experiment simply asks whether language activation probability entropy finds any usable language-specific neurons at this scale and with this pretraining mix at all, before any attribution is attempted. Success looks like convergence, with the weight-based and gradient-based attributions naming an overlapping, small set of latents per neuron, and with patching that set reproducing a substantial share of the output-language shift that patching the neuron itself produces. **The flow is falsified if patching the attributed latents fails to reproduce the neuron's effect, since that would mean the attribution method is finding latents that merely correlate with the neuron rather than sitting causally downstream of it.**

## Research Questions

1. **Neuron replication at small scale.** Does language activation probability entropy identify neurons in Pythia-70M with entropy scores comparable to those Tang et al. report for LLaMA-2 and BLOOM, despite the Pile's English-dominant composition?
2. **Method convergence.** Do the weight-based path attribution and the gradient-based attribution agree, above a chance baseline, on the same small subset of multi-layer sparse autoencoder latents for a given language-specific neuron?
3. **Layer-shift and language identity.** Does the layer at which an implicated latent fires, following Lawson et al.'s finding that individual latents activate at a single layer per token, shift systematically with the language of the input token rather than staying fixed?
4. **Causal validation.** Does patching or ablating the attributed latent subset reproduce a comparable fraction of the output-language shift that patching the underlying neuron directly produces?
5. **Entropy and attribution sharpness.** Does attribution strength scale with a neuron's language activation probability entropy score, such that more language-specific neurons yield sparser, more concentrated latent attributions?
6. **Scale sensitivity.** Does the size of the attributed latent subset grow or shrink between Pythia-70M and Pythia-410M, indicating whether added model capacity concentrates or diffuses the neuron's causal footprint?
7. **Necessity of the gradient check.** Does the cheaper, gradient-free weight-based method alone recover most of the latent subset that the more expensive gradient-based cross-check identifies, or does the gradient check surface latents the closed-form method misses?

## Hypothesis

We predict that the weight-based and gradient-based attributions will converge on overlapping latent subsets, each involving no more than about one percent of the sparse autoencoder's dictionary, for the large majority of language-specific neurons that language activation probability entropy identifies in Pythia. We further predict that patching the attributed latents will reproduce at least half of the output-language shift that patching the underlying neuron produces, which would establish the attribution as tracing a genuine causal pathway rather than a spurious correlation. The hypothesis is falsified if patching the attributed latents fails to reproduce the neuron's effect, or if language activation probability entropy cannot identify neurons with entropy scores distinguishable from a shuffled-language control in Pythia at all.

## Experimental Plan

### Models
The primary model is Pythia-70M, chosen because Lawson et al. released a multi-layer sparse autoencoder checkpoint trained on exactly this model, which removes the need to train a dictionary from scratch before attribution can begin. Pythia-410M serves as a scale check, and a fresh multi-layer sparse autoencoder is trained on it if no released checkpoint covers this size, following Lawson et al.'s published training recipe.

### Baselines
A correlational baseline computes the co-activation statistic, such as Pearson or point-biserial correlation, between each neuron's activation and each latent's activation across a multilingual probe corpus, without using either weights or gradients. A second baseline reproduces Shu et al.'s Gradient Sparse Autoencoder attribution from latent activations to model output, which shows what an attribution method finds when it ignores the neuron side of the pathway entirely.

### Metrics
The primary metric is the Jaccard overlap between the latent subsets that the weight-based and gradient-based methods attribute to a given neuron. A second metric measures the patching effect size on output-language log-probability, comparing the shift produced by patching the attributed latents against the shift produced by patching the neuron directly. A third metric tracks the language activation probability entropy score of each identified neuron before and after the probe corpus is varied, to check the robustness of the initial neuron identification.

### Ablations
One ablation varies model scale, comparing Pythia-70M against Pythia-410M. A second ablation isolates each attribution method, comparing the weight-based method alone, the gradient-based method alone, and the combined method, against the causal patching outcome. A third ablation varies the probe corpus, comparing a naturally occurring multilingual subset of the Pile against synthetic sentences translated into multiple languages, to check whether the attribution depends on the kind of multilingual signal available in Pythia's pretraining data.

---

# Foundational Background

## Toy Models of Superposition
**Authors & Year:** Elhage et al., 2022
**ArXiv/Link:** https://arxiv.org/abs/2209.10652
**Summary:** The authors build a small model in which they can fully describe how a network packs more features into a space than it has dimensions, a phenomenon that they call superposition. They show that superposition is what makes a single neuron respond to several unrelated concepts at once.
**Implication:** Superposition explains why a language-specific neuron cannot be read as one clean signal, so this flow needs a sparse dictionary to name the directions that actually carry language identity. It also motivates RQ2, since two attribution methods that agree on a small latent subset would show that the neuron's signal does unpack into a few dictionary directions.

## Transformer Feed-Forward Layers Are Key-Value Memories
**Authors & Year:** Geva et al., 2021
**ArXiv/Link:** https://arxiv.org/abs/2012.14913
**Summary:** The authors show that a feed-forward layer in a transformer works like a set of key-value memories, where each key matches an input pattern and each value writes a distribution over output tokens. They also show that the output of a feed-forward layer is a sum of these memories, which the residual stream then refines layer by layer.
**Implication:** This paper supplies the mechanical picture that the weight-based path attribution relies on, because it says that a neuron writes into the residual stream through its output weight vector. Without that picture, projecting a neuron's output weight through later layers onto a latent's encoder direction would have no grounding.

## Do Llamas Work in English? On the Latent Language of Multilingual Transformers
**Authors & Year:** Wendler et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.10588
**Summary:** The authors track intermediate embeddings across layers in Llama-2 and find three phases, where the middle layers favour an English version of the correct next token before the last layers move the embedding into a region specific to the input language. They read this as evidence that the abstract concept space of the model sits closer to English than to other languages.
**Implication:** Their three-phase picture predicts that language identity should re-enter the residual stream at particular depths, which is exactly what RQ3 tests when it asks whether the layer at which an implicated latent fires shifts with the language of the input token. It also frames why Pythia, which is even more English-dominant, is a hard test case for RQ1.

## How do Large Language Models Handle Multilingualism?
**Authors & Year:** Zhao et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.18815
**Summary:** The authors propose a multilingual workflow in which a model understands a query, solves the task in English in the middle layers, and then returns to the input language in the final layers. They support it with a detection method that finds the neurons that a given language activates without needing labelled data.
**Implication:** Their workflow gives a concrete prediction about where in the network language identity is carried, which the layer-shift measurement in RQ3 can either confirm or contradict inside the multi-layer sparse autoencoder's latent space. Their label-free neuron detection also serves as an independent point of comparison for the neurons that language activation probability entropy selects in RQ1.

## Unveiling Linguistic Regions in Large Language Models
**Authors & Year:** Zhang et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.14700
**Summary:** The authors find a core region of roughly one percent of a model's parameters whose removal breaks language ability across thirty languages, and they find separate monolingual regions for individual languages. Perturbing even a single parameter along certain dimensions is enough to destroy linguistic competence.
**Implication:** Their one percent figure is the parameter-level counterpart of the prediction in this flow's hypothesis that the attributed latent subset should stay under about one percent of the dictionary. It therefore gives RQ2 and RQ5 an external reference point for what counts as a plausibly sparse footprint.

## Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling
**Authors & Year:** Biderman et al., 2023
**ArXiv/Link:** https://arxiv.org/abs/2304.01373
**Summary:** The authors release sixteen models from 70M to 12B parameters that all see the same public data in the same order, together with 154 checkpoints per model. The suite is built so that researchers can compare behaviour across scale without confounds from differing data or ordering.
**Implication:** This suite is what makes the scale ablation in the experimental plan clean, because Pythia-70M and Pythia-410M differ in size but not in training data or data order. RQ6 depends on that property, since any change in the size of the attributed latent subset can then be read as an effect of capacity rather than of pretraining mix.

## The Pile: An 800GB Dataset of Diverse Text for Language Modeling
**Authors & Year:** Gao et al., 2020
**ArXiv/Link:** https://arxiv.org/abs/2101.00027
**Summary:** The authors assemble an 825 GiB English corpus from twenty-two sources and document its composition in detail. They note the presence of non-English text, but the corpus is built and described as an English dataset.
**Implication:** The English-dominant composition of this corpus is the reason why RQ1 has to be answered before any attribution is attempted, since language activation probability entropy may simply fail to find usable neurons in a model trained this way. The probe-corpus ablation also draws directly on this corpus, because it compares a naturally occurring multilingual subset of it against synthetic translated sentences.

---

# Neuron-Level Identification of Language

## Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models
**Authors & Year:** Tang et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.16438
**Summary:** The authors introduce language activation probability entropy, a detection method that flags neurons whose firing probability is concentrated on one language, and they apply it to LLaMA-2, BLOOM, and Mistral. They find that these neurons sit mostly in the top and bottom layers, and they steer the output language of a model by switching the neurons on or off.
**Implication:** This paper supplies the neuron identification step that the whole flow starts from, so RQ1 is a direct replication test of its detection method at Pythia scale. Its steering result also defines the reference effect for RQ4, because patching the attributed latents has to be compared against patching the neuron itself.

## On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons
**Authors & Year:** Kojima et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2404.02431
**Summary:** The authors find neurons that fire uniquely for each of six languages in decoder-only models, with under five percent overlap between languages and a concentration in the first and last few layers. Changing fewer than one percent of neurons at inference time sharply changes which language the model generates.
**Implication:** Their finding that under one percent of neurons controls output language sets the sparsity scale that RQ5 tests inside latent space rather than neuron space. Their measured overlap between languages also gives the chance baseline that RQ2 needs when it asks whether two attribution methods agree more than they would by accident.

## Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation
**Authors & Year:** Gurgurov et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2507.22608
**Summary:** The authors apply language activation probability entropy across twenty-one languages and three model families, and they find that these neurons cluster in deeper layers, with non-Latin scripts showing stronger specialization. They then steer models by adding and multiplying neuron activations, which works better than simply replacing them.
**Implication:** This is the most careful extension of the detection method that RQ1 replicates, so its reported entropy distributions give the numbers against which Pythia-70M has to be compared. Its finding that related languages share neurons also warns RQ2 that latent subsets attributed to neurons of typologically close languages may overlap for reasons that have nothing to do with attribution quality.

## Unveiling Multilinguality in Transformer Models: Exploring Language Specificity in Feed-Forward Networks
**Authors & Year:** Bhattacharya and Bojar, 2023
**ArXiv/Link:** https://arxiv.org/abs/2310.15552
**Summary:** The authors test whether all feed-forward neurons respond equally to all languages in a bilingual autoregressive model and find that they do not. Layers near the input and the output behave in a more language-specific way than the middle layers do.
**Implication:** Their layer profile is an earlier and independent version of the depth pattern that RQ3 measures, but they measure it on neurons while this flow measures it on multi-layer sparse autoencoder latents. The comparison therefore tells us whether the latent basis preserves or reshapes the depth structure that the neuron basis already shows.

## Sharing Matters: Analysing Neurons Across Languages and Tasks in LLMs
**Authors & Year:** Wang et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.09265
**Summary:** The authors sort neurons into all-shared, partly shared, language-specific, and non-activated groups across nine languages and three tasks. They find that switching off the all-shared neurons hurts performance the most, while activation patterns shift a great deal across tasks and models.
**Implication:** Their four-way split shows that language-specific neurons are only part of the multilingual machinery, which matters for the correlational baseline in the experimental plan. That baseline has to separate latents that track the target neuron from latents that track the shared neurons which fire alongside it.

## Analyzing Individual Neurons in Pre-trained Language Models
**Authors & Year:** Durrani et al., 2020
**ArXiv/Link:** https://arxiv.org/abs/2010.02695
**Summary:** The authors rank individual neurons by how well they predict morphology, syntax, and semantics in pretrained models. They find small neuron subsets that carry each property, and lower-level properties concentrate in fewer neurons than higher-level ones.
**Implication:** This work established the probing style that language activation probability entropy later adapted to language identity, so it sets the methodological baseline that RQ1 inherits. Its finding that simpler properties localize more tightly also predicts the direction of RQ5, where sharper neurons should give sparser latent attributions.

---

# The Proposed Method

## Residual Stream Analysis with Multi-Layer SAEs
**Authors & Year:** Lawson et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2409.04185
**Summary:** The authors train one sparse autoencoder on the residual stream activations from every layer of a transformer, which they call a multi-layer sparse autoencoder. They find that an individual latent is usually active at a single layer for a given token, while the layer at which it becomes active changes across tokens and prompts.
**Implication:** This is the dictionary that the flow attributes onto, and its released Pythia-70M checkpoint is the reason why the experimental plan can skip training a dictionary from scratch. Its single-layer activation finding is the direct object of RQ3, which asks whether that firing layer moves with the language of the input token.

## Group-SAE: Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups
**Authors & Year:** Ghilardi et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2410.21508
**Summary:** The authors group layers whose residual stream representations are similar and train one sparse autoencoder per group rather than one per layer. On Pythia models this speeds up training a great deal while reconstruction quality and downstream performance stay close to per-layer training.
**Implication:** This gives the fallback recipe for the scale check in the experimental plan, since Pythia-410M may have no released multi-layer checkpoint and a fresh dictionary would then have to be trained. Their layer-similarity metric also offers a way to read the RQ6 result, because more similar adjacent layers should spread a neuron's footprint across more latents.

## Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders
**Authors & Year:** Shu et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2505.08080
**Summary:** The authors argue that activated latents do not all matter equally, and they rank latents by output-side gradient information rather than by activation strength alone. Latents with high causal influence turn out to be the effective ones for steering a model.
**Implication:** Their gradient ranking is the template for the gradient-based half of this flow's attribution, except that this flow backpropagates from a latent to a neuron rather than from the output to a latent. Their original output-side method is also the second baseline in the experimental plan, which shows what an attribution finds when it ignores the neuron side of the pathway.

## Sparse Autoencoders Find Highly Interpretable Features in Language Models
**Authors & Year:** Cunningham et al., 2023
**ArXiv/Link:** https://arxiv.org/abs/2309.08600
**Summary:** The authors train sparse autoencoders on language model activations and recover directions that are more interpretable than individual neurons. They argue that this addresses polysemanticity, where a neuron fires in several unrelated contexts.
**Implication:** This paper is the reason why the flow attributes onto latents instead of onto raw residual stream dimensions, since latents are the units that carry a single meaning. It therefore underwrites the premise of RQ2, which assumes that a neuron's influence lands on a small named set of latents rather than smearing over the whole basis.

## Scaling and Evaluating Sparse Autoencoders
**Authors & Year:** Gao et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.04093
**Summary:** The authors use top-k sparse autoencoders to control sparsity directly, which removes the need to tune a reconstruction and sparsity tradeoff by hand, and they find clean scaling laws in dictionary size and sparsity. They also introduce metrics for feature quality based on downstream effect sparsity.
**Implication:** Their downstream effect sparsity metric is close in spirit to what RQ5 measures, since both ask how concentrated a latent's causal reach is. Their scaling laws also give a prior for RQ6, because a larger model paired with a larger dictionary need not produce a larger attributed subset.

## Transcoders Find Interpretable LLM Feature Circuits
**Authors & Year:** Dunefsky et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.11944
**Summary:** The authors replace a dense feed-forward layer with a wider, sparsely activating one so that circuit analysis can pass through the layer using weights rather than activations. The resulting circuits split cleanly into an input-dependent part and an input-invariant part, and the authors train transcoders on models of 120M, 410M, and 1.4B parameters.
**Implication:** Their input-invariant term is the closest published relative of this flow's closed-form path projection, so their construction is the reference for how a weight-only attribution should be built. They also train transcoders at 410M parameters, which matches this flow's scale-check model and gives a usable comparison point for RQ6 and RQ7 even though their smallest reported size does not reach down to 70M.

## Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models
**Authors & Year:** Marks et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2403.19647
**Summary:** The authors build causal subgraphs out of sparse autoencoder features rather than out of neurons or attention heads, using linear approximations to estimate each edge's importance. They then use the resulting circuits to remove features that a human judges irrelevant, which improves how a classifier generalizes.
**Implication:** Their edge-importance estimate is the standard against which the gradient half of this flow's attribution should be judged, so it informs how RQ7 decides whether the gradient check adds anything. Their editing result also shows that a feature-level circuit supports intervention, which is the assumption behind the patching test in RQ4.

---

# Related and Competing Approaches

## Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages
**Authors & Year:** Andrylie et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2507.11230
**Summary:** The authors adapt an activation-probability criterion to sparse autoencoder latents rather than to neurons, and they find language-specific latents mostly in the middle and final layers. These latents affect multilingual performance and work for language identification about as well as fastText.
**Implication:** This is the closest competing account, because it finds language-specific latents directly instead of tracing them back to a neuron, so the novelty of RQ2 rests on the difference between finding a latent and attributing it to a known neuron. Their latents also give an external check on the attributed subsets, since the two sets should overlap if the attribution is doing what it claims.

## Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders
**Authors & Year:** Deng et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2505.05111
**Summary:** The authors define a monolinguality metric for sparse autoencoder features and show that ablating the top-scoring features damages one language while leaving the others nearly untouched. They also find that some languages depend on several features whose joint ablation hurts more than the sum of the individual ablations.
**Implication:** Their monolinguality metric is a ready-made scoring function for checking whether the latents that RQ2 attributes to a neuron are language-specific in their own right. Their synergy result is a direct warning for RQ4, because patching an attributed subset one latent at a time may understate the effect that the subset produces together.

## Language Model Circuits Are Sparse in the Neuron Basis
**Authors & Year:** Arora et al., 2026
**ArXiv/Link:** https://arxiv.org/abs/2601.22594
**Summary:** The authors show that MLP neurons form a feature basis about as sparse as a sparse autoencoder does, and they build a gradient-based attribution pipeline for circuit tracing directly on neurons. Circuits of roughly a hundred neurons are enough to control model behaviour on several tasks.
**Implication:** This is the strongest challenge to the premise of the flow, since it argues that the detour through a latent dictionary buys nothing that neurons do not already give. RQ4 is where the argument gets settled, because patching the attributed latents has to reproduce the neuron's effect in order to justify the extra machinery.

## fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery
**Authors & Year:** Demou et al., 2026
**ArXiv/Link:** https://arxiv.org/abs/2605.09438
**Summary:** The authors extend crosscoders, which are sparse dictionaries shared across layers, with a factorized and masked construction aimed at features that span several layers. Their target is the class of features that emerge over stages of inference or persist in the residual stream.
**Implication:** Crosscoders are the main alternative to the multi-layer sparse autoencoder for tracking a feature across depth, so this paper marks the design choice that the flow is making in RQ3. If latents that fire at a single layer turn out to hide a genuinely cross-layer signal, then a crosscoder rather than a multi-layer sparse autoencoder would be the right dictionary.

## Evolution of SAE Features Across Layers in LLMs
**Authors & Year:** Balcells et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2410.08869
**Summary:** The authors compare features in adjacent layers of separately trained sparse autoencoders and sort them into pass-through, new, and disappearing types. Some of the new features look like logical combinations of features from the preceding layer.
**Implication:** Their pass-through and new categories give a vocabulary for reading the RQ3 result, because a latent whose firing layer moves with the input language should behave unlike a pass-through feature. Their per-layer dictionaries are also the setup that the multi-layer dictionary replaces, so they show what this flow gains by not having to stitch layers together.

## Analyze Feature Flow to Enhance Interpretation and Steering in Language Models
**Authors & Year:** Laptev et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2502.03032
**Summary:** The authors map sparse autoencoder features across consecutive layers with a data-free cosine similarity method and build flow graphs of how a feature persists, changes, or first appears. They then steer generation by amplifying or suppressing features along these graphs.
**Implication:** Their data-free matching is a weight-only method much like the closed-form path projection in this flow, so it is the natural comparison for RQ7, which asks whether the cheap gradient-free method suffices. Their steering result also suggests that a cross-layer footprint is actionable, which is what RQ4 tests through patching.

## How to Use and Interpret Activation Patching
**Authors & Year:** Heimersheim and Nanda, 2024
**ArXiv/Link:** https://arxiv.org/abs/2404.15255
**Summary:** The authors collect practical advice on how to run activation patching and how to read what it produces. They pay particular attention to the choice of metric and to the traps that come with it.
**Implication:** RQ4 turns on a patching comparison between attributed latents and the neuron itself, so their advice on metric choice directly shapes the output-language log-probability measure in the experimental plan. Their warnings also matter for the falsification condition, since a null patching result must not simply be an artefact of a badly chosen metric.

---

# Evaluation Tools and Benchmarks

## SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability
**Authors & Year:** Karvonen et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2503.09532
**Summary:** The authors assemble a suite of evaluations for sparse autoencoders that goes beyond reconstruction loss and sparsity, covering interpretability, feature disentanglement, and downstream use. They show that rankings of autoencoders change depending on which evaluation one looks at.
**Implication:** Their suite gives the quality bar that the Pythia-410M dictionary must clear before any attribution result from RQ6 can be trusted. It also protects the scale ablation, because a difference in attributed subset size between the two models could otherwise come from a difference in dictionary quality.

## Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control
**Authors & Year:** Makelov et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2405.08366
**Summary:** The authors compare unsupervised sparse autoencoder dictionaries against supervised dictionaries on a task where the relevant features are known. They report two failure modes, one where a causally important feature is hidden behind a larger one and one where a single feature splits into many smaller ones.
**Implication:** Feature occlusion is the specific way in which RQ4 could fail without the attribution being wrong, since a latent that genuinely sits downstream of a neuron may be masked by a stronger neighbour during patching. Feature splitting likewise inflates the attributed subset size that RQ2 and RQ5 measure.

## Automatically Interpreting Millions of Features in Large Language Models
**Authors & Year:** Paulo et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2410.13928
**Summary:** The authors build an open pipeline that writes and scores natural language explanations for sparse autoencoder latents, including a score based on the effect of intervening on a latent. Their large-scale comparison confirms that latents are more interpretable than neurons, even when neurons are sparsified afterwards.
**Implication:** Their pipeline is how the flow can label the latents that RQ2 attributes to a neuron, which turns a numeric overlap into a claim about language identity. Their finding that autoencoders trained on nearby layers are highly similar also bears on RQ3, since it predicts weak layer separation in the middle of the model.

## Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
**Authors & Year:** Kantamneni et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2502.16681
**Summary:** The authors test whether sparse autoencoders beat simple baselines on real probing tasks under data scarcity, class imbalance, label noise, and covariate shift. They find no setting in which autoencoder-based methods reliably beat the baselines.
**Implication:** Their result is the reason why the experimental plan includes a plain correlational co-activation baseline, since an attribution method has to beat a cheap statistic before it earns its cost. It sharpens RQ7 as well, because the same argument applies to whether the expensive gradient pass earns its cost over the closed-form projection.

## Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders
**Authors & Year:** Rajamanoharan et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2407.14435
**Summary:** The authors replace the usual activation function in a sparse autoencoder with a discontinuous one and train the sparsity term directly rather than through a proxy. This gives better reconstruction at a fixed sparsity level without costing interpretability.
**Implication:** Their reconstruction and sparsity frontier is the tradeoff that the Pythia-410M dictionary has to be placed on before RQ6 compares subset sizes across scale. A dictionary that reconstructs poorly would push a neuron's influence onto more latents for reasons that have nothing to do with model capacity.

## No Language Left Behind: Scaling Human-Centered Machine Translation
**Authors & Year:** NLLB Team, 2022
**ArXiv/Link:** https://arxiv.org/abs/2207.04672
**Summary:** The team builds translation models and data for over two hundred languages and releases Flores-200, a human-translated benchmark that is parallel across all of them. They evaluate more than forty thousand translation directions against it.
**Implication:** Flores-200 supplies the parallel sentences that the probe-corpus ablation needs, since comparing a naturally occurring Pile subset against synthetic translations requires translations that mean the same thing in every language. Holding meaning fixed is what lets RQ3 read a layer shift as an effect of language rather than of content.

## The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants
**Authors & Year:** Bandarkar et al., 2023
**ArXiv/Link:** https://arxiv.org/abs/2308.16884
**Summary:** The authors build a fully parallel multiple-choice reading comprehension set across 122 language variants on top of Flores-200 passages. Because it is parallel, model performance can be compared directly across all of them.
**Implication:** This benchmark provides a behavioural check on whether the neurons that RQ1 identifies actually matter for a language, rather than merely firing on it. It also gives a way to confirm that patching the attributed latents in RQ4 changes output language without wrecking comprehension.

---

# Theoretical Grounding

## The Linear Representation Hypothesis and the Geometry of Large Language Models
**Authors & Year:** Park et al., 2023
**ArXiv/Link:** https://arxiv.org/abs/2311.03658
**Summary:** The authors give two precise definitions of what it means for a concept to be represented linearly, one in output space and one in input space, and they connect these to linear probing and to steering. They then identify an inner product under which all the usual notions of linear representation agree.
**Implication:** The closed-form path projection in this flow is a sequence of linear maps followed by an inner product with an encoder direction, so its validity rests on the linear representation claim that this paper makes precise. Their causal inner product also warns that a plain dot product may be the wrong similarity, which is a concrete threat to the weight-based half of RQ2.

## The Geometry of Categorical and Hierarchical Concepts in Large Language Models
**Authors & Year:** Park et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.01506
**Summary:** The authors extend the linear representation formalism from binary contrasts to features that have no natural opposite, and they show that categorical concepts appear as polytopes in representation space. They prove a relationship between how concepts nest and how their representations sit geometrically.
**Implication:** Language identity is a categorical concept with many values rather than a binary contrast, so their extension is what licenses treating each language as a direction in the first place. Their hierarchy result also predicts that typologically related languages should have geometrically related directions, which bears on how RQ2 chooses its chance baseline.

## Not All Language Model Features Are One-Dimensionally Linear
**Authors & Year:** Engels et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2405.14860
**Summary:** The authors define what makes a multi-dimensional feature irreducible and then find such features automatically with sparse autoencoders, including circular representations of days and months. Intervention experiments show that these circles are the units the model actually computes with.
**Implication:** If language identity turns out to be irreducibly multi-dimensional, then a per-latent attribution would split one signal across several latents and inflate the subset size that RQ2 and RQ5 measure. Their method therefore gives a concrete diagnostic for reading a large attributed subset as structure rather than as failure.

## Sparse Coding and Autoencoders
**Authors & Year:** Rangamani et al., 2017
**ArXiv/Link:** https://arxiv.org/abs/1708.03735
**Summary:** The authors analyse when an autoencoder trained on sparse data actually recovers the dictionary that generated it. They give conditions under which the autoencoder's learned weights sit near the true dictionary.
**Implication:** Recovery conditions are what justify reading a latent's encoder direction as a real direction in the model rather than as an artefact of training, which the weight-based projection in RQ2 assumes. Where those conditions fail, the closed-form and gradient methods can disagree for reasons that have nothing to do with the neuron.

## Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias
**Authors & Year:** Vig et al., 2020
**ArXiv/Link:** https://arxiv.org/abs/2004.12265
**Summary:** The authors bring causal mediation analysis into model interpretation, treating components as mediators through which an effect flows from input to output. Applied to gender bias, they find effects that are sparse, that combine across components, and that split into direct and indirect parts.
**Implication:** This is the theory behind the whole design, because attributing a neuron's influence onto latents is a claim that the latents mediate the neuron's effect on output language. Their split between direct and indirect effects is exactly what RQ4 has to separate when it compares patching the latents against patching the neuron.

## Locating and Editing Factual Associations in GPT
**Authors & Year:** Meng et al., 2022
**ArXiv/Link:** https://arxiv.org/abs/2202.05262
**Summary:** The authors use a causal intervention to find the activations that decide a factual prediction, and they trace those to middle-layer feed-forward modules. They then edit the responsible weights directly and change the fact that the model recalls.
**Implication:** Their causal tracing procedure is the template for the patching comparison in RQ4, since both localize an effect by restoring activations from a clean run into a corrupted one. Their weight edit also shows that a localized cause can be manipulated at the weight level, which supports the closed-form projection's assumption that weights carry the pathway.

## Towards Best Practices of Activation Patching in Language Models: Metrics and Methods
**Authors & Year:** Zhang and Nanda, 2023
**ArXiv/Link:** https://arxiv.org/abs/2309.16042
**Summary:** The authors run activation patching under many combinations of metric and corruption method and show that the choice changes the localization one ends up reporting. They then argue for particular choices and give recommendations.
**Implication:** RQ4 compares two patching effects against each other, so both have to use the same metric and the same corruption scheme or the comparison means nothing. Their recommendations therefore fix how the output-language log-probability metric in the experimental plan should be set up.

---

# Downstream Application and Validation

## Neural FOXP2: Language Specific Neuron Steering for Targeted Language Improvement in LLMs
**Authors & Year:** Saha et al., 2026
**ArXiv/Link:** https://arxiv.org/abs/2602.00945
**Summary:** The authors localize language neurons by tracing top-ranked sparse autoencoder features back to the units that contribute to them, then find a low-rank steering subspace by singular value decomposition of English-to-target activation differences. Applying a signed sparse shift in low and middle layers makes Hindi or Spanish the model's default language.
**Implication:** Their localization runs in the opposite direction to this flow, since they go from features to neurons while this flow goes from a neuron to latents, which makes their pipeline a useful sanity check on RQ2. Their steering result also shows that a small language-neuron set can control output language, which is the effect that RQ4 tries to reproduce through the latents instead.

## Steering Language Models With Activation Engineering
**Authors & Year:** Turner et al., 2023
**ArXiv/Link:** https://arxiv.org/abs/2308.10248
**Summary:** The authors compute a steering vector by contrasting activations on a pair of prompts and add it during the forward pass to control sentiment or topic. The method needs no optimization and works from a single pair of examples.
**Implication:** Activation addition is the cheapest possible way to shift output language, so it is the control that shows whether the attributed latents in RQ4 offer anything beyond a crude contrast vector. If a contrast vector matches the patching effect, then the attribution has not earned its complexity.

## Improving Steering Vectors by Targeting Sparse Autoencoder Features
**Authors & Year:** Chalnev et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2411.02193
**Summary:** The authors use sparse autoencoders to measure what a steering vector actually does, then build a steering method that targets chosen latents while keeping side effects small. Their method balances the intended effect against coherence better than contrastive steering or direct latent steering.
**Implication:** Their measurement technique is a way to check that patching the attributed latent subset in RQ4 changes output language without dragging other behaviour along with it. It also gives a practical route from the attribution result to a usable language control, which is the point of tracing the pathway in the first place.

## Understanding and Mitigating Language Confusion in LLMs
**Authors & Year:** Marchisio et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.20052
**Summary:** The authors build a benchmark covering fifteen languages for the failure in which a model does not answer in the language the user wants. They find that base models and English-centric instruction-tuned models confuse languages most, and that few-shot prompting and multilingual tuning only partly fix it.
**Implication:** Their benchmark defines the output-language behaviour that RQ4 measures, so it supplies both the prompts and the scoring for the patching comparison. Because base models fare worst on it, it is also well matched to Pythia, which is a base model with no instruction tuning.

## SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs
**Authors & Year:** Deng et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2507.14894
**Summary:** The authors link unwanted language switching to over-activation of language-specific sparse autoencoder features and then finetune the model to keep those features at appropriate pre-activation values. This cuts code-switching by more than half while multilingual benchmark scores hold.
**Implication:** This is the clearest application for whatever latent subset RQ2 identifies, because a subset that genuinely mediates language identity should be the right target for such a constraint. Their pre-activation control also gives an intervention that is gentler than patching, which offers a second way to confirm the RQ4 result.

## Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for Large Language Models
**Authors & Year:** Zhong et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2510.07213
**Summary:** The authors find that a small number of residual stream dimensions carry language identity in English-centric models and use them to control which language the model outputs. Their control is cheap because it touches so few dimensions.
**Implication:** Their sparse dimensions are a competing localization of the same signal that this flow traces, but they work in the raw residual basis rather than in a sparse dictionary. Comparing their dimensions against the latents attributed in RQ2 shows whether the dictionary adds resolution or merely renames what the raw basis already exposes.

## Uncovering Cross-Linguistic Disparities in LLMs using Sparse Autoencoders
**Authors & Year:** Sin Jing Xuan et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2507.18918
**Summary:** The authors compare sparse autoencoder activation patterns across ten languages in Gemma-2-2B and find systematically weaker activations for low-resource languages. Fine-tuning that targets these activations raises performance on those languages while English holds steady.
**Implication:** Their activation gap predicts a difficulty for RQ1, since a model as English-dominant as Pythia may show weak activations for most languages before entropy is even computed. Their targeted fine-tuning also shows a use for the attributed subset, because strengthening the latents that a language-specific neuron feeds is a natural follow-up to RQ4.
