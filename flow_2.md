# Flow 2: Reconciling Neuron-Level and Feature-Level Language Specificity in BLOOM Through Attributed Sparse Autoencoder Latents

## Overview

Two recent lines of work on multilingual language models have developed in parallel without meeting. One line follows Tang et al. in finding that a small set of individual neurons drives a model's handling of a given language, concentrated at specific layers depending on the model family, and later work by Gurgurov et al. shows that this concentration point actually shifts across model families rather than sitting at a fixed depth. A second line, represented by Andrylie et al. and Deng et al., shows that sparse autoencoder features can also be language-specific, using methods that measure language-specificity directly from a feature's own activation statistics rather than from the neurons that feed into it. **These two lines report language-specific units at different layers and describe them with different methods, and neither line asks whether the neurons the first line identifies are the actual causal source of the features the second line identifies.** Without that link, it remains unclear whether the neuron-level and feature-level accounts of multilingual processing describe the same underlying mechanism or two different ones that merely happen to correlate with language.

This flow closes that gap in BLOOM-1.7B, chosen because it carries genuine multilingual pretraining rather than the incidental multilingual exposure of an English-centric model. Neither Andrylie et al. nor Deng et al. tested BLOOM, since the first studies Llama 3.2 1B and Gemma 2 and the second studies Gemma 2 2B, Gemma 2 9B, and Llama 3.1 8B, so this flow applies their published methods to BLOOM itself rather than reusing an existing result. We first identify language-specific neurons with language activation probability entropy, following Tang et al., and separately train Lawson et al.'s multi-layer sparse autoencoder on BLOOM's residual stream across every layer. We then attribute each neuron's causal influence onto the autoencoder's latents using the same weight-based path projection and gradient-based backpropagation as the companion Pythia flow, and, independently, we run Andrylie et al.'s activation-probability method and Deng et al.'s monolinguality metric directly on BLOOM's own sparse autoencoder features to produce two comparison feature sets of our own. **A genuine causal link between the neuron-level and feature-level accounts would show up as our attributed latents landing disproportionately inside the feature sets that these two independently reproduced methods flag as language-specific in BLOOM, well beyond the overlap expected between two random subsets of the same size.**

Validation follows the same causal step as the companion flow, since patching the attributed latents should reproduce most of the output-language shift that patching the underlying neuron produces, and steering through the latents should approach the effectiveness of Gurgurov et al.'s direct neuron-level language arithmetics on the same prompts. BLOOM-560M serves as a compute fallback and a scale check, because a smaller dictionary and a less capable underlying model may separate language-specific features less cleanly than BLOOM-1.7B does. **Success means that three independently derived notions of language-specificity, namely the original neurons, the two feature-level methods, and our attribution, converge on the same latents, while a failure to converge, or a steering effect that appears only at the neuron level and not at the latent level, would falsify the claim that the neuron and feature accounts share a common causal pathway.**

## Research Questions

1. **Cross-method overlap.** Do the latents that our weight-based and gradient-based attribution identify for a given neuron overlap with the feature sets that we obtain by running Andrylie et al.'s SAE-LAPE method and Deng et al.'s monolinguality metric on BLOOM's own sparse autoencoder features, at a rate significantly above the overlap expected from a random latent subset of the same size?
2. **Layer reconciliation.** Where in the multi-layer sparse autoencoder's layer distribution do the attributed latents fire, and does this location sit closer to the neuron-level concentration that Tang et al. and Gurgurov et al. report or to the middle-to-final concentration that Andrylie et al. report for feature-level units?
3. **Causal validation at scale.** Does patching the attributed latents reproduce the neuron-ablation output-language effect at a magnitude comparable to the companion Pythia flow's result, despite BLOOM's much stronger multilingual pretraining?
4. **Typological proximity.** Do languages that Gurgurov et al. find share overlapping neurons because of typological proximity also share overlapping attributed latents in the multi-layer sparse autoencoder?
5. **Scale sensitivity.** Does attribution strength or attributed-latent-set size differ between BLOOM-1.7B and BLOOM-560M, and does any difference track the smaller model's previously reported weaker feature separability?
6. **Latent-level steering.** Can steering through the attributed latents alone, without touching the original neurons, shift output language as effectively as Gurgurov et al.'s direct neuron-level language arithmetics on matched prompts?
7. **Resource-level robustness.** Does the three-way convergence between neuron-level, feature-level, and attribution-based language-specificity hold uniformly across high-resource and low-resource languages in BLOOM's training mix, or does it weaken for low-resource languages specifically?

## Hypothesis

We predict that the attributed latent sets will overlap with both of our BLOOM reproductions of Andrylie et al.'s and Deng et al.'s language-specific feature sets at a rate well above chance, and that patch-based steering of the attributed latents will reproduce most of the effect that Gurgurov et al.'s neuron-level steering achieves on the same prompts. This pattern would show that the neuron-level and feature-level accounts of language specificity describe a shared causal pathway rather than two coincidentally correlated phenomena. The hypothesis is falsified if the overlap with the two reference feature sets is statistically indistinguishable from chance, or if latent-based steering fails to shift output language on prompts where neuron-level steering succeeds.

## Experimental Plan

### Models
The primary model is BLOOM-1.7B, chosen for its genuine multilingual pretraining rather than for any existing published result, since neither Andrylie et al. nor Deng et al. tested BLOOM. BLOOM-560M serves as a compute-constrained fallback and a scale ablation, to be used if activation caching or multi-layer sparse autoencoder training on BLOOM-1.7B exceeds the available single-GPU budget.

### Baselines
A correlational baseline computes co-activation statistics between each neuron and each latent across a multilingual probe corpus without using weights or gradients. Because neither Andrylie et al. nor Deng et al. published feature indices or checkpoints for any model, we apply their published methods, SAE-LAPE's activation-probability criterion and the monolinguality metric, directly to BLOOM's own sparse autoencoder features ourselves, and the two resulting feature sets serve as external reference baselines rather than methods we compete against, since the central question is overlap with them, not superiority over them.

### Metrics
The primary metric is a hypergeometric enrichment or Jaccard overlap score between our attributed latent sets and each of the two reference feature sets. A second metric measures the patch-based output-language shift, compared between attributed latents and the underlying neurons. A third metric measures steering success rate across the twenty-one-language set that Gurgurov et al. use, comparing neuron-level and latent-level steering on matched prompts.

### Ablations
One ablation varies model scale, comparing BLOOM-1.7B against BLOOM-560M. A second ablation isolates each attribution method, comparing weight-based, gradient-based, and combined attribution against the overlap and patching outcomes. A third ablation splits languages by resource level, comparing high-resource languages such as English, French, and Spanish against low-resource languages in BLOOM's training mix, to test whether the three-way convergence weakens where BLOOM's own multilingual signal is weaker.

---

# Foundational Background

## BLOOM: A 176B-Parameter Open-Access Multilingual Language Model
**Authors & Year:** BigScience Workshop et al., 2022
**ArXiv/Link:** https://arxiv.org/abs/2211.05100
**Summary:** BigScience trains BLOOM on 46 natural languages and 13 programming languages, and it releases every model size from 560M to 176B parameters under an open license. The paper documents the architecture, the language mix, and the share of training tokens that each language receives.
**Implication:** BLOOM-1.7B and BLOOM-560M are the two models that this flow studies, so this paper supplies the token shares. Those shares decide which languages count as high-resource and which count as low-resource in the RQ7 ablation.

## Do Llamas Work in English? On the Latent Language of Multilingual Transformers
**Authors & Year:** Wendler et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.10588
**Summary:** The authors track intermediate embeddings through Llama-2 and find three phases, where the middle layers favour English tokens before the final layers move back into an input-language region. They read this as evidence that the abstract concept space sits closer to English than to other languages.
**Implication:** The layer-wise story here predicts where output-language information re-enters the residual stream. It therefore gives RQ2 a prior about which layers of the multi-layer sparse autoencoder should hold the attributed latents.

## Neurons in Large Language Models: Dead, N-gram, Positional
**Authors & Year:** Voita et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2309.04827
**Summary:** The authors show that most feed-forward neurons in large OPT models never activate, while many of the live ones fire on single tokens or n-grams rather than on abstract concepts. They also find neurons whose activation depends on position rather than on content.
**Implication:** A neuron that entropy alone flags as language-specific may really be a token detector for a script or a frequent word. This flow therefore needs the attribution step and the patching check in RQ3 to tell the two cases apart.

## Unsupervised Cross-lingual Representation Learning at Scale
**Authors & Year:** Conneau et al., 2020
**ArXiv/Link:** https://arxiv.org/abs/1911.02116
**Summary:** XLM-R shows large cross-lingual transfer gains from scale, but it also names the curse of multilinguality, where adding languages to a fixed-capacity model eventually hurts each one. The paper measures the trade-off between positive transfer and capacity dilution directly.
**Implication:** Capacity dilution is the reason why RQ5 expects BLOOM-560M to separate language-specific features less cleanly than BLOOM-1.7B does. The smaller model spreads the same 46 languages over far fewer parameters.

## How multilingual is Multilingual BERT?
**Authors & Year:** Pires et al., 2019
**ArXiv/Link:** https://arxiv.org/abs/1906.01502
**Summary:** The authors probe mBERT and find that zero-shot transfer works even across scripts, but it works best between typologically similar languages. They conclude that mBERT builds shared multilingual representations with systematic gaps for certain language pairs.
**Implication:** This is the earliest clear statement that typological similarity governs shared representation. RQ4 tests that same assumption at the latent level rather than at the whole-model level.

## Investigating the Translation Performance of a Large Multilingual Language Model: the Case of BLOOM
**Authors & Year:** Bawden and Yvon, 2023
**ArXiv/Link:** https://arxiv.org/abs/2303.01911
**Summary:** The authors evaluate BLOOM's translation quality across many language pairs and find that quality tracks how much of each language appears in the training mix. Performance drops sharply for languages with a small token share.
**Implication:** This gives the flow an external, per-language measure of BLOOM's own competence. RQ7 can then correlate convergence strength against that measure instead of relying on token counts alone.

---

# Direct Baseline and Predecessor Methods

## Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models
**Authors & Year:** Tang et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.16438
**Summary:** The authors introduce language activation probability entropy, and this score flags a neuron as language-specific if it activates for one language far more often than for the others. They find such neurons concentrated in the top and bottom layers of LLaMA-2, BLOOM, and Mistral, and they steer output language by switching the neurons on or off.
**Implication:** This method defines the neuron set that every later step of the flow attributes from. Its BLOOM layer positions also form one pole of the disagreement that RQ2 tries to settle.

## Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation
**Authors & Year:** Gurgurov et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2507.22608
**Summary:** The authors apply language activation probability entropy to four models across twenty-one languages, and they find the neurons clustered in deeper layers rather than at the top and bottom. They then steer output language by adding and multiplying activations, and they report that related languages share overlapping neurons.
**Implication:** Their twenty-one-language steering set is the prompt set for the flow's third metric, and their deeper-layer result is the second pole of the layer disagreement in RQ2. Their neuron overlap by typological proximity is the exact finding that RQ4 re-tests on attributed latents.

## On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons
**Authors & Year:** Kojima et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2404.02431
**Summary:** The authors locate language-specific neurons in several decoder models and show that fewer than one percent of them control the output language. Turning those neurons on or off changes the language while leaving fluency mostly intact.
**Implication:** This independently confirms that neuron-level language control is sparse. It therefore sets the size expectation for the neuron sets that the flow attributes onto latents, and for the effect size that RQ3 should see under patching.

## Unveiling Linguistic Regions in Large Language Models
**Authors & Year:** Zhang et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.14700
**Summary:** The authors find a core region of roughly one percent of parameters whose removal destroys linguistic competence across thirty languages. They also find separate monolingual regions, and disrupting one of them harms only the matching language.
**Implication:** The split between a shared core and per-language regions predicts that some attributed latents will be language-general rather than language-specific. RQ1 therefore needs a hypergeometric enrichment test rather than a raw overlap count.

## Sharing Matters: Analysing Neurons Across Languages and Tasks in LLMs
**Authors & Year:** Wang et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.09265
**Summary:** The authors sort neurons into all-shared, partially shared, language-specific, and non-activated groups, and they find that the sharing pattern depends heavily on the task. All-shared neurons turn out to matter most for producing correct answers.
**Implication:** Sharing depends on the probe task, so the flow's multilingual probe corpus must stay task-neutral. Otherwise the attributed latent sets in RQ1 would reflect the probe design rather than the language.

## How do Large Language Models Handle Multilingualism?
**Authors & Year:** Zhao et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2402.18815
**Summary:** The authors propose a working picture where a model understands the input in its own language, reasons in English in the middle layers, and generates back in the target language at the end. They support this with layer-wise language-ratio measurements and with neuron detection.
**Implication:** This layer-role picture gives RQ2 a sharp prediction, since output-language latents should sit late in the multi-layer sparse autoencoder's layer distribution if generation really is a final-layer step.

## Language-specific Neurons Do Not Facilitate Cross-Lingual Transfer
**Authors & Year:** Mondal et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2503.17456
**Summary:** The authors fine-tune only the neurons that entropy-based methods flag as language-specific, and they find no gain on cross-lingual tasks in low-resource languages. They report this as a negative result about the usefulness of those neurons.
**Implication:** This is the strongest existing challenge to the assumption behind RQ7. The flow's low-resource ablation therefore has to run on the same kind of low-resource languages and report whether the three-way convergence survives there.

---

# The Proposed Method

## Residual Stream Analysis with Multi-Layer SAEs
**Authors & Year:** Lawson et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2409.04185
**Summary:** The authors train a single sparse autoencoder on residual stream vectors from every layer of a transformer, and they find that individual latents usually fire at one layer for a given token even though that layer shifts across tokens. They quantify the effect with a distribution over layers and its variance.
**Implication:** This is the dictionary that the flow trains fresh on BLOOM, since no released checkpoint exists. Its per-latent layer distribution is also the exact measurement that RQ2 uses to place attributed latents against the neuron-level and feature-level layer claims.

## Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders
**Authors & Year:** Shu et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2505.08080
**Summary:** The authors argue that active latents do not all contribute equally to the output, and they rank latents by output-side gradient rather than by input activation. Steering through the high-gradient latents works better than steering through merely active ones.
**Implication:** This is the gradient-based attribution that the flow backpropagates from latent to neuron. It is also one of the three arms in the attribution-method ablation, alongside the weight-based projection and the combination of the two.

## Sparse Autoencoders Find Highly Interpretable Features in Language Models
**Authors & Year:** Cunningham et al., 2023
**ArXiv/Link:** https://arxiv.org/abs/2309.08600
**Summary:** The authors train sparse autoencoders on language model activations and recover directions that are more interpretable than individual neurons. They also show that these directions support finer causal edits than neuron-level edits do.
**Implication:** This paper is why the flow decomposes BLOOM's residual stream instead of reading neurons directly. The neurons that language activation probability entropy scores are polysemantic, so a sparse dictionary should resolve them into cleaner units.

## Scaling and evaluating sparse autoencoders
**Authors & Year:** Gao et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.04093
**Summary:** The authors use k-sparse autoencoders to control sparsity directly, and they report clean scaling laws that relate dictionary size, sparsity, and reconstruction quality. They also cut the number of dead latents at large scale.
**Implication:** These scaling laws set the dictionary width and the sparsity level that the flow must pick when it trains a multi-layer sparse autoencoder on BLOOM from scratch. They also predict part of the gap that RQ5 expects between BLOOM-1.7B and BLOOM-560M.

## Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders
**Authors & Year:** Rajamanoharan et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2407.14435
**Summary:** The authors replace the ReLU in a sparse autoencoder with a discontinuous JumpReLU and train the sparsity level directly through straight-through estimators. This improves reconstruction at a fixed sparsity without hurting interpretability.
**Implication:** A dictionary that reconstructs poorly would let a failed patch in RQ3 look like a failed attribution. This activation choice is therefore how the flow keeps reconstruction error from confounding the causal validation.

## Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models
**Authors & Year:** Marks et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2403.19647
**Summary:** The authors build causal graphs whose nodes are sparse autoencoder features and whose edges come from linear approximations to patching effects. They then edit these circuits to remove unintended behaviour from a classifier.
**Implication:** The linear edge approximation is the same kind of closed-form path projection that the flow computes from neuron to latent. This paper is therefore the template for the weight-based arm of the attribution ablation.

## AtP*: An efficient and scalable method for localizing LLM behaviour to components
**Authors & Year:** Kramár et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2403.00745
**Summary:** The authors analyse where gradient-based approximations to activation patching fail, and they fix two failure modes that produce false negatives. The corrected method stays cheap while matching full patching much more closely.
**Implication:** The false-negative modes that this paper describes are the ones most likely to make the flow's gradient attribution miss a genuinely causal latent. RQ3 therefore reads a disagreement between gradient attribution and patching against this list rather than treating it as a null result.

---

# Related and Competing Approaches

## Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages
**Authors & Year:** Andrylie et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2507.11230
**Summary:** The authors introduce SAE-LAPE, which scores a sparse autoencoder feature as language-specific from that feature's own activation probability across languages rather than from any neuron. They find these features mostly in the middle to final layers of the feed-forward network, and the features are interpretable enough to serve as a language identifier.
**Implication:** The paper tests Llama 3.2 1B and Gemma 2, not BLOOM, so the flow reproduces this method on BLOOM's own sparse autoencoder features to build the first of the two reference feature sets that RQ1 measures hypergeometric enrichment against. Its middle-to-final concentration in Llama and Gemma is the feature-level pole that RQ2 weighs against the neuron-level layer claims, and because SAE-LAPE never consults the language activation probability entropy neurons, any enrichment the flow measures in its own BLOOM reproduction counts as evidence of a shared causal pathway rather than a shared method.

## Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders
**Authors & Year:** Deng et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2505.05111
**Summary:** The authors define a monolinguality metric over sparse autoencoder features and show that ablating the top-scoring features hurts one language while leaving the others almost untouched. They also find groups of features whose joint ablation hurts more than the sum of the single ablations, and they turn these features into steering vectors.
**Implication:** The paper tests Gemma 2 2B, Gemma 2 9B, and Llama 3.1 8B, not BLOOM, so the flow reproduces this monolinguality metric on BLOOM to build the second reference feature set for the RQ1 enrichment test, and its synergistic groups predict that the flow's attributed latents should arrive in clusters rather than one at a time. Its feature-derived steering vectors give RQ6 a qualitative precedent for latent-level steering, though not a directly comparable magnitude, since the underlying model differs.

## Language Model Circuits Are Sparse in the Neuron Basis
**Authors & Year:** Arora et al., 2026
**ArXiv/Link:** https://arxiv.org/abs/2601.22594
**Summary:** The authors find causally important neurons with a gradient method and show that a circuit of roughly a hundred MLP neurons already controls model behaviour on subject-verb agreement and multi-hop reasoning. They argue that neurons work as interpretable units without any extra dictionary training.
**Implication:** This is the competing view that the flow has to answer, because it predicts that attributed latents add nothing beyond the neurons themselves. RQ3 is the test, since the flow's claim fails if latent patching cannot match neuron patching.

## Transcoders Find Interpretable LLM Feature Circuits
**Authors & Year:** Dunefsky et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.11944
**Summary:** The authors approximate a dense MLP layer with a wider sparse layer and use it to run weights-based circuit analysis through the MLP. The resulting circuits split cleanly into an input-dependent part and an input-invariant part.
**Implication:** The input-invariant part is exactly the weight-only path from a neuron to a latent that the flow computes in closed form. This paper is therefore the closest existing check on whether such a projection can be trusted before patching confirms it.

## Sparse Crosscoders for Cross-Layer Features and Model Diffing
**Authors & Year:** Lindsey et al., 2024
**ArXiv/Link:** https://transformer-circuits.pub/2024/crosscoders/index.html
**Summary:** The authors describe crosscoders, which read from and write to several layers at once, and they use them to track features that persist across layers and to compare two models. Crosscoders also remove duplicate copies of the same feature at neighbouring layers.
**Implication:** A crosscoder is the main alternative to the multi-layer sparse autoencoder for the flow's job. This paper marks that design choice, and it explains why the flow keeps Lawson's formulation, since RQ2 needs a per-latent layer position that a crosscoder deliberately collapses.

## Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
**Authors & Year:** Kantamneni et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2502.16681
**Summary:** The authors test sparse autoencoder features against simple baselines on probing tasks under data scarcity, label noise, and covariate shift. The features rarely beat the baselines by a consistent margin.
**Implication:** This is why the flow keeps a plain co-activation baseline between neurons and latents. RQ1 only means something if weight-based and gradient-based attribution beat that correlational baseline on overlap with the two reference sets.

## Towards Best Practices of Activation Patching in Language Models: Metrics and Methods
**Authors & Year:** Zhang and Nanda, 2024
**ArXiv/Link:** https://arxiv.org/abs/2309.16042
**Summary:** The authors show that the choice of evaluation metric and of corruption method changes which components a patching experiment names as important. They recommend concrete settings that make the results stable.
**Implication:** RQ3 compares the patching effect of attributed latents against the patching effect of the underlying neurons. The flow therefore has to hold metric and corruption fixed across the two conditions, and this paper specifies how to do that.

---

# Evaluation Tools and Benchmarks

## The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset
**Authors & Year:** Laurençon et al., 2022
**ArXiv/Link:** https://arxiv.org/abs/2303.03915
**Summary:** The authors document the 1.6 terabyte corpus in 59 languages that trained BLOOM, and they release a large subset together with the processing tools. The paper reports how much text each language contributes.
**Implication:** The per-language token counts here decide the high-resource and low-resource split in the RQ7 ablation. The released subset also gives the flow a probe corpus that matches BLOOM's own pretraining distribution.

## No Language Left Behind: Scaling Human-Centered Machine Translation
**Authors & Year:** NLLB Team et al., 2022
**ArXiv/Link:** https://arxiv.org/abs/2207.04672
**Summary:** The team builds translation systems for over two hundred languages and releases FLORES-200, a parallel benchmark whose content is professionally translated into every one of them. They evaluate more than forty thousand translation directions against it.
**Implication:** FLORES-200 gives the flow sentence-aligned text in every language that it studies. The multilingual probe corpus needs exactly that, so a difference in neuron-latent co-activation reflects language rather than topic.

## The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants
**Authors & Year:** Bandarkar et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2308.16884
**Summary:** The authors build a fully parallel multiple-choice reading comprehension set over 122 language variants on top of FLORES-200 passages. Because it is parallel, scores compare directly across languages.
**Implication:** This is the task on which the flow can check that latent patching in RQ3 shifts output language without wrecking comprehension. Its parallel design also keeps the high-resource and low-resource comparison in RQ7 fair.

## SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability
**Authors & Year:** Karvonen et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2503.09532
**Summary:** The authors evaluate more than two hundred sparse autoencoders across eight metrics covering interpretability, feature disentanglement, and downstream use. They find that gains on proxy metrics often fail to carry over to practical performance.
**Implication:** The flow trains a new multi-layer sparse autoencoder on BLOOM with no released checkpoint to fall back on. These metrics are how the flow shows that its dictionary is sound before it reports any overlap number for RQ1.

## URIEL+: Enhancing Linguistic Inclusion and Usability in a Typological and Multilingual Knowledge Base
**Authors & Year:** Khan et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2409.18472
**Summary:** The authors extend the URIEL typological knowledge base and the lang2vec tool with more languages and more features, and they add imputation methods for missing values. Users can also customise how the distance between two languages is computed.
**Implication:** RQ4 has to correlate typological distance with attributed-latent overlap, so it needs a numeric distance between every pair of BLOOM's languages. This resource supplies that distance directly.

## Understanding and Mitigating Language Confusion in LLMs
**Authors & Year:** Marchisio et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2406.20052
**Summary:** The authors build the Language Confusion Benchmark over fifteen typologically diverse languages and show that even strong models often answer in the wrong language. They then reduce the failure with few-shot prompting and with multilingual tuning.
**Implication:** This benchmark measures the exact behaviour that the flow's steering metric targets. RQ6 can therefore score latent-level steering against neuron-level steering on prompts whose failure mode is already documented.

## Bag of Tricks for Efficient Text Classification
**Authors & Year:** Joulin et al., 2016
**ArXiv/Link:** https://arxiv.org/abs/1607.01759
**Summary:** The authors present fastText, a linear classifier over bag-of-n-gram features that matches deep models on text classification while training orders of magnitude faster. The same model is the standard tool for language identification.
**Implication:** Andrylie et al. calibrate their language-specific features by comparing them against fastText as a language identifier. The flow needs the same reference in order to reproduce their feature set faithfully for the RQ1 overlap test.

---

# Theoretical Grounding

## Toy Models of Superposition
**Authors & Year:** Elhage et al., 2022
**ArXiv/Link:** https://arxiv.org/abs/2209.10652
**Summary:** The authors build a small model where polysemanticity can be traced exactly, and they show that it arises when a network stores more sparse features than it has dimensions. They also find a phase change and a link to the geometry of uniform polytopes.
**Implication:** Superposition is the reason why a single language-specific neuron cannot be assumed to carry one language concept. That assumption gap between the neuron account and the feature account is what this flow sets out to bridge.

## The Linear Representation Hypothesis and the Geometry of Large Language Models
**Authors & Year:** Park et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2311.03658
**Summary:** The authors give a precise meaning to the claim that a concept is represented as a direction, and they ground it in counterfactual output pairs. They then identify an inner product under which the different notions of linear representation agree.
**Implication:** The flow's weight-based attribution projects a neuron's output direction onto latent decoder directions. That projection only measures a causal path if language is represented linearly in the sense that this paper defines.

## The Geometry of Categorical and Hierarchical Concepts in Large Language Models
**Authors & Year:** Park et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2406.01506
**Summary:** The authors show that categorical concepts appear as simplices and that hierarchically related concepts sit orthogonally to one another in representation space. They confirm the structure on Gemma over nearly a thousand WordNet concepts.
**Implication:** Language is a categorical variable with a family tree over it, so this geometry predicts the pattern that RQ4 looks for. Typologically close languages should share latent directions, while distant ones should not.

## Not All Language Model Features Are One-Dimensionally Linear
**Authors & Year:** Engels et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2405.14860
**Summary:** The authors define irreducible multi-dimensional features and then find real examples, such as circular representations of weekdays and months. They use sparse autoencoders to discover these features automatically.
**Implication:** Language identity may be multi-dimensional rather than a single direction, and a one-latent-per-neuron reading of the attribution would then be wrong. This is why the flow reports attributed latent sets and measures set overlap in RQ1 instead of matching single latents.

## Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias
**Authors & Year:** Vig et al., 2020
**ArXiv/Link:** https://arxiv.org/abs/2004.12265
**Summary:** The authors adapt causal mediation analysis to neural networks and treat neurons and attention heads as mediators between input and output. They find bias effects to be sparse, synergistic, and separable into direct and indirect parts.
**Implication:** The flow's whole design is a mediation claim, since latents are supposed to mediate the effect of a neuron on output language. This paper supplies the vocabulary and the estimator on which the RQ3 patching comparison rests.

## Locating and Editing Factual Associations in GPT
**Authors & Year:** Meng et al., 2022
**ArXiv/Link:** https://arxiv.org/abs/2202.05262
**Summary:** The authors use causal tracing to find the middle-layer feed-forward computations that carry a factual association, and they then edit those weights directly. The edit changes the fact without retraining the model.
**Implication:** Causal tracing is the ancestor of the patching step that validates this flow's attribution. Its finding that feed-forward modules hold the decisive computation also supports starting from BLOOM's feed-forward neurons.

## Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small
**Authors & Year:** Wang et al., 2023
**ArXiv/Link:** https://arxiv.org/abs/2211.00593
**Summary:** The authors reverse-engineer a twenty-six-head circuit for indirect object identification and check it against faithfulness, completeness, and minimality. This was the largest end-to-end circuit account of a natural behaviour at the time.
**Implication:** Their three criteria are what the flow borrows to judge an attributed latent set. RQ3 asks whether patching the set reproduces most of the neuron effect, which is a faithfulness claim in their sense.

---

# Downstream Application and Validation

## Neural FOXP2: Language Specific Neuron Steering for Targeted Language Improvement in LLMs
**Authors & Year:** Saha et al., 2026
**ArXiv/Link:** https://arxiv.org/abs/2602.00945
**Summary:** The authors find language neurons with a sparse autoencoder, pick steering directions by spectral analysis of activation differences between languages, and then shift activations so that Hindi or Spanish becomes the model's default. They describe language preference as a sparse, low-rank control circuit.
**Implication:** This is the closest existing attempt at steering language through sparse autoencoder units rather than raw neurons. RQ6 therefore measures the flow's attributed-latent steering against it as well as against Gurgurov's neuron-level arithmetic.

## SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs
**Authors & Year:** Wang et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2507.14894
**Summary:** The authors find that unwanted code-switching comes with unusually high pre-activation values on the target language's sparse autoencoder features, and they add a loss that holds those values down during fine-tuning. Code-switching drops by more than half across five models and three languages.
**Implication:** This shows that language-specific latents carry enough causal weight to change generation under training pressure. RQ6 tests the same causal claim at inference time, where steering alone has to do the work.

## Linguistic Neuron Overlap Patterns to Facilitate Cross-lingual Transfer on Low-resource Languages
**Authors & Year:** Xu et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2508.17078
**Summary:** The authors study shared rather than language-specific neurons, and they use overlap patterns between language pairs to pick bridge languages for in-context learning. They test fifteen language pairs from seven families.
**Implication:** Their pairwise neuron overlap is the neuron-level counterpart of what RQ4 computes over attributed latents. The flow can therefore check whether latent overlap ranks language pairs in the same order that neuron overlap does.

## Sparse Subnetwork Enhancement for Underrepresented Languages in Large Language Models
**Authors & Year:** Gurgurov et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2510.13580
**Summary:** The authors fine-tune only the neurons that language activation probability entropy selects, and they beat full fine-tuning and LoRA on twelve mid- and low-resource languages while touching under one percent of parameters. They release neuron sets for more than a hundred languages.
**Implication:** This is direct evidence that entropy-selected neurons stay meaningful in low-resource settings. It is therefore the optimistic side of RQ7, and it is the counterweight to the negative result that Mondal et al. report.

## Mechanistic Understanding and Mitigation of Language Confusion in English-Centric Large Language Models
**Authors & Year:** Nie et al., 2025
**ArXiv/Link:** https://arxiv.org/abs/2505.16538
**Summary:** The authors locate the token positions where a model switches language, and they trace the failure to the final layers through lens-based analysis and neuron attribution. Editing a small set of neurons cuts confusion while keeping general ability intact.
**Implication:** Their final-layer localisation is a second concrete prediction for RQ2. Their small edited neuron set also tells RQ3 how many units a successful output-language intervention should need.

## Cross-Lingual Activation Steering for Multilingual Language Models
**Authors & Year:** Pokharel et al., 2026
**ArXiv/Link:** https://arxiv.org/abs/2601.16390
**Summary:** The authors modulate neuron activations at inference time to lift performance in non-dominant languages, and they report gains on classification and generation without hurting high-resource languages. They find that the gains track increased separation between language clusters.
**Implication:** Their cluster-separation result predicts that steering works best where language-specific units are cleanest. That is the mechanism which RQ5 expects to weaken in BLOOM-560M relative to BLOOM-1.7B.

## Steering Language Models With Activation Engineering
**Authors & Year:** Turner et al., 2024
**ArXiv/Link:** https://arxiv.org/abs/2308.10248
**Summary:** The authors compute a steering vector from the difference between activations on a contrast pair of prompts, and they add it to the forward pass at inference. This controls high-level output properties without changing weights and without hurting unrelated tasks.
**Implication:** This is the general form of the intervention that RQ6 uses. It therefore provides the untargeted steering baseline that both neuron-level arithmetic and latent-level steering have to beat before either counts as a language-specific mechanism.
