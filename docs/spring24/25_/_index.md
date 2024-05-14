---
type: docs
bookToc: True
weight: 1
---

# Merging Text Transformer Models from Different Initializations
*Posted by: Kyungtae Kim, Minwoo Kim*

*Authors: Neha Verma (Johns Hopkins University), Maha Elbayad (Meta)*

Although recent works on model merging have exhibited low- or zero-barrier mode connectivity between models with different initialization, model merging on transformer architecture has not yet been studied extensively. The application of previous merging techniques on the transformer structure is limited due to its unique structural characteristics, such as residual connection, multi-head attention (MHA), and sequential input. The paper merges separate transformer minima, proposing a new model merging technique to investigate the relationship between the pre-trained models' minima in the loss landscape. Using permutation-based model merging, authors found lower loss barriers between minima compared to other model merging techniques such as model averaging. The results showed that the model has less sharp and isolated minima than previously expected.

The contributions of the researchers are listed as follows:
1. They introduced a new transformer merging algorithm based on model permutation.
2. They showed that the technique leads to decreased loss barriers between masked language models trained from different initializations compared to other merging methods.
3. They extended their approach to fine-tuned models and showed consistently smaller loss barriers between models compared to vanilla merging.

## Background
### Transformer
<p align="center">
    <img src=figures/transformer.png width="900">
</p>

Transformer is a type of sequence-to-sequence (seq2seq) models that takes a sequence of tokens as an input, and computes according to the input token. Unlike previous seq2seq models where a certain input token had a hard time affecting every output tokens, transformer uses ***self-attention***, which allows all tokens to affect every output tokens. This allows for better performance in data where the distance between tokens has low relationship to the importance of the tokens' importance. For more details on transformers and attention, see the paper ['Attention is All You Need'](https://arxiv.org/abs/1706.03762).

### Loss Landscape
[Loss landscape](https://arxiv.org/abs/1712.09913) is a representation of the loss values around the weight space of the network. Loss landscape helps researchers see how well a neural network has been trained and gives researchers new insights on their models.
<p align="center">
    <img src=figures/loss_landscape.png width="900">
</p>

DNNs are trained by optimizing a loss function with an stochastic gradient descent (SGD) variant. The loss landscapes of these networks have been shown to contain infinitely many global minimizers that are reachable with the help of SGD. One reason for the abundance of minima is **overparameterization**, which leads different functions to behave similarly on the training data. **Permutation and scaling invariances** also lead to functionally identical minima that differ in the weight space. Prior works stated that the optima of loss functions are connected by simple curves over which training and test accuracy are nearly constant (no loss barrier). This is called *mode connectivity*. Other researchers conjectured that if the permutation invariances of neural networks are taken into account, these optima are linearly mode connected, i.e. the linear path connecting these two models has no loss barrier.
In the paper, the authors pay attention on how permutation between models could lead to similar or identical loss landscapes.

### Model Interpolation
Model interpolation is a technique that blends two or more models to create an intermediate model. This process is mostly done by averaging the model weights. Researchers found out that if fine-tuned models lie in a single low error basin, then the weight averaging performs similarly to ensembling, which combines the output of multiple fine-tuned model to hopefully obtain a better result. It is however not guaranteed that fine-tuned models (starting from the same initialization) will reside in the same loss basin. Prior work on linear interpolation-based model merging has focused on improving the algorithms used to bring the hidden units of two networks into alignment, in order to reduce the barrier to interpolation between them.

## Permutation-based Merging
### Feed-Forward Layers
In this section, we explain how the authors of the paper used permutation to find the similarities between two distinct models and merge them. Given two models {{< katex >}}\theta_A{{< /katex >}} and {{< katex >}}\theta_B{{< /katex >}} trained from distinct initializations, the authors compute post-activation features for each layer or sublayer parameter {{< katex >}}\text{W}_l\subset \theta{{< /katex >}} in order to compute the similar parts across models. The researchers compute {{< katex >}}d{{< /katex >}}-dimensional activations across {{< katex >}}n{{< /katex >}} tokens from both models {{< katex >}}\text{X}_A, \text{X}_B\in \mathbb{R}^{n\times d}{{< /katex >}}. Then, the feature relatedness via cross-correlation is computed as

<p>{{< katex display=true >}}
C=\text{corr}(\text{X}_A, \text{X}_B)=\frac{\mathbb{E}[(\text{X}_A-\boldsymbol{\mu}_{\text{X}_A})^\text{T}(\text{X}_B-\boldsymbol{\mu}_{\text{X}_B})]}{\boldsymbol{\sigma}_{\text{X}_A}\boldsymbol{\sigma}_{\text{X}_B}},
{{< /katex >}}</p>

where {{< katex >}}\boldsymbol{\sigma}{{< /katex >}} is a standard deviation vector, and {{< katex >}}\boldsymbol{\mu}{{< /katex >}} is a mean vector. The features are standardized since the magnitude of features values can vary greatly depending on the initialization. Next, the permutation that gives the highest correlation score is computed, and is declared as the optimal computation. More specifically, given {{< katex >}}C\in\mathbb{R}^{d\times d}{{< /katex >}} and a permutation mapping {{< katex >}}\pi{{< /katex >}}, the optimal permutation is computed as follows:

{{< katex display=true >}}
\text{arg}\max_\pi \sum_{i=1}^{d} C(i, \pi(i)).
{{< /katex >}}

cf. The above problem is solved using the Jonker-Volgenant algorithm.

Next, the permutation mapping {{< katex >}}\pi{{< /katex >}} is converted to a permutation matrix {{< katex >}}\text{P}{{< /katex >}}. The matrix is then multiplied to the original weight matrix of {{< katex >}}B{{< /katex >}} denoted as {{< katex >}}\text{W}_l^B \subset \theta_B{{< /katex >}}. Then the permuted weight matrix {{< katex >}}\text{P}\text{W}_l^B{{< /katex >}} closely resembles the weight {{< katex >}}A{{< /katex >}}, denoted as {{< katex >}}\text{W}_l^A \subset \theta_A{{< /katex >}}. Denoting the modified model parameters as {{< katex >}}\theta_B'{{< /katex >}}, the final merged model is computed as {{< katex >}}\lambda\theta_A+(1-\lambda)\theta_B{{< /katex >}} for some {{< katex >}}\lambda\in[0,1]{{< /katex >}}.
cf. If permutation matrix {{< katex >}}\text{P}{{< /katex >}} is multiplied in layer {{< katex >}}l{{< /katex >}}, then {{< katex >}}\text{P}^{\text{T}}=\text{P}^{-1}{{< /katex >}} is applied in the next layer to unpermute the ordering, i.e.,

<p>{{< katex display=true >}}
\text{W}_{l+1}^{B'} \leftarrow \text{W}_{l+1}^{B}\text{P}^{\text{T}}.
{{< /katex >}}</p>

### Multi-head Attentions
<p>Multi-head attention parameters include parameters from key, query, value, and linear layer each denoted as {{< katex >}}\text{W}_K{{< /katex >}}, {{< katex >}}\text{W}_Q{{< /katex >}}, {{< katex >}}\text{W}_V{{< /katex >}}, and {{< katex >}}\text{W}_O{{< /katex >}}. For each key, query, and value weights, the whole parameter {{< katex >}}\text{W} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}{{< /katex >}} is partitioned into {{< katex >}}H{{< /katex >}} attention heads each of output dimension {{< katex >}}d_k = d_{\text{model}}/H{{< /katex >}}. Permutation should be operated on each attention head separately, in order to apply a permutation to full weight matrices and maintain the functional equivalence of the overall model. This is because the final hidden vector from MHA reflects a concatenation of the result from each head, which are computed separately with weights {{< katex >}}\text{W}_{K_i} ,\text{W}_{Q_i} , \text{W}_{V_i}{{< /katex >}} for head {{< katex >}}i{{< /katex >}}. In the paper's case, since the models are trained from different initializations, the correspondence of their attention heads may differ in addition to the correspondence of features within each head. The features are extracted just after the attention computation and before the linear layer. The features are used to compute {{< katex >}}C{{< /katex >}}, and then the correlation matrix is partitioned by heads into {{< katex >}}d_k \times d_k{{< /katex >}} correlation matrices, for each potential attention head pair. Next, optimal permutation for each unique head pair {{< katex >}}(j, k){{< /katex >}} is computed. Each head's internal permutation is computed and stored, and the cost is computed as</p>

{{< katex display=true >}}
\text{cost}(j,k)=\max_\pi \sum_{i=1}^{d_k} C_{jk}(i,\pi(i)),
{{< /katex >}}

where {{< katex >}}C_{jk}{{< /katex >}} refers to the specific partition of the overall correlation matrix. The outer head correspondence permutation is computed as

{{< katex display=true >}}
\pi_{\text{outer}}=\text{arg}\max_\pi \sum_{h=1}^{H} \text{cost}(h,\pi(h)).
{{< /katex >}}

<p align="center">
    <img src=figures/algorithm1.png width="400">
</p>

The algorithm outputs a permuting matrix {{< katex >}}\text{P}_{\text{MHA}}{{< /katex >}}, which is applied to each of {{< katex >}}\text{W}_V{{< /katex >}}, {{< katex >}}\text{W}_K{{< /katex >}} and {{< katex >}}\text{W}_Q{{< /katex >}}.

#### Residual Connections
Each transformer layer comes with two residual connections, as can be seen from FIgure 1. The residual connections can be formulated as follows:

{{< katex display=true >}}
\begin{align}
x_a^r&=\text{LN}(\text{W}_O \text{MHA}(x) + x),\\
x_f^r&=\text{LN}(\text{W}_2 \text{ReLU}(\text{W}_1 x_a^r) + x_a^r).
\end{align}
{{< /katex >}}

The input and output of both sublayers are added to create a new output. This implies that if a permutation operation is applied to the output state, the permutation should be the same for both addends. Also, since the inputs passes through the LayerNorm module, the permutation to the output should also permute the features of the LayerNorm module also. Ignoring the parameters of the LayerNorm,

{{< katex display=true >}}
\begin{align}
\text{P}x_f^r&=\text{P}(\text{W}_2 \text{ReLU}(\text{W}_1 x_a^r) + x_a^r)\\
&=\text{P}\text{W}_2 \text{ReLU}(\text{W}_1 x_a^r) + \text{P}x_a^r\\
&=\text{P}\text{W}_2 \text{ReLU}(\text{W}_1 x_a^r) + \text{P}(\text{W}_O \text{MHA}(x) + x)
\end{align}
{{< /katex >}}

Since the input to each layer must be permuted ({{< katex >}}\text{P}x{{< /katex >}}), and the output of each layer is also permuted ({{< katex >}}\text{P}x_f^r{{< /katex >}}), the entire transformer architecture uses the same {{< katex >}}\{\text{P}, \text{P}^{\text{T}}\}{{< /katex >}} matrices for all weights involved in residual connections.

## Results

## Conclusion

## References
https://arxiv.org/abs/2403.00986

[Loss landscape figure](https://arxiv.org/abs/1712.09913)
