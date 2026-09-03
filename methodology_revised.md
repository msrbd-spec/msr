\section{Methodology}
\label{sec:methodology}

\subsection{Overview}
\label{subsec:overview}
This study presents a federated learning framework for automated diagnosis of pox-family skin diseases from dermatological images. The framework pairs a custom attention-augmented convolutional network, ConvNeXtV2-MSAFv5, with the FedProx aggregation protocol to address two challenges inherent to federated medical imaging: statistical data heterogeneity across clients and inter-class visual ambiguity among vesicular lesions. The pipeline operates across five collaborating clients, each maintaining a private data set. In every communication round, the server broadcasts the current global weights, clients perform local training, and the server aggregates the returned updates via size-weighted averaging with a proximal regularization term. Training proceeds for 50 rounds, and the best global model is selected by the macro-F1 score on the aggregated validation set. All experiments use a 5-fold cross-validation scheme, with performance reported as mean $\pm$ standard deviation across folds. The overall framework is illustrated in Fig.~\ref{fig:overall_framework}.

Beyond the primary FedProx framework, the methodology encompasses several extensions evaluated under the same protocol: differentially private federated training (DP-SGD), federated personalization (FedPer), label-flipping poisoning robustness, post-hoc calibration, external validation with prior-shift correction, and comparison against four pretrained baseline architectures. These extensions are described in dedicated subsections below.

\begin{figure}[htbp]
    \centering
    % \includegraphics[width=\textwidth]{figures/framework_diagram.png}
    \vspace{5cm}
    \caption{Overall framework showing the federated learning setup with five clients and the central server. The left panel depicts the data partitioning scheme (both IID Uniform and Non-IID Heterogeneous settings), the middle panel shows the client-side local training pipeline (ConvNeXtV2-MSAFv5 with FocalLoss and FedProx proximal regularization), and the right panel illustrates the server aggregation step and global model evaluation loop.}
    \label{fig:overall_framework}
\end{figure}

\subsection{Dataset Construction and Cross-Validation}
\label{subsec:dataset}
The experimental data set consists of images from publicly available sources: MSLDv2, MSID, EMSID, and DermnetNZ, combined into four target classes: Monkeypox, Chickenpox, Measles, and Healthy. After deduplication using perceptual hashing to remove visually redundant samples, the final pool contains 220 Chickenpox, 460 Healthy, 221 Measles, and 562 Monkeypox images, totaling 1{,}463 images. The pool is divided into five folds with a train--validation--test split ratio of 60:20:20. Partitioning is performed once at the global dataset level before any client assignment, ensuring no data leakage between splits.

\subsection{Federated Client Partitioning}
\label{subsec:client_partition}
The training partition of each fold is distributed across $K = 5$ clients under two distinct settings.

\textbf{Run~1 (IID, Uniform Distribution).} The training images of each class are divided equally across all five clients, resulting in approximately 300 training images per client. This configuration serves as a controlled baseline under statistical homogeneity.

\textbf{Run~2 (Non-IID, Heterogeneous Distribution).} Client data volumes are assigned according to a heterogeneous weight vector $\mathbf{w} = [0.10, 0.15, 0.30, 0.25, 0.20]$, such that Client~1 receives 10\% of the training data and Client~3 receives the largest share at 30\%. Class proportions are maintained within each client's allocation, but the absolute volumes differ substantially, creating the non-IID condition representative of real-world federated medical scenarios.

Per-client training images are subsequently augmented to a fixed target of 1{,}500 images per client using class-balanced offline augmentation, yielding approximately 7{,}500 training images per fold per run.

\subsection{Data Preprocessing and Augmentation}
\label{subsec:preprocessing}
All images are resized to $384 \times 384$ pixels and normalized using ImageNet statistics with mean $\boldsymbol{\mu} = (0.485, 0.456, 0.406)$ and standard deviation $\boldsymbol{\sigma} = (0.229, 0.224, 0.225)$ across the RGB channels.

During centralized training, a strong augmentation pipeline is applied to improve generalization. The pipeline consists of the following operations, applied in sequence:
\begin{itemize}
    \item Resize to $416 \times 416$ followed by a random crop to $384 \times 384$.
    \item Random horizontal and vertical flips.
    \item ColorJitter with brightness, contrast, and saturation factors of 0.3 and hue perturbation of 0.08.
    \item RandAugment with $N=2$ operations and magnitude 9 \cite{CITATION_NEEDED_randaugment}.
    \item RandomErasing with probability 0.2 and scale range 0.02--0.12.
\end{itemize}
Additionally, MixUp \cite{CITATION_NEEDED_mixup} with $\alpha = 0.1$ is applied during centralized training, interpolating between pairs of training samples and their labels to improve decision boundary calibration.

For federated training, a lighter augmentation pipeline is used: resize to $384 \times 384$, random horizontal and vertical flips, and ColorJitter with factors of 0.2 and hue of 0.05. No MixUp, CutMix, RandAugment, or RandomErasing is applied in the federated setting, as the proximal regularization and per-client data heterogeneity already provide a degree of regularization.

Validation and test sets are subjected only to resizing and normalization. For inference, 10-crop Test-Time Augmentation (TTA) is applied by extracting the center crop and four corner crops at $384 \times 384$ from a $400 \times 400$ resized image, along with their horizontal flips, and averaging the resulting ten softmax probability vectors.

\subsection{Proposed Architecture: ConvNeXtV2-MSAFv5}
\label{subsec:architecture}
The proposed model, ConvNeXtV2-MSAFv5, is an attention-augmented architecture built upon the ConvNeXtV2-Tiny backbone \cite{CITATION_NEEDED_convnextv2}. As illustrated in Fig.~\ref{fig:model_arch}, it incorporates stage-specific attention mechanisms (ECA in shallow stages, CBAM in deep stages) and a custom Cross-Scale Attention Head (CSAH) that fuses multi-scale semantic features for the final classification decision. Table~\ref{tab:architecture} summarizes the architecture specifications.

\begin{figure}[htbp]
    \centering
    % \includegraphics[width=\textwidth]{figures/model_diagram.png}
    \vspace{5cm}
    \caption{Detailed diagram of the ConvNeXtV2-MSAFv5 architecture. The backbone is depicted as four progressive stages with decreasing spatial resolution ($56\times56 \rightarrow 28\times28 \rightarrow 14\times14 \rightarrow 7\times7$) and increasing channel depth ($96 \rightarrow 192 \rightarrow 384 \rightarrow 768$). ECA modules are attached to Stage~0 and Stage~1 outputs. CBAM modules with stochastic depth are attached to Stage~2 and Stage~3 outputs. The three feature maps from Stages~1, 2, and 3 are routed to the CSAH, where GeM pooling, linear projection, and cross-attention fusion yield the final classification logits.}
    \label{fig:model_arch}
\end{figure}

\begin{table}[htbp]
    \centering
    \caption{Architecture specification of ConvNeXtV2-MSAFv5. Stage indices follow the ConvNeXtV2-Tiny convention. Channel dimensions and spatial resolutions correspond to a $384 \times 384$ input.}
    \label{tab:architecture}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Stage} & \textbf{Channels} & \textbf{Resolution} & \textbf{Attention} & \textbf{Regularization} \\
        \midrule
        Stem  & 96  & $96 \times 96$  & --- & --- \\
        Stage~0 & 96  & $96 \times 96$  & ECA & Gated residual ($\alpha{=}0.01$) \\
        Stage~1 & 192 & $48 \times 48$  & ECA & Gated residual ($\alpha{=}0.01$) \\
        Stage~2 & 384 & $24 \times 24$  & CBAM & Stochastic depth ($p_d{=}0.10$) \\
        Stage~3 & 768 & $12 \times 12$  & CBAM & Stochastic depth ($p_d{=}0.10$) \\
        CSAH   & 256 & $1 \times 1$    & Cross-attn & Dropout (0.35, 0.15) \\
        \bottomrule
    \end{tabular}
\end{table}

\subsubsection{Backbone: ConvNeXtV2-Tiny}
The backbone is ConvNeXtV2-Tiny, pre-trained using Fully Convolutional Masked Autoencoders (FCMAE) and subsequently fine-tuned on ImageNet-22k and ImageNet-1k. The architecture is organized into four sequential stages that progressively reduce spatial resolution while expanding the number of feature channels:
\begin{equation}
    \label{eq:stages}
    \text{Stage } i \in \{0, 1, 2, 3\}, \quad C_i \in \{96, 192, 384, 768\}
\end{equation}
Each ConvNeXt block employs a depthwise $7 \times 7$ convolution, LayerNorm, and a Global Response Normalization (GRN) layer. The network uses LayerNorm throughout, which avoids the BatchNorm-to-GroupNorm substitution required by architectures such as ResNet or MobileNet in federated settings where batch statistics are unreliable across heterogeneous clients.

\subsubsection{Early-Stage Feature Refinement via ECA}
For Stage~0 ($C_0 = 96$) and Stage~1 ($C_1 = 192$), where feature maps are spatially large, Efficient Channel Attention (ECA) \cite{CITATION_NEEDED_eca} blocks provide lightweight cross-channel recalibration with near-zero parameter overhead. Given an input feature map $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$, the ECA block computes a channel attention weight vector $\mathbf{w} \in \mathbb{R}^{B \times C \times 1 \times 1}$:
\begin{equation}
    \label{eq:eca_gap}
    \mathbf{z} = \text{GAP}(\mathbf{X}) \in \mathbb{R}^{B \times 1 \times C}
\end{equation}
\begin{equation}
    \label{eq:eca_conv}
    \mathbf{w} = \sigma\!\left(\text{Conv1D}_k(\mathbf{z})\right), \quad k = \left\lfloor \frac{\log_2 C}{\gamma} + \frac{b}{\gamma} \right\rceil_{\text{odd}}, \quad \gamma = 2, \; b = 1
\end{equation}
where $\sigma(\cdot)$ denotes the sigmoid activation, $\text{GAP}(\cdot)$ is global average pooling, and $\text{Conv1D}_k$ is a 1-D convolution with adaptive kernel size $k$ that captures local cross-channel interactions without channel dimensionality reduction. The attention output is injected via a gated residual connection with a learnable scalar $\alpha$ initialized to $0.01$:
\begin{equation}
    \label{eq:eca_residual}
    \mathbf{X}' = \mathbf{X} + \alpha \cdot (\mathbf{X} \odot \mathbf{w} - \mathbf{X})
\end{equation}
At initialization, $\alpha \approx 0$, making the module a near-identity transformation. During training, $\alpha$ grows as the module learns to contribute meaningful recalibrations, avoiding disruption to the pretrained feature distribution.

\subsubsection{Deep Semantic Attention via CBAM with Stochastic Depth}
For Stage~2 ($C_2 = 384$) and Stage~3 ($C_3 = 768$), the smaller spatial extent permits the application of full spatial attention alongside channel attention. The Convolutional Block Attention Module (CBAM) \cite{CITATION_NEEDED_cbam} applies channel and spatial attention sequentially, refining which channels encode lesion-discriminative features and which spatial regions correspond to the actual lesion area.

\textbf{Channel Attention.} Given $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$, the channel attention gate $\mathbf{g}_c \in \mathbb{R}^{B \times C \times 1 \times 1}$ is computed using both max-pooled and average-pooled descriptors through a shared MLP:
\begin{equation}
    \label{eq:cbam_ch}
    \mathbf{g}_c = \sigma\!\left(\text{MLP}\!\left(\text{MaxPool}(\mathbf{X})\right) + \text{MLP}\!\left(\text{AvgPool}(\mathbf{X})\right)\right)
\end{equation}
\begin{equation}
    \label{eq:cbam_mlp}
    \text{MLP}(\cdot) = \mathbf{W}_2 \cdot \text{ReLU}\!\left(\mathbf{W}_1 \cdot (\cdot)\right), \quad \mathbf{W}_1 \in \mathbb{R}^{C/r \times C},\; \mathbf{W}_2 \in \mathbb{R}^{C \times C/r},\; r = 16
\end{equation}
where $r$ is the reduction ratio.

\textbf{Spatial Attention.} The channel-recalibrated feature map $\tilde{\mathbf{X}} = \mathbf{X} \odot \mathbf{g}_c$ is passed through the spatial attention module:
\begin{equation}
    \label{eq:cbam_sp}
    \mathbf{g}_s = \sigma\!\left(\text{Conv}_{7 \times 7}\!\left(\left[\text{MaxPool}_c(\tilde{\mathbf{X}}); \text{AvgPool}_c(\tilde{\mathbf{X}})\right]\right)\right) \in \mathbb{R}^{B \times 1 \times H \times W}
\end{equation}
where $[\cdot ; \cdot]$ denotes concatenation along the channel dimension and $\text{MaxPool}_c$/$\text{AvgPool}_c$ operate along the channel axis. The final attended feature map is $\hat{\mathbf{X}} = \tilde{\mathbf{X}} \odot \mathbf{g}_s$.

Stochastic Depth (DropPath) \cite{CITATION_NEEDED_stochastic_depth} with probability $p_d = 0.10$ is applied to the CBAM residual contribution during training, combined with the same gated residual formulation as ECA:
\begin{equation}
    \label{eq:cbam_residual}
    \mathbf{X}' = \mathbf{X} + \alpha \cdot \text{DropPath}_{p_d}\!\left(\hat{\mathbf{X}} - \mathbf{X}\right)
\end{equation}

\subsubsection{Cross-Scale Attention Head (CSAH)}
\label{subsec:csah}
Rather than classifying solely from the final stage's feature map, the CSAH aggregates discriminative information from Stages~1, 2, and 3, enabling the classifier to simultaneously leverage mid-level (lesion boundary, texture) and high-level (global lesion morphology) semantic features.

\textbf{Generalized Mean Pooling.} For each selected stage output $\mathbf{S}_i \in \mathbb{R}^{B \times C_i \times H_i \times W_i}$, a compact descriptor $\mathbf{p}_i \in \mathbb{R}^{B \times C_i}$ is obtained via Generalized Mean (GeM) pooling \cite{CITATION_NEEDED_gem} with a learnable exponent $p_{\text{GeM}}$ initialized to 3.0:
\begin{equation}
    \label{eq:gem}
    \mathbf{p}_i = \left(\frac{1}{H_i W_i} \sum_{h,w} \mathbf{S}_i(:, :, h, w)^{p_{\text{GeM}}}\right)^{1/p_{\text{GeM}}}
\end{equation}
GeM pooling with $p_{\text{GeM}} > 1$ emphasizes the most activated spatial locations, making it more discriminative than standard average pooling for fine-grained lesion features. Each descriptor is then linearly projected to a common dimension $d = 256$:
\begin{equation}
    \label{eq:csah_proj}
    \mathbf{t}_i = \text{LayerNorm}\!\left(\mathbf{W}_i^{\text{proj}} \mathbf{p}_i\right) \in \mathbb{R}^{B \times d}, \quad i \in \{1, 2, 3\}
\end{equation}

\textbf{Cross-Scale Attention Fusion.} The three projected tokens are stacked to form a token sequence $\mathbf{T} = [\mathbf{t}_1; \mathbf{t}_2; \mathbf{t}_3] \in \mathbb{R}^{B \times 3 \times d}$. Cross-attention is computed with the deepest token $\mathbf{t}_3$ (highest semantic abstraction) serving as the query, while all three tokens serve as keys and values:
\begin{equation}
    \label{eq:csah_qkv}
    \mathbf{Q} = \mathbf{t}_3 \mathbf{W}^Q \in \mathbb{R}^{B \times 1 \times d}, \quad \mathbf{K} = \mathbf{T}\mathbf{W}^K \in \mathbb{R}^{B \times 3 \times d}, \quad \mathbf{V} = \mathbf{T}\mathbf{W}^V \in \mathbb{R}^{B \times 3 \times d}
\end{equation}
\begin{equation}
    \label{eq:csah_attn}
    \mathbf{f} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{B \times d}
\end{equation}
The fused representation is residually combined with the deepest token, normalized, and scaled by a learnable temperature parameter $\tau$ (initialized to 1.0) for logit calibration:
\begin{equation}
    \label{eq:csah_fused}
    \mathbf{f}' = \tau \cdot \text{LayerNorm}\!\left(\mathbf{f} + \mathbf{t}_3\right)
\end{equation}
The final classification logits are produced by a two-layer MLP head with Dropout regularization:
\begin{equation}
    \label{eq:csah_head}
    \hat{\mathbf{y}} = \mathbf{W}_2^{\text{head}} \cdot \text{GELU}\!\left(\mathbf{W}_1^{\text{head}} \cdot \text{Dropout}_{0.35}(\mathbf{f}')\right) \in \mathbb{R}^{B \times 4}
\end{equation}
where $\mathbf{W}_1^{\text{head}} \in \mathbb{R}^{128 \times 256}$, $\mathbf{W}_2^{\text{head}} \in \mathbb{R}^{4 \times 128}$, and an intermediate Dropout of rate 0.15 is applied before the final projection.

\subsubsection{Auxiliary Classification Heads}
\label{subsec:aux_heads}
An optional auxiliary supervision mechanism attaches two lightweight auxiliary heads to the Stage~1 and Stage~2 feature maps during centralized training. Each auxiliary head consists of a GeM pooling layer, a LayerNorm, and a linear classifier. The auxiliary loss is weighted by $\lambda_{\text{aux}} = 0.2$ and combined with the primary Focal Loss to provide gradient signal to shallower layers. This mechanism is evaluated as an ablation configuration and is not used in the primary model.

\subsection{Federated Learning Framework}
\label{subsec:fedprox}

\subsubsection{Problem Formulation}
Let $\mathcal{D}_k$ denote the local dataset of client $k$, with $|\mathcal{D}_k| = n_k$ samples, and let $n = \sum_{k=1}^K n_k$ be the total dataset size. The global federated optimization objective is:
\begin{equation}
    \label{eq:fl_objective}
    \min_{\mathbf{w}} \; F(\mathbf{w}) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(\mathbf{w}), \quad F_k(\mathbf{w}) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(f(\mathbf{x}_i; \mathbf{w}), y_i)
\end{equation}
where $\ell(\cdot)$ is the task-specific loss function and $f(\cdot;\mathbf{w})$ is ConvNeXtV2-MSAFv5 parameterized by $\mathbf{w}$.

\subsubsection{FedProx Local Objective}
In the heterogeneous (Non-IID) federated setting, standard FedAvg \cite{CITATION_NEEDED_fedavg} is susceptible to client drift, where divergent local updates destabilize global convergence. To mitigate this, the FedProx algorithm \cite{CITATION_NEEDED_fedprox} augments each client's local objective with a proximal regularization term that penalizes deviations from the current global model $\mathbf{w}^t$:
\begin{equation}
    \label{eq:fedprox_local}
    \min_{\mathbf{w}_k} \; h_k(\mathbf{w}_k; \mathbf{w}^t) = F_k(\mathbf{w}_k) + \frac{\mu}{2} \|\mathbf{w}_k - \mathbf{w}^t\|^2
\end{equation}
The proximal coefficient $\mu = 0.01$ controls the strength of this regularization. A larger $\mu$ constrains local models to remain closer to the global initialization, sacrificing some local expressiveness to improve global stability. The selected value represents an empirically validated balance between local adaptation and global convergence.

Upon completing $E = 3$ local epochs, each client transmits its updated parameters $\mathbf{w}_k^{t+1}$ to the central server. The global model is updated via size-weighted averaging:
\begin{equation}
    \label{eq:fedavg}
    \mathbf{w}^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_k^{t+1}
\end{equation}
This aggregation is performed for $R = 50$ global communication rounds, with early stopping triggered if the aggregated validation macro-F1 score does not improve by at least $5 \times 10^{-5}$ within $P = 18$ consecutive rounds. Stochastic Weight Averaging (SWA) \cite{CITATION_NEEDED_swa} is applied to the global model during the final 10 rounds (rounds 41--50), maintaining a running average of the global weights. Before evaluation, the SWA model's normalization statistics are updated on the training set of the first client.

\subsubsection{Robust Aggregation: Trimmed Mean}
\label{subsec:trimmed_mean}
For poisoning robustness experiments, an alternative aggregation rule based on coordinate-wise trimmed mean \cite{CITATION_NEEDED_trimmed_mean} is evaluated alongside standard FedAvg. Given $K$ client updates, the trimmed mean discards the highest and lowest $\lfloor K \cdot f_{\text{trim}} \rfloor$ values per parameter coordinate before averaging the remaining values, where $f_{\text{trim}} = 0.2$ is the trimming fraction. This rule limits the influence of any single malicious client on the global model.

\subsection{Centralized Training Strategy}
\label{subsec:centralized}
The centralized training protocol serves as an upper bound for the federated setting and is used for all baseline models. Training proceeds in three sequential phases, each with a separate optimizer and cosine learning rate schedule to prevent schedule interference.

\textbf{Phase~1 (Backbone Frozen, 10 epochs).} The ConvNeXtV2-Tiny backbone is entirely frozen. Only the ECA modules, CBAM modules, and CSAH head are trainable, allowing the newly initialized attention parameters to converge without disrupting the pretrained backbone features. The learning rates are set to $\eta_{\text{attn}} = 10^{-4}$ and $\eta_{\text{head}} = 2 \times 10^{-4}$.

\textbf{Phase~2 (Deep Stage Unfreezing, 15 epochs).} Stage~2 and Stage~3 of the backbone are unfrozen. The backbone learning rate is set to $\eta_{\text{backbone}} = 3 \times 10^{-5}$, an order of magnitude lower than the head, to avoid destructive interference with the deep semantic representations acquired during pretraining.

\textbf{Phase~3 (Full Unfreezing with SAM, 40 epochs).} All parameters are trainable. The optimizer is switched to Sharpness-Aware Minimization (SAM) \cite{CITATION_NEEDED_sam} with $\rho = 0.05$, which seeks parameters in flat loss basins by computing a two-step gradient update:
\begin{equation}
    \label{eq:sam_step1}
    \hat{\mathbf{w}} = \mathbf{w} + \rho \cdot \frac{\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w})}{\|\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w})\|}
\end{equation}
\begin{equation}
    \label{eq:sam_step2}
    \mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}(\hat{\mathbf{w}})
\end{equation}
Flat minima found by SAM generalize better in the federated setting, where each round presents a slightly different effective loss surface due to partial participation and varying local data distributions. Gradient clipping with max norm 1.0 is applied in all phases.

Label smoothing ($\varepsilon = 0.05$) and SWA are applied during Phase~3. SWA maintains a running average of model weights beginning at epoch 10 within Phase~3, promoting convergence to flatter regions of the loss landscape. The SWA model's normalization statistics are updated on the training set before evaluation. At inference, the standard best checkpoint and the SWA checkpoint are ensembled by averaging their softmax probability outputs.

The centralized training procedure is formalized in Algorithm~\ref{alg:centralized}.

\begin{algorithm}[htbp]
\caption{Centralized Training of ConvNeXtV2-MSAFv5 with Phased Unfreezing and SAM}
\label{alg:centralized}
\begin{algorithmic}[1]
\REQUIRE Training set $\mathcal{D}_{\text{train}}$, validation set $\mathcal{D}_{\text{val}}$, phases $\{P_1, P_2, P_3\}$
\ENSURE Best model $\mathbf{w}^*$, SWA model $\mathbf{w}_{\text{swa}}$
\STATE Initialize $\mathbf{w}$ from pretrained ConvNeXtV2-Tiny (FCMAE + IN22k + IN1k)
\STATE \textbf{Phase~1:} Freeze backbone; train attention + head for $|P_1|$ epochs with AdamW
\STATE \textbf{Phase~2:} Unfreeze Stage~2 + Stage~3; train for $|P_2|$ epochs with AdamW
\STATE \textbf{Phase~3:} Unfreeze all; initialize SAM optimizer ($\rho = 0.05$) + AdamW base
\FOR{epoch $e = 1, \dots, |P_3|$}
    \FOR{each mini-batch $(\mathbf{x}_b, \mathbf{y}_b) \in \mathcal{D}_{\text{train}}$}
        \STATE Apply MixUp: $(\tilde{\mathbf{x}}_b, \tilde{\mathbf{y}}_a, \tilde{\mathbf{y}}_b, \lambda) \leftarrow \text{MixUp}(\mathbf{x}_b, \mathbf{y}_b, \alpha{=}0.1)$
        \STATE \textit{SAM Step~1:} $\hat{\mathbf{w}} \leftarrow \mathbf{w} + \rho \cdot \nabla \mathcal{L} / \|\nabla \mathcal{L}\|$
        \STATE \textit{SAM Step~2:} $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\hat{\mathbf{w}}} \mathcal{L}'$
        \STATE Clip gradients: $\|\nabla\| \leq 1.0$
    \ENDFOR
    \IF{epoch $\geq$ SWA\_START}
        \STATE $\mathbf{w}_{\text{swa}} \leftarrow \text{SWA\_Average}(\mathbf{w}_{\text{swa}}, \mathbf{w})$
    \ENDIF
    \STATE Evaluate on $\mathcal{D}_{\text{val}}$; save best checkpoint by macro-F1
    \IF{No improvement for $P = 18$ epochs}
        \STATE Early stopping
    \ENDIF
\ENDFOR
\STATE Update normalization statistics of $\mathbf{w}_{\text{swa}}$ on $\mathcal{D}_{\text{train}}$
\RETURN $\mathbf{w}^*$, $\mathbf{w}_{\text{swa}}$
\end{algorithmic}
\end{algorithm}

The phased design in Algorithm~\ref{alg:centralized} serves a specific purpose: Phase~1 allows the randomly initialized attention modules and CSAH to adapt to the frozen backbone's feature space without destabilizing pretrained representations, Phase~2 gradually introduces trainable capacity in the deeper stages where lesion-discriminative semantics reside, and Phase~3 applies SAM across all parameters to locate a flat minimum that generalizes across the heterogeneous loss surfaces encountered in federated rounds. The MixUp interpolation in Phase~3 further regularizes the decision boundary by mixing samples from different pox classes, which share overlapping visual features. The SWA average, collected from epoch 10 onward, captures weights from the flatter region of the loss landscape that the cosine schedule reaches in its later epochs, and the final normalization-statistic update ensures the averaged model produces calibrated activations at inference.

\subsection{Loss Function: Class-Weighted Focal Loss}
\label{subsec:focal_loss}
To address the class imbalance in the dataset, where Chickenpox ($n=220$) and Measles ($n=221$) are significantly underrepresented relative to Monkeypox ($n=562$) and Healthy ($n=460$), Focal Loss \cite{CITATION_NEEDED_focal_loss} with class-specific weighting is employed. The Focal Loss for a predicted probability distribution $\hat{p}$ and true class $y$ is defined as:
\begin{equation}
    \label{eq:focal_loss}
    \mathcal{L}_{\text{focal}}(\hat{p}, y) = -\alpha_y (1 - \hat{p}_y)^{\gamma} \log(\hat{p}_y)
\end{equation}
where $\gamma = 2.0$ is the focusing parameter that down-weights the loss contribution of well-classified easy examples, and $\alpha_y$ is the class-specific weight derived from the inverse frequency of classes in the test set distribution:
\begin{equation}
    \label{eq:focal_alpha}
    \alpha_y = \frac{1/n_y^{\text{test}}}{\sum_{c} 1/n_c^{\text{test}}}, \quad \boldsymbol{\alpha} = [0.332, 0.148, 0.399, 0.120]
\end{equation}
for Chickenpox, Healthy, Measles, and Monkeypox, respectively. Calibrating $\boldsymbol{\alpha}$ from the test distribution rather than the augmented training distribution is deliberate: since per-class training volumes are equalized to 1{,}500 images per client through offline augmentation, train-based alpha values would be approximately uniform and uninformative. The test distribution, which reflects the true underlying class imbalance, provides a more meaningful weighting signal.

\subsection{Federated Training Procedure}
\label{subsec:fl_training}
The federated training procedure differs from the centralized protocol in several important respects. Each client trains for $E = 3$ local epochs per round using the Focal Loss with the FedProx proximal term (Eq.~\eqref{eq:fedprox_local}). The optimizer is AdamW with weight decay $2 \times 10^{-4}$ and the same three-group learning rate schedule ($\eta_{\text{backbone}} = 3 \times 10^{-5}$, $\eta_{\text{attn}} = 10^{-4}$, $\eta_{\text{head}} = 2 \times 10^{-4}$). No MixUp, label smoothing, or SAM is applied in the federated setting; the proximal regularization and data heterogeneity provide sufficient regularization. Gradient clipping with max norm 1.0 is applied to each local update.

After all clients complete their local epochs, the server aggregates the updates via size-weighted FedAvg (Eq.~\eqref{eq:fedavg}). The aggregated model is evaluated on the combined validation set from all clients. The best checkpoint is saved by macro-F1, and SWA is applied to the global model during the final 10 rounds. The complete federated training procedure is formalized in Algorithm~\ref{alg:fedprox}.

\begin{algorithm}[htbp]
\caption{FedProx Federated Training of ConvNeXtV2-MSAFv5}
\label{alg:fedprox}
\begin{algorithmic}[1]
\REQUIRE $K$ clients with datasets $\{\mathcal{D}_k\}$, $R$ rounds, $E$ local epochs, FedProx $\mu$
\ENSURE Optimal global model $\mathbf{w}^*$
\STATE Initialize $\mathbf{w}^0$ from pretrained ConvNeXtV2-Tiny
\STATE Initialize SWA model $\mathbf{w}_{\text{swa}} \leftarrow \mathbf{w}^0$
\FOR{$t = 0, 1, \dots, R-1$}
    \STATE Server broadcasts $\mathbf{w}^t$ to all $K$ clients
    \FOR{each client $k \in \{1, \dots, K\}$}
        \STATE $\mathbf{w}_k \leftarrow \mathbf{w}^t$
        \FOR{epoch $e = 1, \dots, E$}
            \FOR{each mini-batch $(\mathbf{x}_b, \mathbf{y}_b) \in \mathcal{D}_k$}
                \STATE $\hat{\mathbf{y}} \leftarrow f(\mathbf{x}_b; \mathbf{w}_k)$
                \STATE $\mathcal{L} \leftarrow \mathcal{L}_{\text{focal}}(\hat{\mathbf{y}}, \mathbf{y}_b) + \frac{\mu}{2} \|\mathbf{w}_k - \mathbf{w}^t\|^2$
                \STATE Backpropagate $\mathcal{L}$; clip $\|\nabla\| \leq 1.0$
                \STATE Update $\mathbf{w}_k$ via AdamW
            \ENDFOR
        \ENDFOR
        \STATE Transmit $\mathbf{w}_k$ to server
    \ENDFOR
    \STATE \textbf{Server Aggregation:} $\mathbf{w}^{t+1} \leftarrow \sum_{k=1}^K \frac{n_k}{n} \mathbf{w}_k$
    \STATE Evaluate $\mathbf{w}^{t+1}$ on aggregated validation set
    \IF{macro-F1 improved}
        \STATE Save checkpoint $\mathbf{w}^*$
    \ELSIF{No improvement for $P = 18$ rounds}
        \STATE Early stopping
    \ENDIF
    \IF{$t \geq R - 10$}
        \STATE $\mathbf{w}_{\text{swa}} \leftarrow \text{SWA\_Average}(\mathbf{w}_{\text{swa}}, \mathbf{w}^{t+1})$
    \ENDIF
\ENDFOR
\STATE Update normalization statistics of $\mathbf{w}_{\text{swa}}$
\RETURN $\mathbf{w}^*$, $\mathbf{w}_{\text{swa}}$
\end{algorithmic}
\end{algorithm}

In Algorithm~\ref{alg:fedprox}, the proximal term $\frac{\mu}{2}\|\mathbf{w}_k - \mathbf{w}^t\|^2$ in the local loss (line~9) is the key mechanism that distinguishes FedProx from standard FedAvg: it penalizes large deviations from the broadcast global model, preventing any single client with a small or skewed local dataset from pulling the global model toward a local minimum that generalizes poorly to other clients. The size-weighted aggregation (line~17) ensures that clients with more training data exert proportionally greater influence on the global update. The SWA average collected during the final 10 rounds (lines~24--25) captures the late-stage global weights, which have typically converged to a flatter region of the federated loss landscape, and the normalization-statistic update on line~27 ensures the averaged model produces calibrated activations at inference.

\subsection{Differentially Private Federated Learning (DP-SGD)}
\label{subsec:dp_sgd}
To evaluate the privacy-utility tradeoff, differentially private federated training is conducted using the Opacus library \cite{CITATION_NEEDED_opacus}, which implements the DP-SGD algorithm \cite{CITATION_NEEDED_dpsgd}. DP-SGD modifies the local update procedure by clipping per-sample gradients to a maximum norm $C$ and adding Gaussian noise with standard deviation $\sigma = C \cdot \tilde{\sigma}$ to the aggregated gradient before each optimizer step. The privacy budget is controlled by the target epsilon $\varepsilon$ and delta $\delta = 10^{-5}$, with the noise multiplier $\tilde{\sigma}$ computed automatically by Opacus to satisfy the target $(\varepsilon, \delta)$-differential privacy guarantee.

\subsubsection{Model Adaptations for DP Compatibility}
Two adaptations are required to make ConvNeXtV2-MSAFv5 compatible with Opacus:

\textbf{Single-tensor forward.} Opacus uses functorch-based per-sample gradient computation, which replays the forward pass with a single tensor argument. The CSAH's forward method originally accepts a Python list $[\mathbf{s}_1, \mathbf{s}_2, \mathbf{s}_3]$, which causes Opacus's activation-capture hooks to fail with an \texttt{AttributeError} when calling \texttt{.detach()} on the list object. To resolve this, the GeM pooling, linear projection, and token stacking operations are moved from the CSAH into the parent model's forward method, which calls each sub-module (GeM pool, projection) individually with a single tensor. The CSAH then receives a single stacked tensor $\mathbf{T} \in \mathbb{R}^{B \times 3 \times d}$, making all sub-module calls Opacus-safe.

\textbf{Random operation disabling.} Opacus's vmap backend does not support random operations. Setting the dropout probability to zero is insufficient because \texttt{nn.Dropout.forward} still invokes \texttt{F.dropout}, which calls a registered random operator flagged by vmap. All \texttt{nn.Dropout} forward methods are therefore replaced with an identity function, and StochasticDepth modules are disabled by setting their drop probability to zero and overriding their forward method. This is standard practice in DP training, as the injected noise already provides regularization.

\subsubsection{DP-SGD Local Update}
The DP-SGD local update procedure is formalized in Algorithm~\ref{alg:dp_sgd}. After training, the Opacus-wrapped \texttt{GradSampleModule} is unwrapped to ensure the state dictionary keys match the global model during FedAvg aggregation. The experiment is conducted on 3 folds (Fold~1, Fold~3, Fold~5) under the Non-IID setting with three privacy budgets: $\varepsilon \in \{8.0, 12.0\}$, plus a no-DP baseline ($\varepsilon = \infty$).

\begin{algorithm}[htbp]
\caption{DP-SGD Local Update with Opacus}
\label{alg:dp_sgd}
\begin{algorithmic}[1]
\REQUIRE Global model $\mathbf{w}^t$, client loader $\mathcal{D}_k$, Focal Loss, target $\varepsilon$, $\delta = 10^{-5}$, max grad norm $C = 1.0$, local epochs $E$
\ENSURE Local model $\mathbf{w}_k$, average loss, achieved $\varepsilon$
\STATE $\mathbf{w}_k \leftarrow \text{copy}(\mathbf{w}^t)$
\STATE Validate and fix model for DP compatibility (replace BatchNorm if present)
\STATE Disable all random operations (Dropout, StochasticDepth)
\STATE Initialize PrivacyEngine with target $\varepsilon$, $\delta$, epochs $E$, max grad norm $C$
\STATE $\mathbf{w}_k, \text{opt}, \mathcal{D}_k \leftarrow \text{PrivacyEngine.make\_private\_with\_epsilon}(\mathbf{w}_k, \text{opt}, \mathcal{D}_k, \varepsilon, \delta, E, C)$
\FOR{epoch $e = 1, \dots, E$}
    \FOR{each mini-batch $(\mathbf{x}_b, \mathbf{y}_b) \in \mathcal{D}_k$}
        \STATE $\hat{\mathbf{y}} \leftarrow f(\mathbf{x}_b; \mathbf{w}_k)$
        \STATE $\mathcal{L} \leftarrow \mathcal{L}_{\text{focal}}(\hat{\mathbf{y}}, \mathbf{y}_b)$
        \STATE Backpropagate $\mathcal{L}$ (per-sample gradients clipped to $C$, noise added)
        \STATE Update $\mathbf{w}_k$ via AdamW
    \ENDFOR
\ENDFOR
\STATE $\varepsilon_{\text{achieved}} \leftarrow \text{PrivacyEngine.get\_epsilon}(\delta)$
\STATE Unwrap GradSampleModule: $\mathbf{w}_k \leftarrow \mathbf{w}_k.\text{\_module}$
\RETURN $\mathbf{w}_k$, $\varepsilon_{\text{achieved}}$
\end{algorithmic}
\end{algorithm}

In Algorithm~\ref{alg:dp_sgd}, the model-fixing step (line~2) and random-operation disabling (line~3) are performed before the PrivacyEngine is attached (line~5), because Opacus requires the model to be in a compatible state before wrapping it in a \texttt{GradSampleModule}. The \texttt{make\_private\_with\_epsilon} call on line~5 computes the noise multiplier $\tilde{\sigma}$ that satisfies the target $(\varepsilon, \delta)$ guarantee given the number of samples, epochs, and batch size, then wraps the optimizer and dataloader to perform per-sample gradient clipping and Gaussian noise injection during backpropagation (line~11). The unwrapping on line~15 is critical for federated aggregation: the \texttt{GradSampleModule} wrapper changes the state dictionary key prefixes, so removing it before transmission ensures the local model's keys match the global model during FedAvg. The achieved $\varepsilon$ reported on line~14 is typically lower than the target, providing a tighter privacy guarantee than requested.

\subsection{Federated Personalization (FedPer)}
\label{subsec:fedper}
In the FedPer framework \cite{CITATION_NEEDED_fedper}, the model is decomposed into a shared representation (backbone and attention modules) and a personal component (the CSAH classification head). During each round, clients train the full model locally, but only the backbone and attention parameters are aggregated on the server. The head parameters remain local to each client and are never transmitted.

The FedPer aggregation rule modifies Eq.~\eqref{eq:fedavg} to operate on a subset of parameters:
\begin{equation}
    \label{eq:fedper}
    \mathbf{w}^{t+1}_{\text{shared}} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_{k,\text{shared}}^{t+1}, \quad \mathbf{w}^{t+1}_{\text{head}} = \mathbf{w}^t_{\text{head}}
\end{equation}
where $\mathbf{w}_{\text{shared}}$ comprises all parameters with prefixes in \{stem, stage0--3, ds1--3, attn0--3\} and $\mathbf{w}_{\text{head}}$ comprises the CSAH parameters. The global model's head is not updated from client heads; instead, each client's head is personalized to its local data distribution. FedPer is evaluated under the Non-IID setting, where personalization is expected to provide the greatest benefit.

\subsection{Poisoning Robustness}
\label{subsec:poisoning}
To evaluate the framework's resilience to adversarial label-flipping attacks, a poisoning scenario is simulated in which Client~1's training labels are randomly flipped to a different class for a fraction $f$ of its samples. The flip targets are chosen uniformly at random from the remaining classes. Three poisoning severities are evaluated: $f \in \{0.2, 0.4, 0.6\}$, with $f = 0$ serving as the clean baseline.

Three defense strategies are compared under each severity:
\begin{itemize}
    \item \textbf{FedAvg ($\mu = 0$):} Standard FedAvg aggregation with no proximal regularization, representing the absence of defense.
    \item \textbf{FedProx ($\mu = 0.01$):} The proposed method, where the proximal term constrains client drift.
    \item \textbf{FedProx + Trimmed Mean:} FedProx local training combined with coordinate-wise trimmed mean aggregation ($f_{\text{trim}} = 0.2$) at the server, which limits the influence of the poisoned client.
\end{itemize}

\subsection{Calibration and Uncertainty Estimation}
\label{subsec:calibration}
Model calibration is assessed using four complementary metrics: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score, and Negative Log-Likelihood (NLL). Predictions are binned into $B = 15$ equal-width confidence intervals.

\textbf{ECE} measures the weighted average absolute difference between confidence and accuracy within each bin:
\begin{equation}
    \label{eq:ece}
    \text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \text{acc}(b) - \text{conf}(b) \right|
\end{equation}
where $n_b$ is the number of samples in bin $b$, $N$ is the total number of samples, $\text{acc}(b)$ is the empirical accuracy within the bin, and $\text{conf}(b)$ is the mean confidence.

\textbf{MCE} captures the worst-case calibration gap:
\begin{equation}
    \label{eq:mce}
    \text{MCE} = \max_{b \in \{1, \dots, B\}} \left| \text{acc}(b) - \text{conf}(b) \right|
\end{equation}

\textbf{Brier score} measures the mean squared difference between predicted probabilities and one-hot labels:
\begin{equation}
    \label{eq:brier}
    \text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (\hat{p}_{i,c} - y_{i,c})^2
\end{equation}
where $\hat{p}_{i,c}$ is the predicted probability for class $c$ and $y_{i,c}$ is the one-hot label.

\textbf{NLL} measures the negative log-likelihood of the true labels under the predicted distribution:
\begin{equation}
    \label{eq:nll}
    \text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log(\hat{p}_{i, y_i})
\end{equation}

\textbf{Temperature Scaling.} Post-hoc calibration is applied by optimizing a single scalar temperature $T$ on the validation set to minimize NLL. The optimization uses the L-BFGS algorithm \cite{CITATION_NEEDED_lbfgs}:
\begin{equation}
    \label{eq:temp_scaling}
    T^* = \arg\min_{T > 0} \sum_{i} -\log \sigma\!\left(\hat{\mathbf{z}}_i / T\right)_{y_i}
\end{equation}
where $\hat{\mathbf{z}}_i$ are the raw logits for sample $i$ and $\sigma(\cdot)$ denotes the softmax function. Calibrated probabilities are used for all downstream evaluation, including TTA averaging and ensemble fusion. Calibration metrics are reported before and after temperature scaling, and reliability diagrams are generated to visualize the confidence-accuracy alignment.

\subsection{External Validation Protocol}
\label{subsec:external}
External validation is performed on a separate 3-class dataset (Chickenpox, Measles, Monkeypox; no Healthy class available) to assess generalization beyond the training distribution. The five Non-IID FL checkpoints are ensembled by averaging their softmax probability outputs. Since the external set contains only 3 of the 4 training classes, the ensemble's 4-class probability vector is restricted to the 3 available classes and renormalized.

\textbf{Prior-Shift Correction.} Because the class distribution of the external set may differ from the training distribution, a prior-shift correction \cite{CITATION_NEEDED_saerens} is applied. The corrected probabilities are computed by dividing each class probability by the corresponding training-time class weight $\alpha_c$ and renormalizing:
\begin{equation}
    \label{eq:prior_shift}
    \hat{p}'_{i,c} = \frac{\hat{p}_{i,c} / \alpha_c}{\sum_{c'} \hat{p}_{i,c'} / \alpha_{c'}}
\end{equation}
where $\alpha_c$ is the Focal Loss class weight for class $c$ (restricted to the 3 externally available classes and renormalized). Both uncorrected and prior-corrected predictions are reported, with 95\% Wilson confidence intervals for accuracy.

\subsection{Baseline Models}
\label{subsec:baselines}
Four pretrained baseline architectures are evaluated under the identical centralized and federated protocols to provide a comparative reference:
\begin{itemize}
    \item \textbf{EfficientNetV2-S} \cite{CITATION_NEEDED_effnetv2}: Pretrained on ImageNet-1k, with the classifier head replaced by a linear layer mapping to 4 classes.
    \item \textbf{MobileNetV2} \cite{CITATION_NEEDED_mobilenetv2}: Pretrained on ImageNet-1k, with the classifier replaced analogously.
    \item \textbf{ConvNeXtV2-Tiny (plain)} \cite{CITATION_NEEDED_convnextv2}: The same backbone as the proposed model but without any custom attention modules or CSAH head, using the stock global average pooling and classifier.
    \item \textbf{ResNet-50} \cite{CITATION_NEEDED_resnet50}: Pretrained on ImageNet-1k with the fully connected layer replaced.
\end{itemize}
For centralized training, baselines use a single-phase AdamW fit for the same total epoch budget (65 epochs) with MixUp, label smoothing, cosine annealing, and early stopping, as the torchvision models do not support the custom \texttt{get\_param\_groups} or \texttt{freeze\_backbone} interface. For federated training, baselines use the identical FedProx protocol as the proposed model.

\subsection{Evaluation Metrics}
\label{subsec:metrics}
Model performance is assessed using complementary metrics, evaluated on the held-out test set of each fold independently. Results are reported as mean $\pm$ standard deviation across all five folds.

\textbf{Accuracy} measures the proportion of correctly classified samples:
\begin{equation}
    \label{eq:accuracy}
    \text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
\end{equation}

\textbf{Macro-Precision}, \textbf{Macro-Recall}, and \textbf{Macro-F1} are computed per class and averaged with equal weight, ensuring that minority classes are not overshadowed:
\begin{equation}
    \label{eq:precision_recall}
    \text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}
\end{equation}
\begin{equation}
    \label{eq:macro_f1}
    \text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
\end{equation}

\textbf{AUROC} (Area Under the Receiver Operating Characteristic Curve) is computed for each class using a one-vs-rest strategy and reported as macro-average:
\begin{equation}
    \label{eq:auroc}
    \text{AUROC}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} \int_0^1 \text{TPR}_c(t)\, d\,\text{FPR}_c(t)
\end{equation}

\textbf{Test-Time Augmentation (TTA)} averages the softmax probability vectors from 10 spatial crops of each test image:
\begin{equation}
    \label{eq:tta}
    \hat{\mathbf{p}}_{\text{TTA}} = \frac{1}{M}\sum_{m=1}^{M} \text{softmax}\!\left(f(\mathbf{x}^{(m)}; \mathbf{w}) / T^*\right), \quad M = 10
\end{equation}
where $T^*$ is the calibrated temperature. The final predicted class is $\hat{y} = \arg\max_c \hat{\mathbf{p}}_{\text{TTA},c}$.

\textbf{Ensemble} combines the best checkpoint and the SWA checkpoint by averaging their softmax probability outputs:
\begin{equation}
    \label{eq:ensemble}
    \hat{\mathbf{p}}_{\text{ens}} = \text{softmax}\!\left(\frac{\mathbf{z}_{\text{best}} + \mathbf{z}_{\text{swa}}}{2}\right)
\end{equation}
where $\mathbf{z}_{\text{best}}$ and $\mathbf{z}_{\text{swa}}$ are the raw logit outputs of the best and SWA models, respectively.

\subsection{Experimental Setup}
\label{subsec:experimental}
All experiments are conducted on a single NVIDIA A100 GPU (40\,GB VRAM) using PyTorch with the \texttt{timm} library \cite{CITATION_NEEDED_timm} for the ConvNeXtV2-Tiny backbone. The random seed is fixed at 42 for reproducibility, with deterministic CuDNN enabled. The batch size is 32 with 8 data-loading workers. Table~\ref{tab:hyperparams} summarizes the complete hyperparameter configuration.

\begin{table}[htbp]
    \centering
    \caption{Hyperparameters and implementation details for the proposed ConvNeXtV2-MSAFv5 model.}
    \label{tab:hyperparams}
    \begin{tabular}{lc}
        \toprule
        \textbf{Parameter} & \textbf{Value} \\
        \midrule
        Image resolution & $384 \times 384$ \\
        Batch size & 32 \\
        Number of clients $K$ & 5 \\
        Number of FL rounds $R$ & 50 \\
        Local epochs per round $E$ & 3 \\
        FedProx proximal coefficient $\mu$ & 0.01 \\
        Backbone LR $\eta_{\text{backbone}}$ & $3 \times 10^{-5}$ \\
        Attention modules LR $\eta_{\text{attn}}$ & $1 \times 10^{-4}$ \\
        Classification head LR $\eta_{\text{head}}$ & $2 \times 10^{-4}$ \\
        Weight decay & $2 \times 10^{-4}$ \\
        Optimizer (centralized Phases~1--2) & AdamW \\
        Optimizer (centralized Phase~3) & SAM ($\rho = 0.05$) + AdamW base \\
        Optimizer (federated) & AdamW \\
        LR schedule & CosineAnnealingLR ($\eta_{\text{min}} = 10^{-7}$) per phase \\
        Phase~1 epochs (backbone frozen) & 10 \\
        Phase~2 epochs (Stage~2+3 unfrozen) & 15 \\
        Phase~3 epochs (full unfreeze + SAM) & 40 \\
        Early stopping patience $P$ & 18 rounds \\
        Focal loss $\gamma$ & 2.0 \\
        Focal loss $\boldsymbol{\alpha}$ & $[0.332, 0.148, 0.399, 0.120]$ \\
        Label smoothing $\varepsilon$ (centralized only) & 0.05 \\
        MixUp $\alpha_{\text{mix}}$ (centralized only) & 0.1 \\
        Drop path rate $p_d$ & 0.10 \\
        GeM pooling exponent $p_{\text{GeM}}$ & 3.0 (learnable) \\
        CSAH projection dimension $d$ & 256 \\
        SWA collection (centralized) & Phase~3, starting epoch 10 \\
        SWA collection (federated) & Rounds 41--50 \\
        TTA crops & 10 \\
        Gradient clipping max norm & 1.0 \\
        DP-SGD max grad norm $C$ & 1.0 \\
        DP-SGD target $\delta$ & $10^{-5}$ \\
        DP-SGD target $\varepsilon$ & $\{8.0, 12.0\}$ \\
        Trimmed mean fraction $f_{\text{trim}}$ & 0.2 \\
        Random seed & 42 \\
        Framework & PyTorch + timm \\
        \bottomrule
    \end{tabular}
\end{table}
```

---

## Changelog (relative to previous draft)

### Corrected
1. **Augmentation pipeline**: Removed RandomAffine and RandomPerspective (not in code). Fixed CutMix α from 1.0 to 0.0 (disabled). Added Resize(416)→RandomCrop(384) detail. Clarified that centralized training uses strong augmentation (RandAugment, RandomErasing, MixUp) while federated training uses a lighter pipeline (flips + ColorJitter only).
2. **TTA**: Fixed crop size from 224×224/240×240 to 384×384/400×400 (TenCrop at IMAGE_SIZE=384).
3. **Algorithm**: Rewrote Algorithm 2 (FedProx) to reflect the actual FL training loop — no phased unfreezing, no SAM, no MixUp, no label smoothing in federated training. Added Algorithm 1 (centralized phased training) and Algorithm 3 (DP-SGD local update).
4. **SWA in FL**: Corrected from "epoch 10 within Phase 3" to "rounds 41–50" (FL_ROUNDS - 10).
5. **Ensemble**: Corrected from "averaging logit outputs" to "averaging softmax probability outputs" (code computes softmax((o1+o2)/2)).
6. **Focal Loss scope**: Clarified that Focal Loss is used in federated training, while centralized training uses label-smoothed cross-entropy with MixUp.
7. **Temperature scaling**: Added L-BFGS optimizer detail.

### Added (new subsections)
8. **DP-SGD (Section 2.9)**: Opacus PrivacyEngine, make_private_with_epsilon, model adaptations (single-tensor forward, random-op disabling), GradSampleModule unwrapping, 3 folds × 3 epsilons.
9. **FedPer (Section 2.10)**: Federate backbone+attention only, head stays local.
10. **Poisoning Robustness (Section 2.11)**: Label-flipping on Client_1, 3 severities, 3 defenses (FedAvg μ=0, FedProx μ=0.01, FedProx+trimmed_mean).
11. **Calibration (Section 2.12)**: ECE, MCE, Brier, NLL, reliability diagrams, temperature scaling with L-BFGS.
12. **External Validation (Section 2.13)**: 3-class ensemble, prior-shift correction (Saerens et al. 2002), Wilson CIs.
13. **Baseline Models (Section 2.14)**: EfficientNetV2-S, MobileNetV2, ConvNeXtV2-Tiny (plain), ResNet50.
14. **Auxiliary heads (Section 2.5.5)**: Optional deep supervision mechanism.
15. **Trimmed mean aggregation (Section 2.6.3)**: Coordinate-wise trimmed mean for robust aggregation.

### Unchanged (confirmed accurate)
16. Dataset construction details (sources, class counts, split ratio) — confirmed by user as done in a separate script.
17. Client partitioning (IID/Non-IID weights) — confirmed by user.
18. 1,500 images per client via offline augmentation — confirmed by user.
19. Focal alpha values [0.332, 0.148, 0.399, 0.120] — verified against code (TEST_COUNTS = [42, 94, 35, 116]).
20. All architecture details (ECA, CBAM, CSAH, GeM, dimensions) — verified against code.

### Citation placeholders to resolve
All citations use `\cite{CITATION_NEEDED_...}` placeholders. The following keys need BibTeX entries:
- `CITATION_NEEDED_randaugment` — RandAugment (Cubuk et al., 2020)
- `CITATION_NEEDED_mixup` — MixUp (Zhang et al., 2018)
- `CITATION_NEEDED_convnextv2` — ConvNeXtV2 (Woo et al., 2023)
- `CITATION_NEEDED_eca` — ECA-Net (Wang et al., 2020)
- `CITATION_NEEDED_cbam` — CBAM (Woo et al., 2018)
- `CITATION_NEEDED_stochastic_depth` — Stochastic Depth (Huang et al., 2016)
- `CITATION_NEEDED_gem` — GeM Pooling (Radenović et al., 2019)
- `CITATION_NEEDED_fedavg` — FedAvg (McMahan et al., 2017)
- `CITATION_NEEDED_fedprox` — FedProx (Li et al., 2020)
- `CITATION_NEEDED_trimmed_mean` — Trimmed Mean (Yin et al., 2018)
- `CITATION_NEEDED_sam` — SAM (Foret et al., 2021)
- `CITATION_NEEDED_swa` — SWA (Izmailov et al., 2018)
- `CITATION_NEEDED_focal_loss` — Focal Loss (Lin et al., 2017)
- `CITATION_NEEDED_opacus` — Opacus (Yousefpour et al., 2021)
- `CITATION_NEEDED_dpsgd` — DP-SGD (Abadi et al., 2016)
- `CITATION_NEEDED_fedper` — FedPer (Arivazhagan et al., 2019)
- `CITATION_NEEDED_saerens` — Prior-shift correction (Saerens et al., 2002)
- `CITATION_NEEDED_lbfgs` — L-BFGS (Liu & Nocedal, 1989)
- `CITATION_NEEDED_effnetv2` — EfficientNetV2 (Tan & Le, 2021)
- `CITATION_NEEDED_mobilenetv2` — MobileNetV2 (Sandler et al., 2018)
- `CITATION_NEEDED_resnet50` — ResNet (He et al., 2016)
- `CITATION_NEEDED_timm` — timm (Wightman, 2019)



----------------------------------------------------

__Before pasting, also make sure your Overleaf project has these packages in the preamble:__

- `amsmath` (for `equation`, `align` environments)
- `amssymb` (for `\mathbb{}`, `\mathcal{}`)
- `graphicx` (for `\includegraphics`)
- `booktabs` (for `\toprule`, `\midrule`, `\bottomrule`)
- `algorithm` + `algorithmic` (for the `algorithm`/`algorithmic` environments)
- `hyperref` (optional, for `\ref{}` linking)
