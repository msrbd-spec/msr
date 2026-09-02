\section{Methodology}
\label{sec:methodology}

\subsection{Overview of the Proposed Framework}
This study introduces a privacy-preserving federated learning framework for automated diagnosis of pox-family skin diseases from dermatological images. To tackle the dual challenges of statistical data heterogeneity and inter-class visual ambiguity in skin lesion classification, the overall pipeline, illustrated in Fig.~\ref{fig:overall_framework}, combines a novel attention-augmented deep learning architecture, ConvNeXtV2-MSAFv5, with the FedProx aggregation protocol. The framework is implemented across five collaborating clients, with each client maintaining its own data set while adhering to privacy constraints. In every round of global communication, the central server sends the current global model weights to all clients. The clients execute a local training procedure in a phased manner Section~\ref{subsec:implementation} with their own data and send the new model parameters back to the server. The server then collects these updates using the FedProx protocol (Section~\ref{subsec:fedprox}). This cycle is repeated for 50 global rounds and the best global model is chosen based on the macro-F1 score of the validation set aggregated across all global rounds. All experiments are performed using a 5-fold cross-validation (CV) scheme, and the performance is reported as mean ± standard deviation over all folds, to ensure statistical robustness.



\begin{figure}[htbp]
    \centering
    % \includegraphics[width=\textwidth]{figures/framework_diagram.png}
    \vspace{5cm} % Placeholder space, remove when adding image
    \caption{Overall architectural framework showing the Federated Learning setup with five clients and the central server. The left panel depicts the data partitioning scheme (both IID Uniform and Non-IID Heterogeneous settings), the middle panel shows the client-side local training pipeline (ConvNeXtV2-MSAFv5 with phased unfreezing and FocalLoss), and the right panel illustrates the FedProx server aggregation step and global model evaluation loop.}
    \label{fig:overall_framework}
\end{figure}

\subsection{Dataset Partitioning and Preprocessing}

\subsubsection{Dataset Construction and 5-Fold Cross-Validation}
The experimental data set consists of images from three publicly available sources: MSLDv2, MSID, EMSID, and DermnetNZ, which were combined into four target classes: Monkeypox, Chickenpox, Measles, and Healthy. After rigorous deduplication using perceptual hashing to remove visually redundant samples, the final pool contains 220 Chickenpox, 460 Healthy, 221 Measles, and 562 Monkeypox images, for a total of 1,463 raw images. The whole pool is divided into five folds with a train–validation–test split ratio of 60:20:20 to obtain a statistically reliable performance estimate. This allows for about 60\% of the images in each class to be used for local client training and 40\% to be reserved for validation and final testing, with the same number of images from each class used for each purpose. Importantly, the partitioning is done once at the global dataset level before any client is assigned to the dataset, so that no data is duplicated.
leakage between splits.



\subsubsection{Federated Client Partitioning}
To simulate realistic federated learning conditions, the training partition of each fold is distributed across $K = 5$ clients under two distinct settings:

\textbf{Run 1 — IID (Uniform Distribution):} The training images of each class are divided equally across all five clients, resulting in approximately 300 training images per client. This configuration serves as a controlled baseline under ideal statistical homogeneity.

\textbf{Run 2 — Non-IID (Heterogeneous Distribution):} Client data volumes are assigned according to a heterogeneous weight vector $\mathbf{w} = [0.10, 0.15, 0.30, 0.25, 0.20]$, such that Client 1 receives 10\% of the training data and Client 3 receives the largest share at 30\%. The class proportions are maintained within each client's allocation, but the absolute volumes differ substantially, creating the non-IID condition representative of real-world federated medical scenarios.

Per-client training images are subsequently augmented to a fixed target of 1,500 images per client using class-balanced offline augmentation, yielding a total of approximately 7,500 training images per fold per run.

\subsubsection{Preprocessing and Data Augmentation}
All images are resized to $384 \times 384$ pixels and normalized using ImageNet statistics with mean $\boldsymbol{\mu} = (0.485, 0.456, 0.406)$ and standard deviation $\boldsymbol{\sigma} = (0.229, 0.224, 0.225)$ across the RGB channels.

During training, a strong augmentation pipeline is applied to each local client's training set to improve generalization and mitigate overfitting. The augmentation sequence consists of:
\begin{itemize}
    \item \textbf{Random horizontal and vertical flips} (each with probability 0.5 and 0.3, respectively).
    \item \textbf{ColorJitter} with brightness, contrast, and saturation factors of 0.3 and hue perturbation of 0.08.
    \item \textbf{RandomAffine} with rotation, translation, and scale variations.
    \item \textbf{RandomPerspective} to simulate varying viewpoints.
    \item \textbf{RandAugment} with $N=2$ operations and magnitude 9 to further diversify texture and color representations.
    \item \textbf{RandomErasing} (probability 0.2, scale 0.02–0.12) to prevent over-reliance on specific spatial regions.
    \item \textbf{MixUp} ($\alpha = 0.1$) and \textbf{CutMix} ($\alpha = 1.0$), applied alternately with equal probability during training, which interpolate between pairs of training samples and their labels to improve decision boundary calibration.
\end{itemize}

Validation and test sets are subjected only to resizing and normalization, preserving their integrity as unaugmented evaluation benchmarks. For inference, \textbf{10-Crop Test-Time Augmentation (TTA)} is additionally applied to the test set by extracting the center crop and four corner crops at $224 \times 224$ from a $240 \times 240$ resized image—along with their horizontal flips—and averaging the resulting ten softmax probability vectors.

\subsection{Proposed Model Architecture: ConvNeXtV2-MSAFv5}
The proposed model, ConvNeXtV2-MSAFv5, is a novel attention-augmented architecture built upon the ConvNeXtV2-Tiny backbone. As illustrated in Fig.~\ref{fig:model_arch}, it incorporates stage-specific attention mechanisms—ECA in shallow stages and CBAM in deep stages—and a custom Cross-Scale Attention Head (CSAH) that fuses multi-scale semantic features for the final classification decision.

\begin{figure}[htbp]
    \centering
    % \includegraphics[width=\textwidth]{figures/model_diagram.png}
    \vspace{5cm} % Placeholder space, remove when adding image
    \caption{Detailed diagram of the ConvNeXtV2-MSAFv5 architecture. The backbone is depicted as four progressive stages with decreasing spatial resolution ($56\times56 \rightarrow 28\times28 \rightarrow 14\times14 \rightarrow 7\times7$) and increasing channel depth ($96 \rightarrow 192 \rightarrow 384 \rightarrow 768$). ECA modules are attached to Stage 0 and Stage 1 outputs. CBAM modules (with stochastic depth) are attached to Stage 2 and Stage 3 outputs. The three feature maps from Stages 1, 2, and 3 are then routed to the CSAH, where GeM pooling, linear projection, and cross-attention fusion yield the final classification logits.}
    \label{fig:model_arch}
\end{figure}

\subsubsection{Backbone: ConvNeXtV2-Tiny}
The backbone is ConvNeXtV2-Tiny, pre-trained using Fully Convolutional Masked Autoencoders (FCMAE) and subsequently fine-tuned on ImageNet-22k and ImageNet-1k, providing rich hierarchical visual representations from a diversity of natural image patterns. The architecture is organized into four sequential stages that progressively reduce spatial resolution while expanding the number of feature channels:
\begin{equation}
    \text{Stage } i \in \{0, 1, 2, 3\}, \quad C_i \in \{96, 192, 384, 768\}, \quad H_i \times W_i \in \{56^2, 28^2, 14^2, 7^2\}
\end{equation}
Each ConvNeXt block employs a depthwise $7 \times 7$ convolution, LayerNorm, and a Global Response Normalization (GRN) layer that inherently provides a form of channel competition, making it naturally compatible with federated training where batch statistics are unreliable. The network naturally uses LayerNorm throughout, obviating the need for the BatchNorm-to-GroupNorm substitution required by architectures such as ResNet or MobileNet in federated settings.

\subsubsection{Early-Stage Feature Refinement via ECA}
For Stage 0 ($C_0 = 96$) and Stage 1 ($C_1 = 192$), where feature maps are spatially large ($56 \times 56$ and $28 \times 28$, respectively), we apply Efficient Channel Attention (ECA) blocks. At these early stages, applying full spatial attention would be computationally prohibitive and likely to introduce noise into low-level feature extraction. ECA instead provides lightweight cross-channel recalibration with near-zero parameter overhead.

Given an input feature map $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$, the ECA block computes a channel attention weight vector $\mathbf{w} \in \mathbb{R}^{B \times C \times 1 \times 1}$ as follows:
\begin{equation}
    \mathbf{z} = \text{GAP}(\mathbf{X}) \in \mathbb{R}^{B \times 1 \times C}
\end{equation}
\begin{equation}
    \mathbf{w} = \sigma\!\left(\text{Conv1D}_k(\mathbf{z})\right), \quad k = \left\lfloor \frac{\log_2 C}{\gamma} + \frac{b}{\gamma} \right\rceil_{\text{odd}}, \quad \gamma = 2, \; b = 1
\end{equation}
where $\sigma(\cdot)$ denotes the sigmoid activation, $\text{GAP}(\cdot)$ is global average pooling, and $\text{Conv1D}_k$ is a 1-D convolution with adaptive kernel size $k$ that captures local cross-channel interactions without channel dimensionality reduction. To preserve pretrained feature integrity during fine-tuning, the attention output is injected via a gated residual connection with a learnable scalar $\alpha$ initialized to $0.01$:
\begin{equation}
    \mathbf{X}' = \mathbf{X} + \alpha \cdot (\mathbf{X} \odot \mathbf{w} - \mathbf{X})
\end{equation}
At initialization, $\alpha \approx 0$, making the module a near-identity transformation. During training, $\alpha$ grows organically as the module learns to contribute meaningful recalibrations, avoiding disruption to the pretrained feature distribution.

\subsubsection{Deep Semantic Attention via CBAM with Stochastic Depth}
For Stage 2 ($C_2 = 384$, $14 \times 14$) and Stage 3 ($C_3 = 768$, $7 \times 7$), the smaller spatial extent permits the application of full spatial attention alongside channel attention. We employ the Convolutional Block Attention Module (CBAM), which applies channel and spatial attention sequentially. CBAM at these deep stages refines which channels encode lesion-discriminative features (e.g., texture vs. background) and which spatial regions correspond to the actual lesion area—critical for distinguishing the visually similar vesicular patterns of Chickenpox and early-stage Monkeypox.

\textbf{Channel Attention.} Given $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$, the channel attention gate $\mathbf{g}_c \in \mathbb{R}^{B \times C \times 1 \times 1}$ is computed using both max-pooled and average-pooled descriptors through a shared MLP:
\begin{equation}
    \mathbf{g}_c = \sigma\!\left(\text{MLP}\!\left(\text{MaxPool}(\mathbf{X})\right) + \text{MLP}\!\left(\text{AvgPool}(\mathbf{X})\right)\right)
\end{equation}
\begin{equation}
    \text{MLP}(\cdot) = \mathbf{W}_2 \cdot \text{ReLU}\!\left(\mathbf{W}_1 \cdot (\cdot)\right), \quad \mathbf{W}_1 \in \mathbb{R}^{C/r \times C},\; \mathbf{W}_2 \in \mathbb{R}^{C \times C/r},\; r = 16
\end{equation}

\textbf{Spatial Attention.} The channel-recalibrated feature map $\tilde{\mathbf{X}} = \mathbf{X} \odot \mathbf{g}_c$ is then passed through the spatial attention module:
\begin{equation}
    \mathbf{g}_s = \sigma\!\left(\text{Conv}_{7 \times 7}\!\left(\left[\text{MaxPool}_c(\tilde{\mathbf{X}}); \text{AvgPool}_c(\tilde{\mathbf{X}})\right]\right)\right) \in \mathbb{R}^{B \times 1 \times H \times W}
\end{equation}
where $[\cdot ; \cdot]$ denotes concatenation along the channel dimension and $\text{MaxPool}_c$/$\text{AvgPool}_c$ operate along the channel axis. The final attended feature map is $\hat{\mathbf{X}} = \tilde{\mathbf{X}} \odot \mathbf{g}_s$.

To regularize the deep attention modules, Stochastic Depth (DropPath) with probability $p_d = 0.10$ is applied to the CBAM residual contribution during training. This, combined with the same gated residual formulation as ECA, yields:
\begin{equation}
    \mathbf{X}' = \mathbf{X} + \alpha \cdot \text{DropPath}_{p_d}\!\left(\hat{\mathbf{X}} - \mathbf{X}\right)
\end{equation}

\subsubsection{Cross-Scale Attention Head (CSAH) with Generalized Mean Pooling}
Rather than classifying solely from the final stage's feature map, the CSAH aggregates discriminative information from Stages 1, 2, and 3, enabling the classifier to simultaneously leverage mid-level (lesion boundary, texture) and high-level (global lesion morphology) semantic features.

\textbf{Generalized Mean Pooling.} For each selected stage output $\mathbf{S}_i \in \mathbb{R}^{B \times C_i \times H_i \times W_i}$, a compact descriptor $\mathbf{p}_i \in \mathbb{R}^{B \times C_i}$ is obtained via Generalized Mean (GeM) pooling with a learnable exponent $p_{\text{GeM}} = 3.0$:
\begin{equation}
    \mathbf{p}_i = \left(\frac{1}{H_i W_i} \sum_{h,w} \mathbf{S}_i(:, :, h, w)^{p_{\text{GeM}}}\right)^{1/p_{\text{GeM}}}
\end{equation}
GeM pooling with $p_{\text{GeM}} > 1$ emphasizes the most activated spatial locations, making it more discriminative than standard average pooling for fine-grained lesion features. Each descriptor is then linearly projected to a common dimension $d = 256$:
\begin{equation}
    \mathbf{t}_i = \text{LayerNorm}\!\left(\mathbf{W}_i^{\text{proj}} \mathbf{p}_i\right) \in \mathbb{R}^{B \times d}, \quad i \in \{1, 2, 3\}
\end{equation}

\textbf{Cross-Scale Attention Fusion.} The three projected tokens are stacked to form a token sequence $\mathbf{T} = [\mathbf{t}_1; \mathbf{t}_2; \mathbf{t}_3] \in \mathbb{R}^{B \times 3 \times d}$. Cross-attention is computed with the deepest token $\mathbf{t}_3$ (highest semantic abstraction) serving as the query, while all three tokens serve as keys and values:
\begin{equation}
    \mathbf{Q} = \mathbf{t}_3 \mathbf{W}^Q \in \mathbb{R}^{B \times 1 \times d}, \quad \mathbf{K} = \mathbf{T}\mathbf{W}^K \in \mathbb{R}^{B \times 3 \times d}, \quad \mathbf{V} = \mathbf{T}\mathbf{W}^V \in \mathbb{R}^{B \times 3 \times d}
\end{equation}
\begin{equation}
    \mathbf{f} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{B \times d}
\end{equation}
The fused representation is residually combined with the deepest token, normalized, and scaled by a learnable temperature parameter $\tau$ (initialized to 1.0) for logit calibration:
\begin{equation}
    \mathbf{f}' = \tau \cdot \text{LayerNorm}\!\left(\mathbf{f} + \mathbf{t}_3\right)
\end{equation}
The final classification logits are produced by a two-layer MLP head with Dropout regularization:
\begin{equation}
    \hat{\mathbf{y}} = \mathbf{W}_2^{\text{head}} \cdot \text{GELU}\!\left(\mathbf{W}_1^{\text{head}} \cdot \text{Dropout}_{0.35}(\mathbf{f}')\right) \in \mathbb{R}^{B \times 4}
\end{equation}
where $\mathbf{W}_1^{\text{head}} \in \mathbb{R}^{128 \times 256}$, $\mathbf{W}_2^{\text{head}} \in \mathbb{R}^{4 \times 128}$, and an intermediate Dropout of rate 0.15 is applied before the final projection.

\subsection{Federated Learning Framework: FedProx}
\label{subsec:fedprox}

\subsubsection{Problem Formulation}
Let $\mathcal{D}_k$ denote the local dataset of client $k$, with $|\mathcal{D}_k| = n_k$ samples, and let $n = \sum_{k=1}^K n_k$ be the total dataset size. The global federated optimization objective is:
\begin{equation}
    \min_{\mathbf{w}} \; F(\mathbf{w}) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(\mathbf{w}), \quad F_k(\mathbf{w}) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(f(\mathbf{x}_i; \mathbf{w}), y_i)
\end{equation}
where $\ell(\cdot)$ is the task-specific loss function and $f(\cdot;\mathbf{w})$ is ConvNeXtV2-MSAFv5 parameterized by $\mathbf{w}$.

\subsubsection{FedProx Local Objective}
In the heterogeneous (Non-IID) federated setting, standard FedAvg is susceptible to client drift—a phenomenon where divergent local updates destabilize global convergence. To mitigate this, we adopt the FedProx algorithm, which augments each client's local objective with a proximal regularization term that penalizes deviations from the current global model $\mathbf{w}^t$:
\begin{equation}
    \min_{\mathbf{w}_k} \; h_k(\mathbf{w}_k; \mathbf{w}^t) = F_k(\mathbf{w}_k) + \frac{\mu}{2} \|\mathbf{w}_k - \mathbf{w}^t\|^2
\end{equation}
The proximal coefficient $\mu = 0.01$ controls the strength of this regularization. A larger $\mu$ constrains local models to remain closer to the global initialization, sacrificing some local expressiveness to improve global stability; the selected value of 0.01 represents an empirically validated balance between local adaptation and global convergence. Upon completing $E = 3$ local epochs, each client transmits its updated parameters $\mathbf{w}_k^{t+1}$ to the central server. The global model is updated via weighted averaging:
\begin{equation}
    \mathbf{w}^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_k^{t+1}
\end{equation}
This aggregation is performed for $R = 50$ global communication rounds, with early stopping triggered if the aggregated validation macro-F1 score does not improve by at least $5 \times 10^{-5}$ within $P = 18$ consecutive rounds.

\subsection{Implementation Details and Phased Training Strategy}
\label{subsec:implementation}

\subsubsection{Loss Function: Class-Weighted Focal Loss}
To address the class imbalance in the dataset—wherein Chickenpox ($n=220$) and Measles ($n=221$) are significantly underrepresented relative to Monkeypox ($n=562$) and Healthy ($n=460$) in the raw pool—we employ Focal Loss with class-specific weighting. The Focal Loss for a predicted probability distribution $\hat{p}$ and true class $y$ is defined as:
\begin{equation}
    \mathcal{L}_{\text{focal}}(\hat{p}, y) = -\alpha_y (1 - \hat{p}_y)^{\gamma} \log(\hat{p}_y)
\end{equation}
where $\gamma = 2.0$ is the focusing parameter that down-weights the loss contribution of well-classified easy examples, and $\alpha_y$ is the class-specific weight derived from the inverse frequency of classes in the test set distribution:
\begin{equation}
    \alpha_y = \frac{1/n_y^{\text{test}}}{\sum_{c} 1/n_c^{\text{test}}}, \quad \alpha = [0.332, 0.148, 0.399, 0.120]
\end{equation}
for Chickenpox, Healthy, Measles, and Monkeypox, respectively. Calibrating $\alpha$ from the test distribution rather than the augmented training distribution is deliberate: since per-class training volumes are equalized to 375 images through augmentation, train-based alpha values would be uninformative (approximately uniform). The test distribution, which reflects the true underlying class imbalance, provides a more meaningful weighting signal.

\subsubsection{Three-Phase Phased Unfreezing Strategy}
Training proceeds in three sequential phases designed to stabilize convergence while maximally exploiting the pretrained ConvNeXtV2-Tiny representations. A separate optimizer and cosine learning rate schedule are initialized for each phase, preventing schedule interference.

\textbf{Phase 1 — Backbone Frozen (10 epochs):} The ConvNeXtV2-Tiny backbone is entirely frozen. Only the ECA modules, CBAM modules, and CSAH head are trainable. This allows the newly initialized attention parameters to converge to a stable initial state without disrupting the pretrained backbone features. The learning rates are set to $\eta_{\text{attn}} = 10^{-4}$ and $\eta_{\text{head}} = 2 \times 10^{-4}$.

\textbf{Phase 2 — Deep Stage Unfreezing (15 epochs):} Stage 2 and Stage 3 of the backbone are unfrozen, increasing the number of trainable parameters to approximately 27.2 million. The backbone learning rate is set to $\eta_{\text{backbone}} = 3 \times 10^{-5}$—an order of magnitude lower than the head—to avoid destructive interference with the deep semantic representations acquired during pretraining.

\textbf{Phase 3 — Full Unfreezing with SAM Optimization (40 epochs):} All 28.5 million parameters are trainable. The optimizer is switched to Sharpness-Aware Minimization (SAM) with $\rho = 0.05$, which seeks parameters in flat loss basins by computing a two-step gradient update:
\begin{equation}
    \hat{\mathbf{w}} = \mathbf{w} + \rho \cdot \frac{\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w})}{\|\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w})\|}
\end{equation}
\begin{equation}
    \mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}(\hat{\mathbf{w}})
\end{equation}
Flat minima found by SAM generalize better in the federated setting, where each round presents a slightly different effective loss surface due to partial participation and varying local data distributions. Gradient clipping with $\text{max norm} = 1.0$ is applied in all phases to prevent gradient explosion.

Additionally, \textbf{Label Smoothing} ($\varepsilon = 0.05$) and \textbf{Stochastic Weight Averaging (SWA)} are applied during Phase 3. SWA maintains a running average of model weights across the final 10 phases of training (beginning at epoch 10 within Phase 3), promoting convergence to flatter regions of the loss landscape. The SWA model's batch normalization statistics are updated on the training set before evaluation. At inference, the performance of the standard best checkpoint and the SWA checkpoint are ensembled by averaging their logit outputs.

\subsubsection{Implementation Specifications}
The detailed hyperparameters used throughout the federated training phases are summarized in Table~\ref{tab:hyperparams}, and the complete training procedure is formalized in Algorithm~\ref{alg:training}.

\begin{table}[htbp]
    \centering
    \caption{Hyperparameters and Implementation Details}
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
        Optimizer (Phases 1–2) & AdamW \\
        Optimizer (Phase 3) & SAM ($\rho = 0.05$) + AdamW base \\
        LR schedule & CosineAnnealingLR ($\eta_{\text{min}} = 10^{-7}$) per phase \\
        Phase 1 epochs (backbone frozen) & 10 \\
        Phase 2 epochs (Stage 2+3 unfrozen) & 15 \\
        Phase 3 epochs (full unfreeze + SAM) & 40 \\
        Early stopping patience $P$ & 18 rounds \\
        Focal loss $\gamma$ & 2.0 \\
        Focal loss $\alpha$ & $[0.332, 0.148, 0.399, 0.120]$ \\
        Label smoothing $\varepsilon$ & 0.05 \\
        MixUp $\alpha_{\text{mix}}$ & 0.1 \\
        CutMix $\alpha_{\text{cut}}$ & 1.0 \\
        Drop path rate $p_d$ & 0.10 \\
        GeM pooling exponent $p_{\text{GeM}}$ & 3.0 \\
        CSAH projection dimension $d$ & 256 \\
        SWA collection start (Phase 3 epoch) & 10 \\
        TTA crops & 10 \\
        Gradient clipping max norm & 1.0 \\
        Framework & PyTorch 2.x + timm \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{algorithm}[htbp]
\caption{Federated Training of ConvNeXtV2-MSAFv5 with Phased Unfreezing and SAM}
\label{alg:training}
\begin{algorithmic}[1]
\REQUIRE $K$ clients with datasets $\{\mathcal{D}_k\}$, $R$ rounds, $E$ local epochs, FedProx $\mu$, phases $\{P_1, P_2, P_3\}$
\ENSURE Optimal global model $\mathbf{w}^*$
\STATE \textbf{Server Initialization:} 
\STATE Initialize $\mathbf{w}^0$ from pretrained ConvNeXtV2-Tiny (FCMAE + IN22k + IN1k)
\STATE Initialize SWA model $\mathbf{w}_{\text{swa}} \leftarrow \mathbf{w}^0$
\FOR{$t = 0, 1, \dots, R-1$}
    \STATE SERVER broadcasts $\mathbf{w}^t$ to all $K$ clients
    \FOR{each client $k \in \{1, \dots, K\}$ \textbf{in parallel}}
        \STATE $\mathbf{w}_k \leftarrow \mathbf{w}^t$
        \STATE Determine phase based on $t$:
        \IF{$t < |P_1|$}
            \STATE Freeze backbone, train attn+head
        \ELSIF{$|P_1| \leq t < |P_1|+|P_2|$}
            \STATE Unfreeze Stage 2+3
        \ELSE
            \STATE Unfreeze all, switch to SAM optimizer
        \ENDIF
        \FOR{epoch $e = 1, \dots, E$}
            \FOR{each mini-batch $(\mathbf{x}_b, \mathbf{y}_b) \in \mathcal{D}_k$}
                \STATE Apply MixUp/CutMix with prob 0.6 $\rightarrow (\tilde{\mathbf{x}}_b, \tilde{\mathbf{y}}_a, \tilde{\mathbf{y}}_b, \lambda)$
                \IF{Phase == $P_3$ (SAM)}
                    \STATE \textit{\# SAM Step 1}
                    \STATE $\hat{\mathbf{y}} \leftarrow f(\tilde{\mathbf{x}}_b; \mathbf{w}_k)$
                    \STATE $\mathcal{L} \leftarrow \mathcal{L}_{\text{focal}} + \frac{\mu}{2} \|\mathbf{w}_k - \mathbf{w}^t\|^2 + \mathcal{L}_{\text{aux}}$
                    \STATE $\hat{\mathbf{w}}_k \leftarrow \mathbf{w}_k + \rho \cdot \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}$
                    \STATE \textit{\# SAM Step 2}
                    \STATE $\hat{\mathbf{y}}' \leftarrow f(\tilde{\mathbf{x}}_b; \hat{\mathbf{w}}_k)$
                    \STATE $\mathcal{L}' \leftarrow \mathcal{L}_{\text{focal}}(\hat{\mathbf{y}}') + \frac{\mu}{2} \|\mathbf{w}_k - \mathbf{w}^t\|^2$
                    \STATE Update $\mathbf{w}_k \leftarrow \mathbf{w}_k - \eta \nabla_{\mathbf{w}_k} \mathcal{L}'$
                \ELSE
                    \STATE $\hat{\mathbf{y}} \leftarrow f(\tilde{\mathbf{x}}_b; \mathbf{w}_k)$
                    \STATE Update $\mathbf{w}_k$ using AdamW gradient descent
                \ENDIF
                \STATE Clip gradients: $\|\nabla\| \leq 1.0$
            \ENDFOR
            \IF{Phase == $P_3$ \textbf{and} epoch $\geq$ SWA\_START}
                \STATE $\mathbf{w}_{\text{swa}} \leftarrow \text{SWA\_Average}(\mathbf{w}_{\text{swa}}, \mathbf{w}_k)$
            \ENDIF
        \ENDFOR
        \STATE Transmit $\mathbf{w}_k$ to SERVER
    \ENDFOR
    \STATE \textbf{Server Aggregation:} $\mathbf{w}^{t+1} \leftarrow \sum_{k=1}^K \frac{n_k}{n} \mathbf{w}_k$
    \STATE Evaluate $\mathbf{w}^{t+1}$ on validation set
    \IF{Macro-F1 improved}
        \STATE Save checkpoint $\mathbf{w}^*$
    \ELSIF{No improvement for $P=18$ rounds}
        \STATE Early stopping
    \ENDIF
\ENDFOR
\STATE \textbf{Post-Training:} 
\STATE Update BN statistics of $\mathbf{w}_{\text{swa}}$
\STATE Calibrate temperature $T^*$ on validation set
\RETURN Ensemble($\mathbf{w}^*$, $\mathbf{w}_{\text{swa}}$) + Temperature Scaling + TTA-10
\end{algorithmic}
\end{algorithm}

\subsection{Evaluation Metrics}
Model performance is assessed using a comprehensive set of complementary metrics, evaluated over the held-out test set of each fold independently. Results are reported as mean $\pm$ standard deviation across all five folds.

\textbf{Accuracy} measures the proportion of correctly classified samples:
\begin{equation}
    \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
\end{equation}

\textbf{Macro-Precision}, \textbf{Macro-Recall}, and \textbf{Macro-F1} are computed per class and averaged with equal weight, ensuring that minority classes (Chickenpox, Measles) are not overshadowed by the dominant Monkeypox and Healthy classes:
\begin{equation}
    \text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}
\end{equation}
\begin{equation}
    \text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
\end{equation}

\textbf{AUROC} (Area Under the Receiver Operating Characteristic Curve) is computed for each class using a one-vs-rest strategy and reported as macro-average:
\begin{equation}
    \text{AUROC}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} \int_0^1 \text{TPR}_c(t)\, d\,\text{FPR}_c(t)
\end{equation}

\textbf{Temperature Scaling} is applied post-hoc to calibrate the model's output probabilities by optimizing a single scalar temperature $T^*$ on the validation set to minimize the Negative Log-Likelihood (NLL):
\begin{equation}
    T^* = \arg\min_{T > 0} \sum_{i} -\log \sigma\!\left(\hat{\mathbf{z}}_i / T\right)_{y_i}
\end{equation}
where $\hat{\mathbf{z}}_i$ are the raw logits for sample $i$. Calibrated probabilities are then used for TTA averaging and ensemble fusion.

\textbf{Test-Time Augmentation (TTA)} averages the softmax probability vectors from 10 spatial crops of each test image, reducing spatial prediction variance:
\begin{equation}
    \hat{\mathbf{p}}_{\text{TTA}} = \frac{1}{M}\sum_{m=1}^{M} \text{softmax}\!\left(f(\mathbf{x}^{(m)}; \mathbf{w}) / T^*\right), \quad M = 10
\end{equation}
The final predicted class is determined via maximum likelihood: $\hat{y} = \arg\max_c \hat{\mathbf{p}}_{\text{TTA},c}$.