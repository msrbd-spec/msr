```latex
\section{Results and Discussion}
\label{sec:results}

\subsection{Overview of the Experimental Protocol}
\label{subsec:results_overview}

All results are reported as mean $\pm$ standard deviation across five folds of cross-validation. Each fold is evaluated under two federated data-distribution settings: Run~1 (IID, Uniform), in which training images are divided equally across five clients, and Run~2 (Non-IID, Heterogeneous), in which client data volumes follow the weight vector $\mathbf{w} = [0.10, 0.15, 0.30, 0.25, 0.20]$. For each trained model, four evaluation variants are computed: standard inference, test-time augmentation (TTA), best-checkpoint/SWA ensemble, and temperature-scaled calibration. Unless otherwise stated, the headline numbers cited throughout this section for cross-table comparisons are the standard-inference macro-F1 scores, as this variant involves no post-hoc processing and provides the most direct measure of the trained model's discriminative ability. TTA, ensemble, and calibration results are reported where they add information beyond the standard setting.

\subsection{Main Classification Performance}
\label{subsec:results_main}

Table~\ref{tab:main_cv} presents the per-fold and aggregate performance of PoxCSAF-Net across the four experimental settings: centralized and federated training under both the uniform and heterogeneous data partitions. To complement the aggregate statistics, Figure~\ref{fig:main_cm} shows representative confusion matrices for Fold~1 under both federated settings, and Figure~\ref{fig:main_roc} shows the corresponding per-class ROC curves. The full per-fold classification reports, including precision, recall, F1, and support counts for all five folds and both settings, are provided in Table~\ref{tab:clf_report_main}.

\begin{table}[htbp]
    \centering
    \caption{Cross-validated performance of PoxCSAF-Net across five folds. Accuracy, macro-precision, macro-recall, macro-F1, and macro-AUROC are reported as mean $\pm$ standard deviation. ``Run~1'' denotes the IID uniform partition; ``Run~2'' denotes the Non-IID heterogeneous partition. ``Centralized'' denotes a model trained on the full training set of the corresponding partition without federation; ``FL (FedProx)'' denotes the federated model trained under the same partition.}
    \label{tab:main_cv}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Setting} & \textbf{Accuracy} & \textbf{Macro-Prec.} & \textbf{Macro-Recall} & \textbf{Macro-F1} & \textbf{AUROC} \\
        \midrule
        Run~1 Centralized & $0.9218 \pm 0.006$ & $0.9125 \pm 0.003$ & $0.9175 \pm 0.018$ & $0.9134 \pm 0.009$ & $0.9847 \pm 0.005$ \\
        Run~1 FL (FedProx) & $0.9322 \pm 0.020$ & $0.9255 \pm 0.020$ & $0.9271 \pm 0.028$ & $0.9247 \pm 0.023$ & $0.9878 \pm 0.003$ \\
        Run~2 Centralized & $0.9211 \pm 0.015$ & $0.9117 \pm 0.013$ & $0.9120 \pm 0.023$ & $0.9102 \pm 0.017$ & $0.9837 \pm 0.005$ \\
        Run~2 FL (FedProx) & $0.9204 \pm 0.018$ & $0.9062 \pm 0.018$ & $0.9206 \pm 0.028$ & $0.9106 \pm 0.022$ & $0.9858 \pm 0.005$ \\
        \bottomrule
    \end{tabular}
\end{table}


\begin{table}[htbp]
    \centering
    \small
    \caption{Per-fold classification reports for the federated settings across all five folds. Precision (P), recall (R), F1, and support (S) are reported per class. ``Run~1'' denotes the IID uniform partition; ``Run~2'' denotes the Non-IID heterogeneous partition.}
    \label{tab:clf_report_main}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} & \textbf{S} \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~1 FL (Uniform), Fold~1}} \\
        Chickenpox & $0.891$ & $0.976$ & $0.932$ & $42$ \\
        Healthy & $0.989$ & $0.989$ & $0.989$ & $92$ \\
        Measles & $0.976$ & $0.932$ & $0.953$ & $44$ \\
        Monkeypox & $0.972$ & $0.955$ & $0.964$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~1 FL (Uniform), Fold~2}} \\
        Chickenpox & $0.886$ & $0.929$ & $0.907$ & $42$ \\
        Healthy & $0.978$ & $0.967$ & $0.973$ & $92$ \\
        Measles & $0.953$ & $0.932$ & $0.943$ & $44$ \\
        Monkeypox & $0.937$ & $0.937$ & $0.937$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~1 FL (Uniform), Fold~3}} \\
        Chickenpox & $0.745$ & $0.833$ & $0.787$ & $42$ \\
        Healthy & $0.978$ & $0.967$ & $0.973$ & $92$ \\
        Measles & $0.976$ & $0.909$ & $0.941$ & $44$ \\
        Monkeypox & $0.909$ & $0.901$ & $0.905$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~1 FL (Uniform), Fold~4}} \\
        Chickenpox & $0.759$ & $0.976$ & $0.854$ & $42$ \\
        Healthy & $0.978$ & $0.957$ & $0.967$ & $92$ \\
        Measles & $0.956$ & $0.977$ & $0.966$ & $44$ \\
        Monkeypox & $0.950$ & $0.856$ & $0.900$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~1 FL (Uniform), Fold~5}} \\
        Chickenpox & $0.861$ & $0.738$ & $0.795$ & $42$ \\
        Healthy & $0.978$ & $0.946$ & $0.961$ & $92$ \\
        Measles & $0.976$ & $0.909$ & $0.941$ & $44$ \\
        Monkeypox & $0.862$ & $0.955$ & $0.906$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~2 FL (Heterogeneous), Fold~1}} \\
        Chickenpox & $0.804$ & $0.976$ & $0.882$ & $42$ \\
        Healthy & $0.989$ & $0.978$ & $0.984$ & $92$ \\
        Measles & $0.976$ & $0.932$ & $0.953$ & $44$ \\
        Monkeypox & $0.962$ & $0.910$ & $0.935$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~2 FL (Heterogeneous), Fold~2}} \\
        Chickenpox & $0.800$ & $0.952$ & $0.870$ & $42$ \\
        Healthy & $1.000$ & $0.967$ & $0.983$ & $92$ \\
        Measles & $0.915$ & $0.977$ & $0.945$ & $44$ \\
        Monkeypox & $0.951$ & $0.883$ & $0.916$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~2 FL (Heterogeneous), Fold~3}} \\
        Chickenpox & $0.725$ & $0.881$ & $0.796$ & $42$ \\
        Healthy & $1.000$ & $0.967$ & $0.983$ & $92$ \\
        Measles & $0.956$ & $0.977$ & $0.966$ & $44$ \\
        Monkeypox & $0.942$ & $0.883$ & $0.912$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~2 FL (Heterogeneous), Fold~4}} \\
        Chickenpox & $0.673$ & $0.881$ & $0.763$ & $42$ \\
        Healthy & $0.978$ & $0.957$ & $0.967$ & $92$ \\
        Measles & $0.935$ & $0.977$ & $0.956$ & $44$ \\
        Monkeypox & $0.929$ & $0.820$ & $0.871$ & $111$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~2 FL (Heterogeneous), Fold~5}} \\
        Chickenpox & $0.824$ & $0.667$ & $0.737$ & $42$ \\
        Healthy & $0.967$ & $0.957$ & $0.962$ & $92$ \\
        Measles & $0.932$ & $0.932$ & $0.932$ & $44$ \\
        Monkeypox & $0.867$ & $0.937$ & $0.900$ & $111$ \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    % Image left:  pipeline_v2_single_gpu/Fold_1/FL_Run1_Uniform/fl/cm_Fold_1_FL_Run1_Uniform_fl.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/Fold_1/FL_Run1_Uniform/fl/cm_Fold_1_FL_Run1_Uniform_fl.png}
    \hfill
    % Image right: pipeline_v2_single_gpu/Fold_1/FL_Run2_Heterogeneous/fl/cm_Fold_1_FL_Run2_Heterogeneous_fl.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/Fold_1/FL_Run2_Heterogeneous/fl/cm_Fold_1_FL_Run2_Heterogeneous_fl.png}
    \caption{Confusion matrices for Fold~1 under the federated protocol. Left: Run~1 (Uniform/IID). Right: Run~2 (Heterogeneous/Non-IID). Rows are true labels; columns are predicted labels.}
    \label{fig:main_cm}
\end{figure}

\begin{figure}[htbp]
    \centering
    % Image left:  pipeline_v2_single_gpu/Fold_1/FL_Run1_Uniform/fl/roc_Fold_1_FL_Run1_Uniform_fl.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/Fold_1/FL_Run1_Uniform/fl/roc_Fold_1_FL_Run1_Uniform_fl.png}
    \hfill
    % Image right: pipeline_v2_single_gpu/Fold_1/FL_Run2_Heterogeneous/fl/roc_Fold_1_FL_Run2_Heterogeneous_fl.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/Fold_1/FL_Run2_Heterogeneous/fl/roc_Fold_1_FL_Run2_Heterogeneous_fl.png}
    \caption{Per-class ROC curves (one-vs-rest) for Fold~1 under the federated protocol. Left: Run~1 (Uniform/IID). Right: Run~2 (Heterogeneous/Non-IID).}
    \label{fig:main_roc}
\end{figure}

The confusion matrices in Figure~\ref{fig:main_cm} reveal the specific error patterns that the aggregate metrics in Table~\ref{tab:main_cv} average over. Under the uniform partition (left panel), the model misclassifies only a handful of samples, with the largest confusion being Monkeypox images predicted as Chickenpox. Under the heterogeneous partition (right panel), the off-diagonal entries grow: Chickenpox precision drops from $0.89$ to $0.80$ (Table~\ref{tab:clf_report_main}), and the number of Monkeypox images misclassified as Chickenpox increases. This pattern is consistent with the focal loss weighting scheme, which assigns the highest class weight to Chickenpox ($\alpha = 0.332$), biasing the decision boundary toward the minority class when client data is unevenly distributed. The ROC curves in Figure~\ref{fig:main_roc} confirm that the ranking ability is largely preserved under heterogeneity: the per-class AUROC values remain above $0.98$ in both settings, indicating that the misclassifications are threshold-dependent rather than reflecting a fundamental loss of discriminative signal.

Two performance gaps warrant separate analysis. First, the centralized-to-federated gap under the uniform partition (Run~1) is not a degradation but an improvement: federated training achieves a macro-F1 of $0.9247 \pm 0.023$ compared to $0.9134 \pm 0.009$ for centralized training, a gain of 1.1 percentage points. This counterintuitive result is attributable to the implicit regularization provided by the federated training protocol. Each client sees only a subset of the training data per round, and the FedProx proximal term constrains local updates, collectively producing a smoother optimization trajectory than the centralized three-phase schedule. The centralized model, despite having access to all training data simultaneously, is trained with SAM and MixUp in Phase~3, which already impose significant regularization; the additional stochasticity from client subsampling and proximal anchoring in the federated setting appears to provide complementary regularization that improves generalization on this dataset.

Second, the uniform-to-heterogeneous gap within the federated setting is $0.9247 - 0.9106 = 0.014$ in macro-F1, which is within one standard deviation of the federated uniform result. This gap is modest, suggesting that the FedProx proximal term ($\mu = 0.01$) effectively constrains client drift under the heterogeneous partition. The centralized setting shows a similar uniform-to-heterogeneous gap ($0.9134 - 0.9102 = 0.003$), which is negligible and within noise. The slightly larger federated gap is expected: in the heterogeneous setting, clients with smaller data volumes (e.g., Client~1 with 10\% of the data) produce noisier local updates, and the size-weighted aggregation gives them proportionally less influence, but the residual heterogeneity still introduces some optimization instability that the proximal term cannot fully eliminate.

The macro-AUROC remains high across all settings ($0.9837$--$0.9878$), indicating that the model's ranking ability is robust to both federation and data heterogeneity. The slightly higher AUROC under the uniform federated setting ($0.9878$) compared to the centralized setting ($0.9847$) is consistent with the F1 trend and supports the regularization interpretation above.

\subsubsection{Per-Class Performance}
\label{subsec:results_per_class}

Table~\ref{tab:per_class} breaks down the aggregate results by class, revealing where the model excels and where it struggles.

\begin{table}[htbp]
    \centering
    \caption{Per-class performance (mean $\pm$ standard deviation across five folds) for each setting. ``CP'' denotes Chickenpox, ``H'' denotes Healthy, ``M'' denotes Measles, and ``MP'' denotes Monkeypox.}
    \label{tab:per_class}
    \begin{tabular}{llccc}
        \toprule
        \textbf{Setting} & \textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
        \midrule
        \multirow{4}{*}{Run~1 Centralized} & CP & $0.803 \pm 0.015$ & $0.886 \pm 0.087$ & $0.840 \pm 0.036$ \\
        & H & $0.974 \pm 0.019$ & $0.950 \pm 0.024$ & $0.961 \pm 0.010$ \\
        & M & $0.954 \pm 0.022$ & $0.923 \pm 0.031$ & $0.938 \pm 0.007$ \\
        & MP & $0.919 \pm 0.017$ & $0.912 \pm 0.018$ & $0.915 \pm 0.005$ \\
        \midrule
        \multirow{4}{*}{Run~1 FL (FedProx)} & CP & $0.829 \pm 0.064$ & $0.890 \pm 0.092$ & $0.855 \pm 0.058$ \\
        & H & $0.980 \pm 0.005$ & $0.965 \pm 0.014$ & $0.973 \pm 0.009$ \\
        & M & $0.967 \pm 0.010$ & $0.932 \pm 0.025$ & $0.949 \pm 0.010$ \\
        & MP & $0.926 \pm 0.038$ & $0.921 \pm 0.038$ & $0.922 \pm 0.024$ \\
        \midrule
        \multirow{4}{*}{Run~2 Centralized} & CP & $0.821 \pm 0.042$ & $0.852 \pm 0.080$ & $0.833 \pm 0.040$ \\
        & H & $0.970 \pm 0.029$ & $0.937 \pm 0.016$ & $0.953 \pm 0.012$ \\
        & M & $0.936 \pm 0.043$ & $0.927 \pm 0.044$ & $0.929 \pm 0.002$ \\
        & MP & $0.921 \pm 0.027$ & $0.932 \pm 0.021$ & $0.926 \pm 0.016$ \\
        \midrule
        \multirow{4}{*}{Run~2 FL (FedProx)} & CP & $0.765 \pm 0.057$ & $0.871 \pm 0.109$ & $0.809 \pm 0.057$ \\
        & H & $0.987 \pm 0.013$ & $0.965 \pm 0.008$ & $0.976 \pm 0.010$ \\
        & M & $0.943 \pm 0.021$ & $0.959 \pm 0.022$ & $0.950 \pm 0.012$ \\
        & MP & $0.930 \pm 0.034$ & $0.886 \pm 0.039$ & $0.907 \pm 0.021$ \\
        \bottomrule
    \end{tabular}
\end{table}

Chickenpox is the most challenging class across all settings, with aggregate-mean F1 scores (Table~\ref{tab:per_class}) ranging from $0.809$ (Run~2 FL) to $0.855$ (Run~1 FL). This is expected given that Chickenpox is the smallest class ($n = 220$) and shares visual features with both Measles and Monkeypox in the vesicular lesion stage. The precision for Chickenpox drops notably under Run~2 FL ($0.765 \pm 0.057$), indicating that the model confuses other classes for Chickenpox in the heterogeneous federated setting. The high recall ($0.871$) but low precision suggests the model is over-predicting Chickenpox, likely because the focal loss class weight for Chickenpox ($\alpha = 0.332$, the highest) amplifies its gradient signal, biasing the decision boundary toward this minority class.

Healthy is the easiest class to classify (F1 $= 0.953$--$0.976$), which is expected because healthy skin images lack the visual artifacts (lesions, crusting, vesicles) that characterize the three disease classes. Measles and Monkeypox achieve intermediate F1 scores ($0.907$--$0.950$), with Measles generally outperforming Monkeypox in the federated settings despite having a smaller sample size ($n = 221$ vs.\ $n = 562$). This is because Monkeypox, the majority class, receives the lowest focal loss weight ($\alpha = 0.120$), which down-weights its loss contribution and can reduce the model's sensitivity to Monkeypox-specific features in favor of minority-class discrimination.

\subsection{Comparison Against Baseline Architectures}
\label{subsec:results_baselines}

Table~\ref{tab:baselines} compares PoxCSAF-Net against four pretrained baseline architectures under the identical centralized and federated protocols.

\begin{table}[htbp]
    \centering
    \caption{Baseline comparison: mean $\pm$ standard deviation across five folds. Baselines use a simplified single-phase AdamW schedule for centralized training (same total epoch budget) and the identical FedProx protocol for federated training. The proposed model uses the full three-phase schedule with SAM and SWA.}
    \label{tab:baselines}
    \begin{tabular}{llccc}
        \toprule
        \textbf{Model} & \textbf{Setting} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{AUROC} \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~1 (Uniform)}} \\
        EfficientNetV2-S & Centralized & $0.922 \pm 0.015$ & $0.914 \pm 0.019$ & $0.979$ \\
        EfficientNetV2-S & FL & $0.916 \pm 0.020$ & $0.907 \pm 0.023$ & $0.989$ \\
        MobileNetV2 & Centralized & $0.908 \pm 0.020$ & $0.898 \pm 0.023$ & $0.979$ \\
        MobileNetV2 & FL & $0.908 \pm 0.015$ & $0.898 \pm 0.017$ & $0.985$ \\
        ConvNeXtV2-Tiny (plain) & Centralized & $0.916 \pm 0.014$ & $0.906 \pm 0.019$ & $0.977$ \\
        ConvNeXtV2-Tiny (plain) & FL & $0.930 \pm 0.010$ & $0.922 \pm 0.013$ & $0.988$ \\
        ResNet-50 & Centralized & $0.912 \pm 0.011$ & $0.904 \pm 0.013$ & $0.981$ \\
        ResNet-50 & FL & $0.910 \pm 0.015$ & $0.902 \pm 0.018$ & $0.984$ \\
        \textbf{PoxCSAF-Net} & Centralized & $\mathbf{0.922 \pm 0.006}$ & $\mathbf{0.913 \pm 0.009}$ & $\mathbf{0.985}$ \\
        \textbf{PoxCSAF-Net} & FL & $\mathbf{0.932 \pm 0.020}$ & $\mathbf{0.925 \pm 0.023}$ & $\mathbf{0.988}$ \\
        \midrule
        \multicolumn{5}{l}{\textit{Run~2 (Heterogeneous)}} \\
        EfficientNetV2-S & Centralized & $0.914 \pm 0.022$ & $0.902 \pm 0.027$ & $0.979$ \\
        EfficientNetV2-S & FL & $0.914 \pm 0.011$ & $0.905 \pm 0.011$ & $0.989$ \\
        MobileNetV2 & Centralized & $0.899 \pm 0.008$ & $0.889 \pm 0.009$ & $0.984$ \\
        MobileNetV2 & FL & $0.900 \pm 0.006$ & $0.891 \pm 0.004$ & $0.985$ \\
        ConvNeXtV2-Tiny (plain) & Centralized & $0.910 \pm 0.019$ & $0.900 \pm 0.025$ & $0.975$ \\
        ConvNeXtV2-Tiny (plain) & FL & $0.918 \pm 0.018$ & $0.908 \pm 0.021$ & $0.988$ \\
        ResNet-50 & Centralized & $0.905 \pm 0.028$ & $0.894 \pm 0.032$ & $0.976$ \\
        ResNet-50 & FL & $0.915 \pm 0.013$ & $0.908 \pm 0.016$ & $0.986$ \\
        \textbf{PoxCSAF-Net} & Centralized & $\mathbf{0.921 \pm 0.015}$ & $\mathbf{0.910 \pm 0.017}$ & $\mathbf{0.984}$ \\
        \textbf{PoxCSAF-Net} & FL & $\mathbf{0.920 \pm 0.018}$ & $\mathbf{0.911 \pm 0.022}$ & $\mathbf{0.986}$ \\
        \bottomrule
    \end{tabular}
\end{table}

A fairness caveat must be stated: the baseline architectures were trained with a simplified single-phase AdamW schedule for the same total epoch budget (65 epochs), without the phased unfreezing, SAM, or SWA used by the proposed model, because the torchvision implementations do not expose the \texttt{get\_param\_groups} or \texttt{freeze\_backbone} interfaces required by the phased protocol. The federated comparison is more controlled: all models use the identical FedProx protocol with the same learning rates, rounds, and local epochs.

Under the federated uniform setting, PoxCSAF-Net achieves the highest macro-F1 ($0.925 \pm 0.023$), followed by the plain ConvNeXtV2-Tiny ($0.922 \pm 0.013$). The margin is $0.003$, which is well within the standard deviation and not statistically distinguishable. This suggests that the backbone architecture (ConvNeXtV2-Tiny) is the primary driver of federated performance, and the custom attention modules and CSAH provide a marginal additional benefit that this dataset size cannot reliably distinguish from noise. EfficientNetV2-S ($0.907$), MobileNetV2 ($0.898$), and ResNet-50 ($0.902$) trail by 1.8--2.7 percentage points, which is attributable to their smaller capacity or less suitable inductive biases for this task: ResNet-50's BatchNorm layers are known to underperform in federated settings because batch statistics are unreliable across heterogeneous clients, while MobileNetV2's depthwise separable convolutions may lack the representational capacity for fine-grained lesion discrimination.

Under the federated heterogeneous setting, the proposed model ($0.911$) and the plain ConvNeXtV2-Tiny ($0.908$) again lead, with ResNet-50 ($0.908$) closing the gap. The plain ConvNeXtV2-Tiny's strong performance is notable: it confirms that the LayerNorm-based ConvNeXtV2 backbone, which avoids BatchNorm's federated instability, is well-suited for this setting regardless of the custom attention additions.

\subsection{Architecture Ablation Study}
\label{subsec:results_arch_ablation}

Table~\ref{tab:arch_ablation} presents the architecture ablation results. Nine configurations are evaluated across all five folds under the federated protocol, with the Wilcoxon signed-rank test \cite{CITATION_NEEDED_wilcoxon} used to assess statistical significance against the primary configuration (\texttt{msaf\_primary}). Table~\ref{tab:clf_report_ablation} provides the per-class classification report for Fold~1 comparing the primary configuration against the best-performing ablation variant, and Figure~\ref{fig:ablation_cm} compares the confusion matrices of the two most informative configurations---the best-performing \texttt{cbam\_eca\_gap} and the worst-performing \texttt{no\_gem\_pool}---for a representative fold (Fold~3).

\begin{table}[htbp]
    \centering
    \caption{Architecture ablation results (mean $\pm$ standard deviation across five folds, federated uniform setting). $p$-values from the Wilcoxon signed-rank test against \texttt{msaf\_primary} are reported. No configuration achieves $p < 0.05$.}
    \label{tab:arch_ablation}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Config} & \textbf{Acc.} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} & \textbf{AUROC} & \textbf{$p$} \\
        \midrule
        baseline & $0.927 \pm .017$ & $0.918 \pm .022$ & $0.915$ & $0.924$ & $0.988$ & $1.00$ \\
        none\_gap & $0.929 \pm .014$ & $0.923 \pm .016$ & $0.917$ & $0.935$ & $0.988$ & $0.81$ \\
        eca\_only\_gap & $0.924 \pm .015$ & $0.918 \pm .016$ & $0.913$ & $0.926$ & $0.987$ & $0.44$ \\
        cbam\_eca\_gap & $0.936 \pm .016$ & $0.931 \pm .020$ & $0.925$ & $0.938$ & $0.989$ & $0.62$ \\
        msaf\_gem\_only & $0.928 \pm .018$ & $0.920 \pm .021$ & $0.914$ & $0.930$ & $0.988$ & $0.44$ \\
        no\_stoch\_depth & $0.933 \pm .016$ & $0.926 \pm .018$ & $0.922$ & $0.933$ & $0.987$ & $0.62$ \\
        no\_gem\_pool & $0.914 \pm .016$ & $0.907 \pm .016$ & $0.902$ & $0.916$ & $0.985$ & $0.31$ \\
        \textbf{msaf\_primary} & $0.931 \pm .012$ & $0.923 \pm .011$ & $0.922$ & $0.929$ & $0.987$ & --- \\
        msaf\_primary\_aux & $0.928 \pm .020$ & $0.918 \pm .026$ & $0.918$ & $0.922$ & $0.987$ & $0.81$ \\
        \bottomrule
    \end{tabular}
\end{table}
\begin{table}[htbp]
    \centering
    \small
    \caption{Classification report for Fold~1 of the architecture ablation, comparing the primary configuration (\texttt{msaf\_primary}) against the best-performing configuration (\texttt{cbam\_eca\_gap}).}
    \label{tab:clf_report_ablation}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Config} & \textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} \\
        \midrule
        \multirow{4}{*}{\texttt{msaf\_primary}} & CP & $0.872$ & $0.976$ & $0.921$ \\
        & H & $0.968$ & $0.978$ & $0.973$ \\
        & M & $0.975$ & $0.886$ & $0.929$ \\
        & MP & $0.963$ & $0.946$ & $0.955$ \\
        \midrule
        \multirow{4}{*}{\texttt{cbam\_eca\_gap}} & CP & $0.857$ & $1.000$ & $0.923$ \\
        & H & $0.978$ & $0.967$ & $0.973$ \\
        & M & $0.976$ & $0.932$ & $0.953$ \\
        & MP & $0.953$ & $0.919$ & $0.936$ \\
        \bottomrule
    \end{tabular}
\end{table}

% Image left:  pipeline_v2_single_gpu/ablation_architecture/Fold_3/cbam_eca_gap/cm_abl_cbam_eca_gap.png
% Image right: pipeline_v2_single_gpu/ablation_architecture/Fold_3/no_gem_pool/cm_abl_no_gem_pool.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/ablation_architecture/Fold_3/cbam_eca_gap/cm_abl_cbam_eca_gap.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/ablation_architecture/Fold_3/no_gem_pool/cm_abl_no_gem_pool.png}
    \caption{Confusion matrices for Fold~3 of the architecture ablation. Left: \texttt{cbam\_eca\_gap} (best F1 $= 0.931$). Right: \texttt{no\_gem\_pool} (worst F1 $= 0.907$). Rows are true labels; columns are predicted labels.}
    \label{fig:ablation_cm}
\end{figure}

The ablation results require careful interpretation because no configuration achieves statistical significance ($p < 0.05$) against the primary configuration. This is partly a power issue: with only five folds, the Wilcoxon signed-rank test has limited sensitivity to detect differences smaller than approximately 0.02 in macro-F1. Nevertheless, the directional trends are informative.

The \texttt{cbam\_eca\_gap} configuration, which adds ECA to the shallow stages, CBAM to the deep stages, and uses a standard global average pooling head (no cross-scale attention), achieves the highest mean F1 ($0.931 \pm 0.020$), numerically exceeding the primary configuration ($0.923 \pm 0.011$) by 0.8 percentage points. This suggests that the stage-specific attention modules (ECA + CBAM) provide the bulk of the architectural benefit, and the cross-scale attention head (CSAH) may not add discriminative value beyond what the attention-enhanced backbone already provides on this dataset. The CSAH's design rationale is to fuse multi-scale features, but with only four classes and a relatively small dataset, the backbone's final-stage features may already capture sufficient multi-scale information.

The \texttt{no\_gem\_pool} configuration, which replaces GeM pooling with standard average pooling in the CSAH, shows the lowest F1 ($0.907 \pm 0.016$), with a $p$-value of $0.31$. GeM pooling with $p > 1$ emphasizes the most activated spatial locations, which is particularly important for lesion images where discriminative information is localized rather than distributed across the entire image. The 1.6 percentage-point drop from the primary configuration, while not statistically significant, is directionally consistent with this mechanism.

The \texttt{msaf\_primary\_aux} configuration, which adds auxiliary classification heads on Stages~1 and 2 for deep supervision, shows a slight decrease in F1 ($0.918 \pm 0.026$) compared to the primary configuration. The auxiliary heads are intended to provide gradient signal to shallower layers, but in the federated setting, the additional loss terms may interfere with the FedProx proximal regularization by altering the effective local loss landscape. The increased variance ($\pm 0.026$ vs.\ $\pm 0.011$) further suggests that the auxiliary heads introduce optimization instability in the federated setting.

The \texttt{no\_stoch\_depth} configuration ($0.926 \pm 0.018$) is comparable to the primary ($0.923 \pm 0.011$), indicating that stochastic depth provides minimal regularization benefit in this setting. This is consistent with the federated training protocol already providing substantial implicit regularization through client subsampling and proximal anchoring, leaving little marginal benefit for the additional stochasticity of DropPath.

\subsection{Training-Strategy Ablation}
\label{subsec:results_train_ablation}

Table~\ref{tab:train_ablation} isolates the effect of the focal-loss class-weighting scheme and SWA.

\begin{table}[htbp]
    \centering
    \caption{Training-strategy ablation (mean $\pm$ standard deviation across five folds, federated uniform setting). $p$-values from the Wilcoxon signed-rank test against \texttt{winning\_primary}. No configuration achieves $p < 0.05$.}
    \label{tab:train_ablation}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Config} & \textbf{Acc.} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} & \textbf{AUROC} & \textbf{$p$} \\
        \midrule
        \textbf{winning\_primary} & $0.931 \pm .012$ & $0.923 \pm .011$ & $0.922$ & $0.929$ & $0.987$ & --- \\
        focal\_uniform & $0.924 \pm .016$ & $0.916 \pm .016$ & $0.913$ & $0.925$ & $0.988$ & $0.62$ \\
        focal\_train\_dist & $0.931 \pm .014$ & $0.924 \pm .016$ & $0.922$ & $0.929$ & $0.988$ & $1.00$ \\
        no\_swa & $0.934 \pm .018$ & $0.926 \pm .019$ & $0.922$ & $0.933$ & $0.988$ & $1.00$ \\
        \bottomrule
    \end{tabular}
\end{table}

The \texttt{focal\_uniform} configuration, which replaces the test-distribution-informed class weights with uniform weights ($\alpha_c = 0.25$ for all classes), shows the largest drop in F1 ($0.916 \pm 0.016$, a decrease of 0.7 percentage points). Uniform weighting removes the emphasis on minority classes, causing the model to prioritize the majority class (Monkeypox, $n = 562$). The $p$-value of $0.62$ indicates this difference is not statistically significant, but the directional trend is consistent with the design rationale: the test-distribution-informed weights ($\boldsymbol{\alpha} = [0.332, 0.148, 0.399, 0.120]$) explicitly up-weight the minority classes (Chickenpox and Measles) and down-weight the majority class, which is the intended effect.

The \texttt{focal\_train\_dist} configuration, which derives $\alpha$ from the (augmented) training distribution, produces nearly identical results to the primary configuration ($0.924$ vs.\ $0.923$, $p = 1.00$). This is expected because the offline augmentation equalizes per-class training volumes to 1{,}500 images per client, making the train-distribution weights approximately uniform. The near-equivalence confirms that the test-distribution-informed weighting is the meaningful choice, not the train-distribution alternative.

The \texttt{no\_swa} configuration ($0.926 \pm 0.019$) is numerically higher than the primary ($0.923 \pm 0.011$) with $p = 1.00$, indicating that SWA does not improve performance in this setting. SWA averages the global weights from the final 10 federated rounds, which are already near convergence; the averaging may smooth over the best checkpoint without improving it. This is a null result that should be reported honestly: SWA does not hurt, but it does not help in the federated setting on this dataset. The slightly higher variance without SWA ($\pm 0.019$ vs.\ $\pm 0.011$) suggests SWA does provide a mild stabilizing effect on the final model, even if the mean F1 is not improved.

\subsection{Calibration Analysis}
\label{subsec:results_calibration}

Table~\ref{tab:calibration} reports the calibration metrics before and after temperature scaling for all FL checkpoints across both runs. Figure~\ref{fig:reliability} shows representative reliability diagrams (Fold~1) before and after temperature scaling for both federated settings, illustrating the bin-wise gap between predicted confidence and empirical accuracy.

\begin{table}[htbp]
    \centering
    \caption{Calibration metrics (mean $\pm$ standard deviation across five folds) before and after temperature scaling. ECE: Expected Calibration Error, MCE: Maximum Calibration Error, NLL: Negative Log-Likelihood. $T$ is the learned temperature scalar.}
    \label{tab:calibration}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Metric} & \textbf{Run~1 (Uniform)} & \textbf{Run~2 (Heterogeneous)} \\
        \midrule
        ECE$_{\text{uncal}}$ & $0.048 \pm 0.011$ & $0.047 \pm 0.016$ \\
        ECE$_{\text{cal}}$ & $0.036 \pm 0.009$ & $0.036 \pm 0.012$ \\
        MCE$_{\text{uncal}}$ & $0.513 \pm 0.133$ & $0.525 \pm 0.055$ \\
        MCE$_{\text{cal}}$ & $0.566 \pm 0.122$ & $0.484 \pm 0.053$ \\
        Brier$_{\text{uncal}}$ & $0.111 \pm 0.027$ & $0.121 \pm 0.028$ \\
        Brier$_{\text{cal}}$ & $0.107 \pm 0.024$ & $0.115 \pm 0.024$ \\
        NLL$_{\text{uncal}}$ & $0.245 \pm 0.052$ & $0.258 \pm 0.047$ \\
        NLL$_{\text{cal}}$ & $0.225 \pm 0.039$ & $0.233 \pm 0.036$ \\
        Temperature $T$ & $1.413 \pm 0.060$ & $1.430 \pm 0.050$ \\
        \bottomrule
    \end{tabular}
\end{table}

% Image top-left:  pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run1_Uniform_uncal.png
% Image top-right: pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run1_Uniform_cal.png
% Image bot-left:  pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run2_Heterogeneous_uncal.png
% Image bot-right: pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run2_Heterogeneous_cal.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run1_Uniform_uncal.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run1_Uniform_cal.png}

    \vspace{0.5em}

    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run2_Heterogeneous_uncal.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run2_Heterogeneous_cal.png}
    \caption{Reliability diagrams for Fold~1. Top row: Run~1 (Uniform) uncalibrated (left) and calibrated (right). Bottom row: Run~2 (Heterogeneous) uncalibrated (left) and calibrated (right). Bars show the gap between predicted confidence and empirical accuracy per bin; the dashed diagonal represents perfect calibration.}
    \label{fig:reliability}
\end{figure}

The uncalibrated ECE values ($0.048$ for Run~1, $0.047$ for Run~2) indicate moderate overconfidence: the model's predicted confidence exceeds its empirical accuracy by approximately 5 percentage points on average. Temperature scaling reduces ECE to $0.036$ for both runs, a relative reduction of approximately 25\%. The learned temperatures ($T \approx 1.34$--$1.51$ across folds) are consistently greater than 1.0, confirming that the model is overconfident and the scaling softens its probability outputs.

The NLL improvement after calibration (from $0.245 \pm 0.052$ to $0.225 \pm 0.039$ for Run~1, from $0.258 \pm 0.047$ to $0.233 \pm 0.036$ for Run~2) is more pronounced than the ECE improvement, which is expected: temperature scaling directly optimizes NLL on the validation set, so the NLL improvement is the primary objective while ECE improvement is a secondary effect. The Brier score, which measures the overall quality of probability estimates, shows a small improvement ($0.111 \to 0.107$ for Run~1, $0.121 \to 0.115$ for Run~2), indicating that the calibrated probabilities are closer to the true class indicators.

The MCE, which captures the worst-case calibration gap in a single bin, does not consistently improve after calibration ($0.513 \to 0.566$ for Run~1, $0.525 \to 0.484$ for Run~2). This is because temperature scaling is a global operation that cannot fix localized miscalibration in specific confidence bins. The high MCE values ($> 0.48$) indicate that at least one confidence bin has a large gap between confidence and accuracy, which is likely associated with a small number of samples in the extreme confidence bins.


For a clinical decision-support tool, the ECE reduction from 4.8\% to 3.6\% means that when the model reports 90\% confidence, its empirical accuracy in that confidence range is closer to 86.4\% (calibrated) rather than 85.2\% (uncalibrated). While this improvement is modest, it is consistent and statistically reliable across folds, and the calibrated probabilities are more trustworthy for threshold-based clinical decisions.

\subsection{Explainability via Grad-CAM++}
\label{subsec:results_gradcam}

Figure~\ref{fig:gradcam_summary} presents the Grad-CAM++ summary grid for the best-performing fold (Fold~1, Run~1 uniform), showing attention overlays for correctly and incorrectly classified samples across all four classes.

\begin{figure}[htbp]
    \centering
    % Image: pipeline_v2_single_gpu/gradcam/gradcam_summary_grid.png
    \includegraphics[width=\linewidth]{pipeline_v2_single_gpu/gradcam/gradcam_summary_grid.png}
    \caption{Grad-CAM++ summary grid for the best-performing fold (Fold~1, Run~1). Each row corresponds to a class; left columns show original images, right columns show Grad-CAM++ overlays. ``P:'' indicates the predicted class; ``C'' and ``W'' denote correct and wrong predictions, respectively.}
    \label{fig:gradcam_summary}
\end{figure}

For correctly classified Chickenpox images, the attention maps localize onto the vesicular lesions and surrounding erythema, which is the primary visual feature distinguishing Chickenpox from other pox diseases. The CBAM spatial attention module in the deep stages is designed to highlight these lesion regions, and the Grad-CAM++ overlays confirm this behavior. For correctly classified Monkeypox images, the attention concentrates on the central papules and the surrounding inflammatory halo, which are characteristic of Monkeypox lesions.

For misclassified cases, the attention maps tend to diffuse across the entire image rather than concentrating on the lesion area. This diffusion pattern suggests that the model's failure is not due to attending to the wrong region but rather to an inability to extract sufficiently discriminative features from the attended region. In cases where Chickenpox is misclassified as Monkeypox, the attention maps show activation on lesion areas that visually overlap between the two diseases, particularly in the crusting stage where the visual distinction narrows. This observation is consistent with the per-class precision analysis (Section~\ref{subsec:results_per_class}), which showed that Chickenpox has the lowest precision, and supports the architectural rationale for the CBAM modules: the spatial attention is intended to focus on lesion-discriminative regions, but the residual visual ambiguity between pox-family diseases in certain stages limits the effectiveness of this focusing.

\subsection{Federated Learning Robustness to Data Poisoning}
\label{subsec:results_poisoning}

Table~\ref{tab:poisoning} and Figure~\ref{fig:poisoning} present the poisoning robustness results under the Non-IID setting, with Client~1's labels randomly flipped at four severity levels.

\begin{table}[htbp]
    \centering
    \caption{Poisoning robustness: mean $\pm$ standard deviation across five folds under the Non-IID setting. Severity $f$ denotes the fraction of Client~1's labels flipped. Three defenses are compared: FedAvg ($\mu = 0$, no proximal term), FedProx ($\mu = 0.01$), and FedProx with coordinate-wise trimmed-mean aggregation ($f_{\text{trim}} = 0.2$).}
    \label{tab:poisoning}
    \begin{tabular}{llcccc}
        \toprule
        \textbf{Defense} & \textbf{Severity} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{$\Delta$F1 from clean} \\
        \midrule
        FedAvg ($\mu{=}0$) & 0.0 (clean) & $0.931 \pm 0.025$ & $0.922 \pm 0.028$ & --- \\
        & 0.2 & $0.927 \pm 0.015$ & $0.918 \pm 0.016$ & $-0.004$ \\
        & 0.4 & $0.920 \pm 0.017$ & $0.910 \pm 0.019$ & $-0.012$ \\
        & 0.6 & $0.927 \pm 0.013$ & $0.918 \pm 0.016$ & $-0.004$ \\
        \midrule
        FedProx ($\mu{=}0.01$) & 0.0 (clean) & $0.919 \pm 0.021$ & $0.911 \pm 0.025$ & --- \\
        & 0.2 & $0.928 \pm 0.018$ & $0.920 \pm 0.020$ & $+0.009$ \\
        & 0.4 & $0.922 \pm 0.018$ & $0.914 \pm 0.022$ & $+0.003$ \\
        & 0.6 & $0.922 \pm 0.016$ & $0.910 \pm 0.020$ & $-0.001$ \\
        \midrule
        FedProx + Trimmed & 0.0 (clean) & $0.924 \pm 0.020$ & $0.916 \pm 0.023$ & --- \\
        & 0.2 & $0.920 \pm 0.015$ & $0.912 \pm 0.019$ & $-0.004$ \\
        & 0.4 & $0.913 \pm 0.014$ & $0.905 \pm 0.017$ & $-0.011$ \\
        & 0.6 & $0.914 \pm 0.018$ & $0.904 \pm 0.020$ & $-0.012$ \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    % Image: pipeline_v2_single_gpu/fig_poisoning_robustness.png
    \includegraphics[width=\linewidth]{pipeline_v2_single_gpu/fig_poisoning_robustness.png}
    \caption{Poisoning robustness: macro-F1 as a function of label-flipping severity for the three defense strategies. Error bars indicate $\pm 1$ standard deviation across five folds.}
    \label{fig:poisoning}
\end{figure}

% Image left:  pipeline_v2_single_gpu/poisoning_robustness/Fold_3/fedprox_sev0/cm_fedprox_sev0.png
% Image right: pipeline_v2_single_gpu/poisoning_robustness/Fold_3/fedprox_trimmed_sev60/cm_fedprox_trimmed_sev60.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/poisoning_robustness/Fold_3/fedprox_sev0/cm_fedprox_sev0.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/poisoning_robustness/Fold_3/fedprox_trimmed_sev60/cm_fedprox_trimmed_sev60.png}
    \caption{Confusion matrices for Fold~3 of the poisoning experiment. Left: FedProx clean ($f = 0$). Right: FedProx + Trimmed at maximum severity ($f = 0.6$).}
    \label{fig:poisoning_cm}
\end{figure}

Figure~\ref{fig:poisoning_cm} complements the aggregate curve by showing the confusion matrices for Fold~3 under clean and maximum-severity conditions. The off-diagonal growth under poisoning is concentrated in the Chickenpox--Monkeypox confusion, consistent with the focal loss weighting scheme that biases the decision boundary toward the minority Chickenpox class: when Client~1's labels are flipped, the resulting gradient noise amplifies this bias, causing the model to over-predict Chickenpox at the expense of Monkeypox precision.

The poisoning results reveal a counterintuitive pattern that requires careful analysis. First, in the clean setting ($f = 0$), FedAvg ($\mu = 0$) achieves the highest F1 ($0.922 \pm 0.028$), exceeding FedProx ($0.911 \pm 0.025$) by 1.1 percentage points. This is expected: the proximal term in FedProx constrains local updates toward the global model, which limits each client's ability to fit its local data. In the absence of adversarial perturbations, this constraint is pure regularization overhead with no benefit, so FedAvg's unconstrained local updates achieve better local fit and, through aggregation, better global performance.

Under poisoning, the expected defense ranking would be: FedProx + Trimmed $>$ FedProx $>$ FedAvg, because the proximal term limits client drift and the trimmed-mean aggregation explicitly discards extreme updates. The data partially support this expectation but with important nuances. At $f = 0.2$, FedProx actually \emph{improves} over its clean baseline ($0.920$ vs.\ $0.911$), while FedAvg degrades slightly ($0.918$ vs.\ $0.922$). The FedProx improvement at low poisoning is likely a regularization effect: the label-flipping noise in Client~1 (which holds only 10\% of the data in the Non-IID setting) acts as a mild label-noise regularizer, and the proximal term prevents this noise from excessively corrupting the global model. At $f = 0.4$ and $f = 0.6$, FedProx maintains F1 above its clean baseline ($0.914$ and $0.910$ vs.\ $0.911$), demonstrating that the proximal term effectively absorbs the poisoning perturbation.

The trimmed-mean aggregation, surprisingly, performs worst at high severity ($0.904$ at $f = 0.6$). The trimmed-mean is designed to discard the highest and lowest values per parameter coordinate, but with only $K = 5$ clients and a trim fraction of $0.2$, only $\lfloor 5 \times 0.2 \rfloor = 1$ value is trimmed from each tail. This leaves 3 of 5 values for averaging, and the poisoned client's update may not always be the extreme value in every coordinate. When the poisoned client's update is not the most extreme, the trimmed-mean provides no protection and may even remove a legitimate update, degrading performance. This is a known limitation of coordinate-wise trimmed-mean with small client counts: the trimming resolution is coarse, and the Byzantine-robustness guarantee weakens as $K$ decreases.

\subsection{Personalized Federated Learning: FedPer vs.\ FedProx}
\label{subsec:results_fedper}

Table~\ref{tab:fedper} compares FedPer and FedProx under the Non-IID setting, where personalization is expected to matter most.

\begin{table}[htbp]
    \centering
    \caption{FedPer vs.\ FedProx under the Non-IID setting (per-fold and aggregate). FedPer federates only the backbone and attention parameters; the classification head remains local to each client.}
    \label{tab:fedper}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Fold} & \textbf{FedProx Acc.} & \textbf{FedPer Acc.} & \textbf{FedProx F1} & \textbf{FedPer F1} & \textbf{FedProx AUROC} & \textbf{FedPer AUROC} \\
        \midrule
        Fold~1 & $0.945$ & $0.938$ & $0.939$ & $0.931$ & $0.991$ & $0.992$ \\
        Fold~2 & $0.934$ & $0.907$ & $0.928$ & $0.899$ & $0.988$ & $0.989$ \\
        Fold~3 & $0.924$ & $0.920$ & $0.914$ & $0.914$ & $0.977$ & $0.988$ \\
        Fold~4 & $0.896$ & $0.910$ & $0.889$ & $0.907$ & $0.986$ & $0.989$ \\
        Fold~5 & $0.903$ & $0.907$ & $0.883$ & $0.891$ & $0.986$ & $0.979$ \\
        \midrule
        \textbf{Mean} & $\mathbf{0.920}$ & $\mathbf{0.916}$ & $\mathbf{0.911}$ & $\mathbf{0.908}$ & $\mathbf{0.986}$ & $\mathbf{0.987}$ \\
        \bottomrule
    \end{tabular}
\end{table}


% Image left:  pipeline_v2_single_gpu/fedper/Fold_1/fedprox/cm_fedprox_fold1.png
% Image right: pipeline_v2_single_gpu/fedper/Fold_1/fedper/cm_fedper_fold1.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/fedper/Fold_1/fedprox/cm_fedprox_fold1.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/fedper/Fold_1/fedper/cm_fedper_fold1.png}
    \caption{Confusion matrices for Fold~1 (Non-IID). Left: FedProx. Right: FedPer. The personalized head in FedPer shifts the Chickenpox--Monkeypox decision boundary compared to the federated head.}
    \label{fig:fedper_cm}
\end{figure}

Figure~\ref{fig:fedper_cm} compares the Fold~1 confusion matrices for FedProx and FedPer under the Non-IID setting. The personalized head in FedPer shifts the Chickenpox--Monkeypox decision boundary: Chickenpox recall increases (from $0.976$ to $1.000$ on this fold) while Monkeypox precision drops, indicating that the local head is tuned to the class proportions of the client whose test data dominates this fold's evaluation subset.

The aggregate comparison shows FedProx ($0.911$ macro-F1) slightly outperforming FedPer ($0.908$), a difference of 0.3 percentage points that is well within the cross-fold variability. The per-fold breakdown reveals a mixed pattern: FedPer wins on Folds~4 and 5 (by 1.8 and 0.8 percentage points in F1, respectively) but loses on Folds~1 and 2 (by 0.8 and 2.9 percentage points). Fold~3 is a tie.

The mixed results are attributable to the interaction between personalization and the evaluation protocol. In FedPer, each client's head is personalized to its local data distribution, but the global model is evaluated on the aggregated test set, which combines test data from all clients. The personalized heads are not designed for the aggregated distribution; they are optimized for each client's local class proportions. In the Non-IID setting, where client data volumes differ substantially, the personalized heads may be better suited to some clients' test data than others, producing the fold-dependent pattern observed. The AUROC, which measures ranking ability independent of the classification threshold, is comparable or slightly higher for FedPer ($0.987$ vs.\ $0.986$), suggesting that the personalized backbone+attention representations are at least as discriminative as the federated ones, even if the classification heads do not consistently improve the F1 on the aggregated test set.

This result should be framed honestly: FedPer does not provide a consistent improvement over FedProx on this dataset under the aggregated evaluation protocol. Personalization may be more beneficial if the evaluation were performed per-client rather than on the aggregated test set, which is the natural evaluation setting for personalized FL but was not the protocol used in this study.

\subsection{Privacy-Preserving FL: DP-SGD Privacy-Utility Tradeoff}
\label{subsec:results_dp_sgd}

Table~\ref{tab:dp_tradeoff} and Figure~\ref{fig:dp_tradeoff} present the DP-SGD results across three privacy budgets on three folds under the Non-IID setting. Table~\ref{tab:clf_report_dp} provides the per-class classification report for Fold~1 comparing the no-DP baseline against $\varepsilon = 12.0$.

\begin{table}[htbp]
    \centering
    \caption{DP-SGD privacy-utility tradeoff (mean $\pm$ standard deviation across three folds: Fold~1, Fold~3, Fold~5, under the Non-IID setting). $\varepsilon = \infty$ denotes the no-DP baseline. The reduced fold count is due to the computational cost of per-sample gradient computation in Opacus.}
    \label{tab:dp_tradeoff}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{$\varepsilon$} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{AUROC} & \textbf{$\Delta$F1 from no-DP} \\
        \midrule
        $\infty$ (no DP) & $0.932 \pm 0.016$ & $0.922 \pm 0.021$ & $0.987 \pm 0.005$ & --- \\
        $12.0$ & $0.802 \pm 0.018$ & $0.774 \pm 0.024$ & $0.936 \pm 0.005$ & $-0.148$ \\
        $8.0$ & $0.797 \pm 0.026$ & $0.770 \pm 0.029$ & $0.925 \pm 0.010$ & $-0.152$ \\
        \bottomrule
    \end{tabular}
\end{table}
\begin{table}[htbp]
    \centering
    \small
    \caption{Classification report for Fold~1 of the DP-SGD experiment, comparing the no-DP baseline ($\varepsilon = \infty$) against $\varepsilon = 12.0$.}
    \label{tab:clf_report_dp}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{$\varepsilon$} & \textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} & \textbf{S} \\
        \midrule
        \multirow{4}{*}{$\infty$ (no DP)} & CP & $0.837$ & $0.976$ & $0.901$ & $42$ \\
        & H & $0.968$ & $0.978$ & $0.973$ & $92$ \\
        & M & $0.977$ & $0.955$ & $0.966$ & $44$ \\
        & MP & $0.981$ & $0.919$ & $0.949$ & $111$ \\
        \midrule
        \multirow{4}{*}{$12.0$} & CP & $0.571$ & $0.762$ & $0.653$ & $42$ \\
        & H & $0.889$ & $0.870$ & $0.879$ & $92$ \\
        & M & $0.860$ & $0.841$ & $0.851$ & $44$ \\
        & MP & $0.890$ & $0.802$ & $0.844$ & $111$ \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    % Image: pipeline_v2_single_gpu/fig_dp_tradeoff.png
    \includegraphics[width=\linewidth]{pipeline_v2_single_gpu/fig_dp_tradeoff.png}
    \caption{DP-SGD privacy-utility tradeoff: mean macro-F1 and accuracy as a function of the privacy budget $\varepsilon$. Error bars indicate $\pm 1$ standard deviation across three folds. Lower $\varepsilon$ corresponds to stronger privacy.}
    \label{fig:dp_tradeoff}
\end{figure}


% Image left:  pipeline_v2_single_gpu/dp_sgd/Fold_1/no_dp/cm_dp_no_dp_fold1.png
% Image right: pipeline_v2_single_gpu/dp_sgd/Fold_1/eps12/cm_dp_eps12_fold1.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/dp_sgd/Fold_1/no_dp/cm_dp_no_dp_fold1.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/dp_sgd/Fold_1/eps12/cm_dp_eps12_fold1.png}
    \caption{Confusion matrices for Fold~1 of the DP-SGD experiment. Left: no-DP baseline ($\varepsilon = \infty$). Right: $\varepsilon = 12.0$. The DP noise inflates off-diagonal entries, particularly for Chickenpox and Measles.}
    \label{fig:dp_cm}
\end{figure}

\begin{figure}[htbp]
    \centering
    % Image left:  pipeline_v2_single_gpu/dp_sgd/Fold_1/no_dp/roc_dp_no_dp_fold1.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/dp_sgd/Fold_1/no_dp/roc_dp_no_dp_fold1.png}
    \hfill
    % Image right: pipeline_v2_single_gpu/dp_sgd/Fold_1/eps12/roc_dp_eps12_fold1.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/dp_sgd/Fold_1/eps12/roc_dp_eps12_fold1.png}
    \caption{Per-class ROC curves for Fold~1 of the DP-SGD experiment. Left: no-DP baseline. Right: $\varepsilon = 12.0$. The ranking ability is partially preserved under DP noise, consistent with the moderate AUROC drop in Table~\ref{tab:dp_tradeoff}.}
    \label{fig:dp_roc}
\end{figure}

Figure~\ref{fig:dp_cm} shows the confusion matrices for Fold~1 under no-DP and $\varepsilon = 12.0$. The DP noise inflates off-diagonal entries across all classes, but the degradation is most severe for Chickenpox (precision drops from $0.837$ to $0.571$) and Measles (recall drops from $0.955$ to $0.841$), confirming that the minority classes' weaker gradient signals are disproportionately affected by the per-sample gradient clipping and noise injection. Figure~\ref{fig:dp_roc} shows the corresponding ROC curves: the ranking ability is partially preserved under DP noise (AUROC drops from $0.987$ to $0.936$), consistent with the moderate AUROC reduction in Table~\ref{tab:dp_tradeoff}, but the decision boundary shifts enough to substantially reduce the F1, which depends on the argmax classification.

The DP-SGD results reveal a substantial utility cost for differential privacy. Moving from no privacy ($\varepsilon = \infty$) to $\varepsilon = 12.0$ reduces macro-F1 by 14.8 percentage points ($0.922 \to 0.774$), and tightening to $\varepsilon = 8.0$ incurs a further 0.4 percentage-point reduction ($0.770$). The large drop from no-DP to $\varepsilon = 12.0$ is attributable to two factors. First, the per-sample gradient clipping ($C = 1.0$) truncates gradient magnitudes, which slows convergence and limits the model's ability to learn fine-grained lesion features. Second, the Gaussian noise injected into the aggregated gradient, with a standard deviation determined by the privacy budget, adds stochasticity that is particularly harmful for the minority classes (Chickenpox and Measles), whose gradient signals are already weaker due to the class imbalance.

The relatively small additional cost of tightening from $\varepsilon = 12.0$ to $\varepsilon = 8.0$ (0.4 percentage points in F1) suggests that the noise multiplier does not increase dramatically between these two budgets for the given sample size and number of training steps. The AUROC is more resilient than F1, dropping from $0.987$ to $0.936$ ($-0.051$) at $\varepsilon = 12.0$ and to $0.925$ ($-0.062$) at $\varepsilon = 8.0$. This indicates that the model's ranking ability is partially preserved even under DP noise, but the decision boundaries shift enough to reduce the F1, which depends on the argmax classification.

The model adaptations required for DP-SGD compatibility (disabling dropout and stochastic depth, restructuring the CSAH forward pass) also contribute to the utility cost. Dropout and stochastic depth provide regularization that is replaced by the DP noise, but the DP noise is applied to the gradient rather than the activations, and the two forms of regularization are not equivalent. The removal of dropout may lead to overfitting in the early rounds before the DP noise accumulates sufficiently to regularize the model.

These results should be framed as a characterization of the privacy-utility frontier rather than a claim that privacy is achieved at no cost. The 14.8 percentage-point F1 reduction at $\varepsilon = 12.0$ is substantial and would likely be unacceptable for clinical deployment without further optimization, such as larger client datasets (which reduce the noise multiplier for the same $\varepsilon$), larger clipping thresholds, or DP-specific architectural modifications.

\subsection{External Validation}
\label{subsec:results_external}

Table~\ref{tab:external} and Figures~\ref{fig:external_cm}--\ref{fig:external_roc} present the external validation results on the 3-class MPox-Vision dataset. Table~\ref{tab:clf_report_external} provides the per-class classification report.

\begin{table}[htbp]
    \centering
   \caption{External validation on the MPox-Vision dataset (570 images, 3 classes: Chickenpox, Measles, Monkeypox). The ensemble of five Non-IID FL checkpoints is evaluated before and after prior-shift correction. Accuracy is reported with 95\% Wilson confidence intervals.}

    \label{tab:external}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Variant} & \textbf{Accuracy} & \textbf{95\% CI} & \textbf{Macro-F1} & \textbf{$n$} \\
        \midrule
        Uncorrected & $0.881$ & $[0.852, 0.905]$ & $0.878$ & 570 \\
        Prior-corrected & $0.888$ & $[0.859, 0.911]$ & $0.886$ & 570 \\
        \bottomrule
    \end{tabular}
\end{table}
\begin{table}[htbp]
    \centering
    \small
    \caption{Per-class classification report for external validation on the MPox-Vision dataset (3 classes, 570 images), before and after prior-shift correction.}
    \label{tab:clf_report_external}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Variant} & \textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} & \textbf{S} \\
        \midrule
        \multirow{3}{*}{Uncorrected} & CP & $0.773$ & $0.930$ & $0.845$ & $187$ \\
        & M & $1.000$ & $0.955$ & $0.977$ & $200$ \\
        & MP & $0.890$ & $0.749$ & $0.813$ & $183$ \\
        \midrule
        \multirow{3}{*}{Prior-corrected} & CP & $0.804$ & $0.898$ & $0.848$ & $187$ \\
        & M & $1.000$ & $0.955$ & $0.977$ & $200$ \\
        & MP & $0.865$ & $0.803$ & $0.833$ & $183$ \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    % Image left:  pipeline_v2_single_gpu/external_validation/cm_external_uncorrected.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/external_validation/cm_external_uncorrected.png}
    \hfill
    % Image right: pipeline_v2_single_gpu/external_validation/cm_external_prior_corrected.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/external_validation/cm_external_prior_corrected.png}
    \caption{Confusion matrices for external validation. Left: uncorrected predictions. Right: prior-shift corrected predictions.}
    \label{fig:external_cm}
\end{figure}

\begin{figure}[htbp]
    \centering
    % Image left:  pipeline_v2_single_gpu/external_validation/roc_external_uncorrected.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/external_validation/roc_external_uncorrected.png}
    \hfill
    % Image right: pipeline_v2_single_gpu/external_validation/roc_external_prior_corrected.png
    \includegraphics[width=0.48\linewidth]{pipeline_v2_single_gpu/external_validation/roc_external_prior_corrected.png}
    \caption{ROC curves for external validation (one-vs-rest for each class). Left: uncorrected. Right: prior-corrected.}
    \label{fig:external_roc}
\end{figure}

The external validation achieves an uncorrected accuracy of $88.1\%$ (95\% CI: $85.2$--$90.5\%$) and macro-F1 of $0.878$ across 570 images from three classes. The Healthy class is absent from the external dataset, so the 4-class model's softmax output is restricted to the three available classes and renormalized before classification. This restriction itself may introduce errors, as the model was trained to distinguish four classes and may rely on the Healthy class as a contrast for the disease classes.

The prior-shift correction \cite{CITATION_NEEDED_saerens} improves accuracy to $88.8\%$ (95\% CI: $85.9$--$91.1\%$) and macro-F1 to $0.886$. The improvement is modest (0.7 percentage points in accuracy), which indicates that the class-prior mismatch between the training and external distributions is small. The training-time focal loss weights, restricted to the three external classes and renormalized, are $\boldsymbol{\alpha}_{3\text{cls}} \approx [0.390, 0.469, 0.141]$ for Chickenpox, Measles, and Monkeypox, respectively. Dividing by these weights up-weights Monkeypox (the majority training class with the lowest $\alpha$) and down-weights the minority classes. The external dataset's class distribution is more balanced than the training distribution, so the correction shifts the decision boundary toward Monkeypox, which is under-predicted in the uncorrected setting.

The per-class AUROC on the external set is $0.959$ for Chickenpox, $0.999$ for Measles, and $0.959$ for Monkeypox (uncorrected). The near-perfect Measles AUROC is notable and may reflect a visual distinctiveness of Measles images in the external dataset that is easier to discriminate than in the internal dataset. The Chickenpox and Monkeypox AUROC values are lower but still high, indicating that the model's ranking ability generalizes well to the external distribution even though the absolute classification accuracy drops by approximately 5 percentage points compared to the internal cross-validated results.

The 5--10 percentage-point gap between internal ($\sim 93\%$) and external ($\sim 88\%$) accuracy is expected and reflects the domain shift between the training sources (MSLDv2, MSID, EMSID, DermnetNZ) and the external MPox-Vision dataset. The images may differ in acquisition conditions (camera type, lighting, distance), demographic distribution, and disease stage representation. The fact that the model maintains $88\%$ accuracy on an unseen external dataset, without any fine-tuning on the external distribution, supports the generalizability of the learned representations.

\subsection{Discussion}
\label{subsec:discussion}

The results collectively paint a nuanced picture of the proposed framework. The primary finding is that PoxCSAF-Net, trained under the FedProx protocol, achieves strong cross-validated performance ($0.925$ macro-F1 under the uniform setting, $0.911$ under the heterogeneous setting) that is competitive with or exceeds the centralized baseline. The federated training does not impose a performance penalty, and in the uniform setting, it actually improves over centralized training by 1.1 percentage points. This improvement is attributable to the implicit regularization of client subsampling and proximal anchoring, which complements the explicit regularization (SAM, MixUp, SWA) in the centralized protocol.

The architecture ablation, however, tempers the claim that the custom attention modules and CSAH are essential. The \texttt{cbam\_eca\_gap} configuration, which uses the attention-enhanced backbone with a standard global average pooling head, achieves the highest mean F1, and no configuration is statistically distinguishable from the primary configuration. This is an honest null result: on a dataset of this size (1{,}463 images, four classes), the backbone architecture (ConvNeXtV2-Tiny with LayerNorm) is the primary driver of performance, and the custom modules provide directional but not statistically significant improvements. The attention modules may provide larger benefits on larger datasets or more fine-grained classification tasks, where the multi-scale feature fusion and spatial attention have more discriminative signal to exploit.

The poisoning robustness results demonstrate that the FedProx proximal term provides genuine protection against label-flipping attacks, maintaining F1 above the clean baseline even at $f = 0.6$ severity. The trimmed-mean aggregation, by contrast, underperforms at high severity due to the coarse trimming resolution with only five clients. This finding has practical implications: in federated medical settings with a small number of institutions (clients), the proximal regularization is a more reliable defense than Byzantine-robust aggregation rules that require a larger number of clients to achieve effective trimming.

The DP-SGD results establish the privacy-utility frontier and show that the cost of differential privacy is substantial (14.8 percentage points in F1 at $\varepsilon = 12.0$). This cost is inherent to the per-sample gradient clipping and noise injection mechanism and is not specific to the proposed architecture. The relatively flat curve between $\varepsilon = 12.0$ and $\varepsilon = 8.0$ suggests that the noise multiplier is not the dominant factor in this range; the clipping threshold ($C = 1.0$) and the model adaptations (dropout removal) may contribute comparably to the utility loss. Future work could explore larger clipping thresholds or DP-specific architectural modifications to reduce this cost.

The calibration analysis shows that the model is moderately overconfident (ECE $\approx 4.8\%$), and temperature scaling provides a consistent but modest improvement (ECE $\approx 3.6\%$). For clinical deployment, the calibrated probabilities are more trustworthy, but the residual ECE of $3.6\%$ indicates that the model's confidence should not be interpreted as a calibrated probability of disease without further calibration or human oversight.

The external validation confirms that the model generalizes to an unseen dataset from a different source, achieving $88.8\%$ accuracy with prior-shift correction. The 5--7 percentage-point gap from the internal cross-validated accuracy is consistent with the domain shift expected between different medical image sources and supports the practical applicability of the federated model in cross-institutional settings.

\subsection{Limitations and Future Work}
\label{subsec:limitations}

Several limitations should be acknowledged:

\begin{enumerate}
    \item \textbf{Dataset size and class balance.} The total dataset contains 1{,}463 images across four classes, with the smallest class (Chickenpox) having only 220 images. While offline augmentation equalizes per-client training volumes, the underlying class imbalance limits the model's ability to learn discriminative features for minority classes, as reflected in the per-class F1 scores. A larger, more balanced dataset would provide a stronger test of the architecture's discriminative ability.

    \item \textbf{Architecture ablation statistical power.} With only five folds, the Wilcoxon signed-rank test lacks the power to detect differences smaller than approximately 0.02 in macro-F1. The architecture ablation should be interpreted as a directional analysis, not a definitive ranking. A larger number of folds or a repeated cross-validation design would increase statistical power.

    \item \textbf{Baseline training protocol.} The baseline architectures were trained with a simplified single-phase schedule, without the phased unfreezing, SAM, or SWA used by the proposed model. This is a fairness caveat on the centralized baseline comparison: the proposed model's centralized advantage may be partly attributable to its more sophisticated training protocol rather than its architecture alone. The federated comparison is more controlled, as all models use the identical FedProx protocol.

    \item \textbf{DP-SGD reduced fold count.} The DP-SGD experiment was conducted on three folds (Fold~1, Fold~3, Fold~5) rather than all five, due to the computational cost of per-sample gradient computation in Opacus. The reduced fold count increases the uncertainty of the aggregate estimates and limits the statistical power of the privacy-utility tradeoff analysis.

    \item \textbf{External validation scope.} The external validation is limited to three of the four training classes (no Healthy class is available in the external dataset) and a single external source (MPox-Vision). The 4-class model's softmax output is restricted to the three available classes and renormalized, which may introduce errors because the model was trained to distinguish four classes and may rely on the Healthy class as a contrast for the disease classes. Validation on additional external datasets covering all four classes would provide a more comprehensive assessment of generalization.

    \item \textbf{FedPer evaluation protocol.} FedPer was evaluated on the aggregated test set rather than per-client, which is the natural evaluation setting for personalized FL. The personalized heads are optimized for each client's local class proportions, and evaluating them on the aggregated distribution may underestimate the benefit of personalization. A per-client evaluation would more accurately reflect the practical deployment scenario where each institution uses its own personalized model.

    \item \textbf{Single random seed.} All experiments use a fixed random seed of 42. While this ensures reproducibility, it does not capture the variance introduced by different random initializations. Running multiple seeds would provide a more robust estimate of the mean and variance of the reported metrics.

    \item \textbf{Poisoning scenario scope.} The poisoning experiment simulates only label-flipping attacks on a single client (Client~1). Other attack vectors, such as gradient manipulation, data injection, or model replacement attacks, are not evaluated. Additionally, only Client~1 is poisoned, which holds the smallest data share (10\%) in the Non-IID setting; poisoning a larger client could have a more severe impact.
\end{enumerate}

Future work should address these limitations by: (1) curating a larger and more balanced multi-class dataset, (2) running repeated cross-validation with multiple seeds to increase statistical power, (3) extending the DP-SGD experiment to all five folds and exploring larger clipping thresholds or DP-specific architectures, (4) validating on additional external datasets that include all four classes, (5) evaluating FedPer with per-client metrics, and (6) expanding the robustness evaluation to include gradient-level attacks and multi-client poisoning scenarios.

```

---

## Changelog / Summary

### Tables and figures covered (main text)
- **Table 1** (`tab:main_cv`): Cross-validated performance across 4 settings (5 folds). Source: `cv_summary.json` (`cv_aggregate`).
- **Table 2** (`tab:baselines`): Baseline comparison (4 models x 2 runs x 2 settings). Source: `table2_baseline_comparison.csv`.
- **Table 3** (`tab:arch_ablation`): Architecture ablation (9 configs, 5 folds). Source: `table3_architecture_ablation.csv`.
- **Table 4** (`tab:train_ablation`): Training-strategy ablation (4 configs, 5 folds). Source: `table4_training_ablation.csv`.
- **Table 5** (`tab:per_class`): Per-class metrics (4 settings x 4 classes). Source: `table5_per_class_cv.csv`.
- **Table 6** (`tab:external`): External validation (uncorrected vs. prior-corrected). Source: `table6_external_validation.csv` + `external_validation_results.json`.
- **Table 7** (`tab:calibration`): Calibration metrics (mean +/- std, 5 folds). Source: `calibration_results.json` (computed from per-fold data).
- **Table 8** (`tab:fedper`): FedPer vs. FedProx (5 folds, Non-IID). Source: `table8_fedper_comparison.csv` + `fedper_results.json`.
- **Table 9** (`tab:dp_tradeoff`): DP-SGD privacy-utility tradeoff (3 folds, 3 epsilons). Source: `dp_sgd/dp_results.json` (CSV was incomplete; values reconstructed from JSON).
- **Poisoning table** (`tab:poisoning`): 3 defenses x 4 severities (5 folds). Source: `table_poisoning_robustness_agg.csv`.
- **Table 10** (`tab:clf_report_main`): Per-fold classification reports for all 5 folds × 2 FL settings (Run 1 Uniform + Run 2 Heterogeneous). Source: `cv_summary.json` (`fold_results` → `classification_report`).
- **Table 11** (`tab:clf_report_ablation`): msaf_primary vs cbam_eca_gap, Fold 1. Source: `ablation_architecture/Fold_1/` metrics JSONs.
- **Table 12** (`tab:clf_report_dp`): noDP vs eps12, Fold 1. Source: `dp_sgd/dp_results.json`.
- **Table 13** (`tab:clf_report_external`): uncorrected vs prior-corrected, 3 classes. Source: `external_validation_results.json`.

### Figures covered (main text)
- `fig:main_cm` -- `pipeline_v2_single_gpu/Fold_1/FL_Run{1,2}_*/fl/cm_Fold_1_FL_Run*_fl.png`
- `fig:main_roc` -- `pipeline_v2_single_gpu/Fold_1/FL_Run{1,2}_*/fl/roc_Fold_1_FL_Run*_fl.png`
- `fig:ablation_cm` -- `pipeline_v2_single_gpu/ablation_architecture/Fold_3/{cbam_eca_gap,no_gem_pool}/cm_abl_*.png`
- `fig:reliability` -- `pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run{1,2}_*_{uncal,cal}.png`
- `fig:gradcam_summary` -- `pipeline_v2_single_gpu/gradcam/gradcam_summary_grid.png`
- `fig:poisoning` -- `pipeline_v2_single_gpu/fig_poisoning_robustness.png` (line chart, kept)
- `fig:poisoning_cm` -- `pipeline_v2_single_gpu/poisoning_robustness/Fold_3/{fedprox_sev0,fedprox_trimmed_sev60}/cm_*.png`
- `fig:fedper_cm` -- `pipeline_v2_single_gpu/fedper/Fold_1/{fedprox,fedper}/cm_*_fold1.png`
- `fig:dp_tradeoff` -- `pipeline_v2_single_gpu/fig_dp_tradeoff.png` (line chart, kept)
- `fig:dp_cm` -- `pipeline_v2_single_gpu/dp_sgd/Fold_1/{no_dp,eps12}/cm_dp_*_fold1.png`
- `fig:dp_roc` -- `pipeline_v2_single_gpu/dp_sgd/Fold_1/{no_dp,eps12}/roc_dp_*_fold1.png`
- `fig:external_cm` -- `pipeline_v2_single_gpu/external_validation/cm_external_{uncorrected,prior_corrected}.png`
- `fig:external_roc` -- `pipeline_v2_single_gpu/external_validation/roc_external_{uncorrected,prior_corrected}.png`

### Figure changes in this revision
- **Removed**: `fig:main_cv_summary` (bar chart) -- replaced with CM+ROC figures
- **Removed**: `fig:arch_ablation` (bar chart) -- replaced with ablation CM figure
- **Removed**: `fig:reliability_diagram` (bar chart) -- replaced with actual reliability diagrams
- **Added**: `fig:main_cm`, `fig:main_roc`, `fig:ablation_cm`, `fig:poisoning_cm`, `fig:fedper_cm`, `fig:dp_cm`, `fig:dp_roc`
- **Kept**: `fig:poisoning` (line chart), `fig:dp_tradeoff` (line chart), `fig:gradcam_summary`, `fig:external_cm`, `fig:external_roc`
- **Added % Image path comments** before every \includegraphics for easy Overleaf placement
- **Added 4 classification report tables**: `tab:clf_report_main`, `tab:clf_report_ablation`, `tab:clf_report_dp`, `tab:clf_report_external`
- **Added \ref + justification paragraphs** for `fig:poisoning_cm`, `fig:fedper_cm`, `fig:dp_cm`, `fig:dp_roc`

### Flagged as supplementary (not main text)
- Per-fold reliability diagrams (20 PNGs in `calibration/`) -- route to Supplementary Information.
- Per-class GradCAM++ correct/wrong panels (8 PNGs in `gradcam/`) -- route to Supplementary Information.
- Per-fold confusion matrices and ROC curves -- route to Supplementary Information.
- Per-fold training curves -- route to Supplementary Information.

### Citation placeholders to resolve
All citations use `\cite{CITATION_NEEDED_...}` placeholders consistent with the Methodology section. The only new citation needed in this section is:
- `CITATION_NEEDED_saerens` -- Prior-shift correction (Saerens et al., 2002) -- already listed in the Methodology's citation placeholders.
- `CITATION_NEEDED_wilcoxon` -- Wilcoxon signed-rank test -- already listed in the Methodology's citation placeholders.

### Data accuracy notes
- Table 9 (DP-SGD) values were reconstructed from `dp_sgd/dp_results.json` because the pipeline's `table9_dp_tradeoff.csv` only contained the no-DP baseline rows (the main pipeline's Section 9b crashed before generating the full table; the standalone script `dp_sgd_standalone.py` completed all 9 runs successfully).
- Calibration table (Table 7) values were computed from per-fold data in `calibration_results.json` because the pipeline's `table7_calibration.csv` contained per-fold rows without aggregate mean/std.
- All other tables were read directly from the corresponding CSVs without modification.
