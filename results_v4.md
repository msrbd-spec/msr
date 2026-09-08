\section{Results and Discussion}
\label{sec:results}
% ============================================================

Results are reported as mean $\pm$ standard deviation across five
cross-validation folds. Run~1 represents the IID uniform partition, while
Run~2 represents the Non-IID heterogeneous partition with client weights
$\mathbf{w}=[0.10,0.15,0.30,0.25,0.20]$. Standard-inference macro-F1
is used for the main comparisons, while calibration and other evaluation
variants are discussed in their corresponding sections.


% ============================================================
\subsection{Main Classification Performance}
\label{subsec:results_main}

Table~\ref{tab:main_cv} summarizes the overall performance of
PoxCSAF-Net. Figure~\ref{fig:main_cm} shows representative confusion
matrices for Fold~1 under the two federated settings, and
Figure~\ref{fig:main_roc} shows the corresponding per-class ROC curves.
The full per-fold classification reports, including precision, recall, F1,
and support counts for all five folds and both settings, are provided in
Table~\ref{tab:clf_report_main}.

\begin{table}[htbp]
    \centering
    \caption{Cross-validated performance of PoxCSAF-Net across five folds.
    Accuracy, macro-precision, macro-recall, macro-F1, and macro-AUROC are
    reported as mean $\pm$ standard deviation. ``Run~1'' denotes the IID
    uniform partition; ``Run~2'' denotes the Non-IID heterogeneous partition.}
    \label{tab:main_cv}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Setting} &
        \textbf{Accuracy} &
        \textbf{Macro-Prec.} &
        \textbf{Macro-Recall} &
        \textbf{Macro-F1} &
        \textbf{AUROC} \\
        \midrule

        Run~1 Centralized
        & $0.9218 \pm 0.006$
        & $0.9125 \pm 0.003$
        & $0.9175 \pm 0.018$
        & $0.9134 \pm 0.009$
        & $0.9847 \pm 0.005$ \\

        Run~1 FL (FedProx)
        & $0.9322 \pm 0.020$
        & $0.9255 \pm 0.020$
        & $0.9271 \pm 0.028$
        & $0.9247 \pm 0.023$
        & $0.9878 \pm 0.003$ \\

        Run~2 Centralized
        & $0.9211 \pm 0.015$
        & $0.9117 \pm 0.013$
        & $0.9120 \pm 0.023$
        & $0.9102 \pm 0.017$
        & $0.9837 \pm 0.005$ \\

        Run~2 FL (FedProx)
        & $0.9204 \pm 0.018$
        & $0.9062 \pm 0.018$
        & $0.9206 \pm 0.028$
        & $0.9106 \pm 0.022$
        & $0.9858 \pm 0.005$ \\

        \bottomrule
    \end{tabular}
\end{table}

% Per-fold breakdown of the aggregate results in Table~\ref{tab:main_cv}.
\begin{table}[htbp]
    \centering
    \small
    \caption{Per-fold classification reports for the federated settings
    across all five folds. Precision (P), recall (R), F1, and support (S)
    are reported per class.}
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

% Image left:  pipeline_v2_single_gpu/Fold_1/FL_Run1_Uniform/fl/cm_Fold_1_FL_Run1_Uniform_fl.png
% Image right: pipeline_v2_single_gpu/Fold_1/FL_Run2_Heterogeneous/fl/cm_Fold_1_FL_Run2_Heterogeneous_fl.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]
    {figs/cm_Fold_1_FL_Run1_Uniform_fl.png}
    \hfill
    \includegraphics[width=0.48\linewidth]
    {figs/cm_Fold_1_FL_Run2_Heterogeneous_fl.png}
    \caption{Confusion matrices for Fold~1 under the federated protocol.
    Left: Run~1 (Uniform/IID). Right: Run~2 (Heterogeneous/Non-IID).
    Rows are true labels; columns are predicted labels.}
    \label{fig:main_cm}
\end{figure}

% Image left:  pipeline_v2_single_gpu/Fold_1/FL_Run1_Uniform/fl/roc_Fold_1_FL_Run1_Uniform_fl.png
% Image right: pipeline_v2_single_gpu/Fold_1/FL_Run2_Heterogeneous/fl/roc_Fold_1_FL_Run2_Heterogeneous_fl.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]
    {figs/roc_Fold_1_FL_Run1_Uniform_fl.png}
    \hfill
    \includegraphics[width=0.48\linewidth]
    {figs/roc_Fold_1_FL_Run2_Heterogeneous_fl.png}
    \caption{Per-class ROC curves (one-vs-rest) for Fold~1 under the
    federated protocol. Left: Run~1 (Uniform/IID). Right: Run~2
    (Heterogeneous/Non-IID). The high AUROC values indicate that the
    ranking ability is largely preserved under heterogeneity.}
    \label{fig:main_roc}
\end{figure}

Under Run~1, federated training achieved a macro-F1 of
$0.9247 \pm 0.023$, compared with $0.9134 \pm 0.009$ for centralized
training. Under Run~2, the corresponding scores were
$0.9106 \pm 0.022$ and $0.9102 \pm 0.017$. Within the federated
setting, moving from uniform to heterogeneous client sizes reduced macro-F1
by 1.4 percentage points.

Figure~\ref{fig:main_cm} shows that the heterogeneous setting produced
more confusion between Chickenpox and Monkeypox. In Fold~1, Chickenpox
precision decreased from $0.89$ to $0.80$, while more Monkeypox samples
were classified as Chickenpox. This pattern is consistent with the
focal-loss weighting scheme \cite{ref66}, which assigns higher weights to
minority classes such as Chickenpox. Nevertheless, macro-AUROC remained high
across all settings ($0.9837$--$0.9878$), indicating that the ranking
ability of the model was less affected than its decision boundaries.


% ============================================================
\subsubsection{Per-Class Performance}
\label{subsec:results_per_class}

Table~\ref{tab:per_class} summarizes the class-level results.

\begin{table}[htbp]
    \centering
    \caption{Per-class performance (mean $\pm$ standard deviation across
    five folds). ``CP'' denotes Chickenpox, ``H'' Healthy,
    ``M'' Measles, and ``MP'' Monkeypox.}
    \label{tab:per_class}
    \begin{tabular}{llccc}
        \toprule
        \textbf{Setting} & \textbf{Class} &
        \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
        \midrule

        \multirow{4}{*}{Run~1 Centralized}
        & CP & $0.803 \pm 0.015$ & $0.886 \pm 0.087$ & $0.840 \pm 0.036$ \\
        & H  & $0.974 \pm 0.019$ & $0.950 \pm 0.024$ & $0.961 \pm 0.010$ \\
        & M  & $0.954 \pm 0.022$ & $0.923 \pm 0.031$ & $0.938 \pm 0.007$ \\
        & MP & $0.919 \pm 0.017$ & $0.912 \pm 0.018$ & $0.915 \pm 0.005$ \\
        \midrule

        \multirow{4}{*}{Run~1 FL (FedProx)}
        & CP & $0.829 \pm 0.064$ & $0.890 \pm 0.092$ & $0.855 \pm 0.058$ \\
        & H  & $0.980 \pm 0.005$ & $0.965 \pm 0.014$ & $0.973 \pm 0.009$ \\
        & M  & $0.967 \pm 0.010$ & $0.932 \pm 0.025$ & $0.949 \pm 0.010$ \\
        & MP & $0.926 \pm 0.038$ & $0.921 \pm 0.038$ & $0.922 \pm 0.024$ \\
        \midrule

        \multirow{4}{*}{Run~2 Centralized}
        & CP & $0.821 \pm 0.042$ & $0.852 \pm 0.080$ & $0.833 \pm 0.040$ \\
        & H  & $0.970 \pm 0.029$ & $0.937 \pm 0.016$ & $0.953 \pm 0.012$ \\
        & M  & $0.936 \pm 0.043$ & $0.927 \pm 0.044$ & $0.929 \pm 0.002$ \\
        & MP & $0.921 \pm 0.027$ & $0.932 \pm 0.021$ & $0.926 \pm 0.016$ \\
        \midrule

        \multirow{4}{*}{Run~2 FL (FedProx)}
        & CP & $0.765 \pm 0.057$ & $0.871 \pm 0.109$ & $0.809 \pm 0.057$ \\
        & H  & $0.987 \pm 0.013$ & $0.965 \pm 0.008$ & $0.976 \pm 0.010$ \\
        & M  & $0.943 \pm 0.021$ & $0.959 \pm 0.022$ & $0.950 \pm 0.012$ \\
        & MP & $0.930 \pm 0.034$ & $0.886 \pm 0.039$ & $0.907 \pm 0.021$ \\

        \bottomrule
    \end{tabular}
\end{table}

Chickenpox was the most difficult class, with federated F1 scores of
$0.855$ in Run~1 and $0.809$ in Run~2. Its precision fell to $0.765$
under heterogeneous FL, indicating more false-positive Chickenpox
predictions, consistent with the higher focal-loss weight assigned to this
minority class. Healthy was the most stable class, with F1 values above $0.95$
in all settings. Measles also remained strong, while Monkeypox F1 decreased
from $0.922$ to $0.907$ between the two federated settings.


% ============================================================
\subsection{Comparison Against Baseline Architectures}
\label{subsec:results_baselines}

Table~\ref{tab:baselines} compares PoxCSAF-Net with four pretrained
baseline architectures.

\begin{table}[htbp]
    \centering
    \caption{Baseline comparison: mean $\pm$ standard deviation across
    five folds. Baselines use a simplified single-phase AdamW schedule for
    centralized training and the same FedProx protocol for federated training.
    The proposed model uses the full three-phase schedule with SAM and SWA.}
    \label{tab:baselines}
    \begin{tabular}{llccc}
        \toprule
        \textbf{Model} & \textbf{Setting} &
        \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{AUROC} \\
        \midrule

        \multicolumn{5}{l}{\textit{Run~1 (Uniform)}} \\
        EfficientNetV2-S & Centralized & $0.922 \pm 0.015$ & $0.914 \pm 0.019$ & $0.979$ \\
        EfficientNetV2-S & FL          & $0.916 \pm 0.020$ & $0.907 \pm 0.023$ & $0.989$ \\
        MobileNetV2      & Centralized & $0.908 \pm 0.020$ & $0.898 \pm 0.023$ & $0.979$ \\
        MobileNetV2      & FL          & $0.908 \pm 0.015$ & $0.898 \pm 0.017$ & $0.985$ \\
        ConvNeXtV2-Tiny (plain) & Centralized & $0.916 \pm 0.014$ & $0.906 \pm 0.019$ & $0.977$ \\
        ConvNeXtV2-Tiny (plain) & FL          & $0.930 \pm 0.010$ & $0.922 \pm 0.013$ & $0.988$ \\
        ResNet-50        & Centralized & $0.912 \pm 0.011$ & $0.904 \pm 0.013$ & $0.981$ \\
        ResNet-50        & FL          & $0.910 \pm 0.015$ & $0.902 \pm 0.018$ & $0.984$ \\
        \textbf{PoxCSAF-Net} & Centralized & $\mathbf{0.922 \pm 0.006}$ & $\mathbf{0.913 \pm 0.009}$ & $\mathbf{0.985}$ \\
        \textbf{PoxCSAF-Net} & FL          & $\mathbf{0.932 \pm 0.020}$ & $\mathbf{0.925 \pm 0.023}$ & $\mathbf{0.988}$ \\
        \midrule

        \multicolumn{5}{l}{\textit{Run~2 (Heterogeneous)}} \\
        EfficientNetV2-S & Centralized & $0.914 \pm 0.022$ & $0.902 \pm 0.027$ & $0.979$ \\
        EfficientNetV2-S & FL          & $0.914 \pm 0.011$ & $0.905 \pm 0.011$ & $0.989$ \\
        MobileNetV2      & Centralized & $0.899 \pm 0.008$ & $0.889 \pm 0.009$ & $0.984$ \\
        MobileNetV2      & FL          & $0.900 \pm 0.006$ & $0.891 \pm 0.004$ & $0.985$ \\
        ConvNeXtV2-Tiny (plain) & Centralized & $0.910 \pm 0.019$ & $0.900 \pm 0.025$ & $0.975$ \\
        ConvNeXtV2-Tiny (plain) & FL          & $0.918 \pm 0.018$ & $0.908 \pm 0.021$ & $0.988$ \\
        ResNet-50        & Centralized & $0.905 \pm 0.028$ & $0.894 \pm 0.032$ & $0.976$ \\
        ResNet-50        & FL          & $0.915 \pm 0.013$ & $0.908 \pm 0.016$ & $0.986$ \\
        \textbf{PoxCSAF-Net} & Centralized & $\mathbf{0.921 \pm 0.015}$ & $\mathbf{0.910 \pm 0.017}$ & $\mathbf{0.984}$ \\
        \textbf{PoxCSAF-Net} & FL          & $\mathbf{0.920 \pm 0.018}$ & $\mathbf{0.911 \pm 0.022}$ & $\mathbf{0.986}$ \\

        \bottomrule
    \end{tabular}
\end{table}

The centralized comparison uses different optimization schedules, whereas
the federated models follow the same FedProx protocol \cite{ref38}. Under
Run~1 FL, PoxCSAF-Net achieved the highest macro-F1
($0.925\pm0.023$), followed closely by plain ConvNeXtV2-Tiny
($0.922\pm0.013$). Under Run~2 FL, PoxCSAF-Net reached $0.911$, while
ConvNeXtV2-Tiny and ResNet-50 both achieved approximately $0.908$.
The results show that ConvNeXtV2-Tiny itself provides a strong baseline,
with a smaller additional gain from the proposed model. The difference of
$0.003$ in macro-F1 is well within the cross-fold variation and is not
statistically distinguishable.


% ============================================================
\subsection{Architecture Ablation Study}
\label{subsec:results_arch_ablation}

Table~\ref{tab:arch_ablation} summarizes the architecture ablation.
Statistical comparisons use the Wilcoxon signed-rank test \cite{ref78}.
Table~\ref{tab:clf_report_ablation} provides the per-class classification
report for Fold~1 comparing the primary configuration against the
best-performing ablation variant, and Figure~\ref{fig:ablation_cm}
compares the confusion matrices of the best-performing
\texttt{cbam\_eca\_gap} and the worst-performing \texttt{no\_gem\_pool}
configurations for a representative fold (Fold~3).

\begin{table}[htbp]
    \centering
    \caption{Architecture ablation results (mean $\pm$ standard deviation
    across five folds, federated uniform setting). $p$-values are from the
    Wilcoxon signed-rank test against \texttt{msaf\_primary}.}
    \label{tab:arch_ablation}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Config} & \textbf{Acc.} & \textbf{F1} &
        \textbf{Prec.} & \textbf{Recall} &
        \textbf{AUROC} & \textbf{$p$} \\
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

% Per-class comparison of the primary configuration against the best ablation variant.
\begin{table}[htbp]
    \centering
    \small
    \caption{Classification report for Fold~1 of the architecture ablation,
    comparing the primary configuration (\texttt{msaf\_primary}) against the
    best-performing configuration (\texttt{cbam\_eca\_gap}).}
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
    \includegraphics[width=0.48\linewidth]{figs/cm_abl_cbam_eca_gap.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/cm_abl_no_gem_pool.png}
    \caption{Confusion matrices for Fold~3 of the architecture ablation.
    Left: \texttt{cbam\_eca\_gap} (best F1 $= 0.931$). Right:
    \texttt{no\_gem\_pool} (worst F1 $= 0.907$).}
    \label{fig:ablation_cm}
\end{figure}

No configuration reached $p<0.05$. With only five paired observations,
the Wilcoxon signed-rank test has very limited statistical power, so the
ablation comparisons should be interpreted as descriptive rather than
definitive. The \texttt{cbam\_eca\_gap}
configuration achieved the highest mean F1 ($0.931\pm0.020$), compared
with $0.923\pm0.011$ for the primary configuration. This configuration
retains ECA and CBAM but uses global average pooling instead of the
cross-scale attention head, suggesting that the stage-specific attention
modules provide most of the architectural benefit and that the cross-scale
attention head has not demonstrated an additional advantage on this dataset.

The \texttt{no\_gem\_pool} configuration produced the lowest F1
($0.907\pm0.016$). GeM pooling emphasizes strongly activated spatial
regions \cite{ref58}, which may help preserve localized lesion information.
The auxiliary-head and no-stochastic-depth variants produced F1 scores of
$0.918$ and $0.926$, respectively.


% ============================================================
\subsection{Training-Strategy Ablation}
\label{subsec:results_train_ablation}

Table~\ref{tab:train_ablation} compares the focal-loss weighting schemes
and SWA. The \texttt{focal\_uniform} configuration had the lowest F1
($0.916\pm0.016$), compared with $0.923\pm0.011$ for the primary
configuration. The primary focal-loss weights place greater emphasis on
minority classes, following the role of class weighting in focal loss
\cite{ref66}. The \texttt{focal\_train\_dist} result indicates that the
specific weight derivation has minimal impact ($0.924$ vs.\ $0.923$).

\begin{table}[htbp]
    \centering
    \caption{Training-strategy ablation (mean $\pm$ standard deviation
    across five folds, federated uniform setting). $p$-values are from the
    Wilcoxon signed-rank test against \texttt{winning\_primary}.}
    \label{tab:train_ablation}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Config} & \textbf{Acc.} & \textbf{F1} &
        \textbf{Prec.} & \textbf{Recall} &
        \textbf{AUROC} & \textbf{$p$} \\
        \midrule

        \textbf{winning\_primary}
        & $0.931 \pm .012$ & $0.923 \pm .011$
        & $0.922$ & $0.929$ & $0.987$ & --- \\

        focal\_uniform
        & $0.924 \pm .016$ & $0.916 \pm .016$
        & $0.913$ & $0.925$ & $0.988$ & $0.62$ \\

        focal\_train\_dist
        & $0.931 \pm .014$ & $0.924 \pm .016$
        & $0.922$ & $0.929$ & $0.988$ & $1.00$ \\

        no\_swa
        & $0.934 \pm .018$ & $0.926 \pm .019$
        & $0.922$ & $0.933$ & $0.988$ & $1.00$ \\

        \bottomrule
    \end{tabular}
\end{table}

Removing SWA produced a slightly higher mean F1
($0.926\pm0.019$). SWA \cite{ref60} therefore did not increase mean
performance in this experiment, although the primary configuration had a
smaller standard deviation.


% ============================================================
\subsection{Calibration Analysis}
\label{subsec:results_calibration}

Table~\ref{tab:calibration} reports calibration before and after
temperature scaling \cite{ref70}. Figure~\ref{fig:reliability} shows
representative reliability diagrams (Fold~1) before and after temperature
scaling for both federated settings, illustrating the bin-wise gap between
predicted confidence and empirical accuracy.

\begin{table}[htbp]
    \centering
    \caption{Calibration metrics (mean $\pm$ standard deviation across
    five folds) before and after temperature scaling. ECE: Expected
    Calibration Error, MCE: Maximum Calibration Error, NLL:
    Negative Log-Likelihood. $T$ is the learned temperature scalar.}
    \label{tab:calibration}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Metric} &
        \textbf{Run~1 (Uniform)} &
        \textbf{Run~2 (Heterogeneous)} \\
        \midrule

        ECE$_{\text{uncal}}$ & $0.048 \pm 0.011$ & $0.047 \pm 0.016$ \\
        ECE$_{\text{cal}}$   & $0.036 \pm 0.009$ & $0.036 \pm 0.012$ \\
        MCE$_{\text{uncal}}$ & $0.513 \pm 0.133$ & $0.525 \pm 0.055$ \\
        MCE$_{\text{cal}}$   & $0.566 \pm 0.122$ & $0.484 \pm 0.053$ \\
        Brier$_{\text{uncal}}$ & $0.111 \pm 0.027$ & $0.121 \pm 0.028$ \\
        Brier$_{\text{cal}}$   & $0.107 \pm 0.024$ & $0.115 \pm 0.024$ \\
        NLL$_{\text{uncal}}$ & $0.245 \pm 0.052$ & $0.258 \pm 0.047$ \\
        NLL$_{\text{cal}}$   & $0.225 \pm 0.039$ & $0.233 \pm 0.036$ \\
        Temperature $T$      & $1.413 \pm 0.060$ & $1.430 \pm 0.050$ \\

        \bottomrule
    \end{tabular}
\end{table}

% Image top-left:  pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run1_Uniform_uncal.png
% Image top-right: pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run1_Uniform_cal.png
% Image bot-left:  pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run2_Heterogeneous_uncal.png
% Image bot-right: pipeline_v2_single_gpu/calibration/reliability_Fold_1_FL_Run2_Heterogeneous_cal.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{figs/reliability_Fold_1_FL_Run1_Uniform_uncal.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/reliability_Fold_1_FL_Run1_Uniform_cal.png}

    \vspace{2pt}

    \includegraphics[width=0.48\linewidth]{figs/reliability_Fold_1_FL_Run2_Heterogeneous_uncal.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/reliability_Fold_1_FL_Run2_Heterogeneous_cal.png}
    \caption{Reliability diagrams for Fold~1. Top row: Run~1 (Uniform).
    Bottom row: Run~2 (Heterogeneous). Left: before temperature scaling;
    Right: after. The diagonal represents perfect calibration.}
    \label{fig:reliability}
\end{figure}

Temperature scaling reduced ECE from $0.048$ to $0.036$ in Run~1 and
from $0.047$ to $0.036$ in Run~2. NLL and Brier scores also improved.
The learned temperature values were greater than 1.0, consistent with
overconfident uncalibrated predictions.

MCE did not improve consistently, increasing in Run~1 but decreasing in
Run~2. Temperature scaling therefore produced a clearer improvement in
average calibration than in the largest individual calibration error.


% ============================================================
\subsection{Explainability via Grad-CAM++}
\label{subsec:results_gradcam}

Figure~\ref{fig:gradcam_summary} shows Grad-CAM++ visualizations
\cite{ref79} for correctly and incorrectly classified samples.

% Image: pipeline_v2_single_gpu/gradcam/gradcam_summary_grid.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]
    {figs/gradcam_summary_grid.png}
    \caption{Grad-CAM++ summary grid for the best-performing fold
    (Fold~1, Run~1). Each row corresponds to a class; left columns show
    original images and right columns show Grad-CAM++ overlays.
    ``P:'' indicates the predicted class; ``C'' and ``W'' denote correct
    and incorrect predictions, respectively.}
    \label{fig:gradcam_summary}
\end{figure}

As a qualitative visualization tool, Grad-CAM++ provides suggestive but not
causal evidence about model behavior. Correctly classified images generally
showed stronger activation around visible lesion regions. Chickenpox and Monkeypox errors showed more similar
activation patterns, consistent with the confusion between these classes in
the classification results. Misclassified examples also tended to have more
diffuse activation.


% ============================================================
\subsection{Federated Learning Robustness to Data Poisoning}
\label{subsec:results_poisoning}

Table~\ref{tab:poisoning} summarizes the label-flipping experiment.
FedAvg, FedProx, and coordinate-wise trimmed mean follow
\cite{ref37,ref38,ref61}. Figure~\ref{fig:poisoning} shows macro-F1 as a
function of poisoning severity for each defense, and
Figure~\ref{fig:poisoning_cm} compares the confusion matrices for Fold~3
under clean and maximum-severity conditions.

\begin{table}[htbp]
    \centering
    \caption{Poisoning robustness: mean $\pm$ standard deviation across
    five folds under the Non-IID setting. Severity $f$ denotes the fraction
    of Client~1 labels flipped.}
    \label{tab:poisoning}
    \begin{tabular}{llcccc}
        \toprule
        \textbf{Defense} & \textbf{Severity} &
        \textbf{Accuracy} & \textbf{Macro-F1} &
        \textbf{$\Delta$F1 from clean} \\
        \midrule

        FedAvg ($\mu{=}0$)
        & 0.0 (clean) & $0.931 \pm 0.025$ & $0.922 \pm 0.028$ & --- \\
        & 0.2 & $0.927 \pm 0.015$ & $0.918 \pm 0.016$ & $-0.004$ \\
        & 0.4 & $0.920 \pm 0.017$ & $0.910 \pm 0.019$ & $-0.012$ \\
        & 0.6 & $0.927 \pm 0.013$ & $0.918 \pm 0.016$ & $-0.004$ \\
        \midrule

        FedProx ($\mu{=}0.01$)
        & 0.0 (clean) & $0.919 \pm 0.021$ & $0.911 \pm 0.025$ & --- \\
        & 0.2 & $0.928 \pm 0.018$ & $0.920 \pm 0.020$ & $+0.009$ \\
        & 0.4 & $0.922 \pm 0.018$ & $0.914 \pm 0.022$ & $+0.003$ \\
        & 0.6 & $0.922 \pm 0.016$ & $0.910 \pm 0.020$ & $-0.001$ \\
        \midrule

        FedProx + Trimmed
        & 0.0 (clean) & $0.924 \pm 0.020$ & $0.916 \pm 0.023$ & --- \\
        & 0.2 & $0.920 \pm 0.015$ & $0.912 \pm 0.019$ & $-0.004$ \\
        & 0.4 & $0.913 \pm 0.014$ & $0.905 \pm 0.017$ & $-0.011$ \\
        & 0.6 & $0.914 \pm 0.018$ & $0.904 \pm 0.020$ & $-0.012$ \\

        \bottomrule
    \end{tabular}
\end{table}

% Image: pipeline_v2_single_gpu/fig_poisoning_robustness.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{figs/fig_poisoning_robustness.png}
    \caption{Macro-F1 vs.\ poisoning severity for each defense.
    The non-monotonic trends reflect experimental variability rather than
    a systematic protective effect.}
    \label{fig:poisoning}
\end{figure}

% Image left:  pipeline_v2_single_gpu/poisoning_robustness/Fold_3/FedProx/cm_poison_Fold_3_FedProx_clean.png
% Image right: pipeline_v2_single_gpu/poisoning_robustness/Fold_3/FedProx/cm_poison_Fold_3_FedProx_sev60.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{figs/cm_fedprox_sev0.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/cm_fedprox_trimmed_sev60.png}
    \caption{Confusion matrices for Fold~3 of the poisoning experiment. Left: FedProx clean ($f = 0$). Right: FedProx + Trimmed at maximum severity ($f = 0.6$).}
    \label{fig:poisoning_cm}
\end{figure}

FedAvg achieved the highest clean F1 ($0.922\pm0.028$). The poisoning
results are non-monotonic: for example, FedProx F1 rises from $0.911$ in
the clean setting to $0.920$ at $f=0.2$, and FedAvg drops at $f=0.4$ then
partly recovers at $f=0.6$. These fluctuations should be treated as
experimental variability rather than evidence of a systematic effect. Overall,
FedProx remained within a narrow range, with F1 values of $0.920$, $0.914$,
and $0.910$ at $f=0.2$, $0.4$, and $0.6$, respectively.

The trimmed-mean variant decreased from $0.916$ in the clean setting to
$0.904$ at $f=0.6$. With five clients and $f_{\text{trim}}=0.2$, one
value is removed from each tail at each parameter coordinate, so a corrupted
update is not necessarily removed in every case.


% ============================================================
\subsection{Personalized Federated Learning: FedPer vs.\ FedProx}
\label{subsec:results_fedper}

Table~\ref{tab:fedper} compares FedPer \cite{ref69} with FedProx
\cite{ref38} under the Non-IID setting, where personalization is expected
to matter most. Figure~\ref{fig:fedper_cm} compares the Fold~1 confusion
matrices for FedProx and FedPer, showing how the personalized head shifts
the Chickenpox--Monkeypox decision boundary.

\begin{table}[htbp]
    \centering
    \caption{FedPer vs.\ FedProx under the Non-IID setting
    (per-fold and aggregate).}
    \label{tab:fedper}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Fold} &
        \textbf{FedProx Acc.} & \textbf{FedPer Acc.} &
        \textbf{FedProx F1} & \textbf{FedPer F1} &
        \textbf{FedProx AUROC} & \textbf{FedPer AUROC} \\
        \midrule

        Fold~1 & $0.945$ & $0.938$ & $0.939$ & $0.931$ & $0.991$ & $0.992$ \\
        Fold~2 & $0.934$ & $0.907$ & $0.928$ & $0.899$ & $0.988$ & $0.989$ \\
        Fold~3 & $0.924$ & $0.920$ & $0.914$ & $0.914$ & $0.977$ & $0.988$ \\
        Fold~4 & $0.896$ & $0.910$ & $0.889$ & $0.907$ & $0.986$ & $0.989$ \\
        Fold~5 & $0.903$ & $0.907$ & $0.883$ & $0.891$ & $0.986$ & $0.979$ \\
        \midrule

        \textbf{Mean}
        & $\mathbf{0.920}$ & $\mathbf{0.916}$
        & $\mathbf{0.911}$ & $\mathbf{0.908}$
        & $\mathbf{0.986}$ & $\mathbf{0.987}$ \\

        \bottomrule
    \end{tabular}
    }
\end{table}

% Image left:  pipeline_v2_single_gpu/Fold_1/FL_Run2_Heterogeneous/fl/cm_Fold_1_FL_Run2_Heterogeneous_fl.png
% Image right: pipeline_v2_single_gpu/fedper/Fold_1/cm_Fold_1_FedPer.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{figs/cm_Fold_1_FL_Run2_Heterogeneous_fl.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/cm_fedper_Fold_1_FL_Run2_Heterogeneous.png}
    \caption{Confusion matrices for Fold~1 under the Non-IID setting.
    Left: FedProx. Right: FedPer. The error patterns are similar, with
    FedPer showing slightly more Chickenpox--Monkeypox confusion.}
    \label{fig:fedper_cm}
\end{figure}

FedProx achieved a mean macro-F1 of $0.911$, compared with $0.908$ for
FedPer. FedPer performed better on Folds~4 and~5, worse on Folds~1 and~2,
and matched FedProx on Fold~3. Mean AUROC was almost identical
($0.986$ vs.\ $0.987$).

The fold-level variation is consistent with FedPer's use of client-specific
classification heads, which are trained for local client distributions
\cite{ref69}.


% ============================================================
\subsection{Privacy-Preserving FL: DP-SGD}
\label{subsec:results_dp_sgd}

Table~\ref{tab:dp_tradeoff} and Figure~\ref{fig:dp_tradeoff} summarize
the DP-SGD experiments \cite{ref68}, implemented with Opacus
\cite{ref67}. Table~\ref{tab:clf_report_dp} provides the per-class
classification report for Fold~1 comparing the no-DP baseline against
$\varepsilon=12$, and Figures~\ref{fig:dp_cm} and~\ref{fig:dp_roc} show
the corresponding confusion matrices and ROC curves.

\begin{table}[htbp]
    \centering
    \caption{DP-SGD privacy--utility trade-off
    (mean $\pm$ standard deviation across Fold~1, Fold~3, and Fold~5
    under the Non-IID setting). $\varepsilon=\infty$ denotes no DP.}
    \label{tab:dp_tradeoff}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{$\varepsilon$} &
        \textbf{Accuracy} &
        \textbf{Macro-F1} &
        \textbf{AUROC} &
        \textbf{$\Delta$F1 from no-DP} \\
        \midrule

        $\infty$ (no DP)
        & $0.932 \pm 0.016$
        & $0.922 \pm 0.021$
        & $0.987 \pm 0.005$
        & --- \\

        $12.0$
        & $0.802 \pm 0.018$
        & $0.774 \pm 0.024$
        & $0.936 \pm 0.005$
        & $-0.148$ \\

        $8.0$
        & $0.797 \pm 0.026$
        & $0.770 \pm 0.029$
        & $0.925 \pm 0.010$
        & $-0.152$ \\

        \bottomrule
    \end{tabular}
\end{table}

% Per-class comparison of no-DP vs.\ $\varepsilon=12$ on Fold~1.
\begin{table}[htbp]
    \centering
    \small
    \caption{Classification report for Fold~1 of the DP-SGD experiment,
    comparing no-DP ($\varepsilon=\infty$) against $\varepsilon=12$.}
    \label{tab:clf_report_dp}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Config} & \textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} \\
        \midrule
        \multirow{4}{*}{No DP} & CP & $0.833$ & $0.905$ & $0.868$ \\
        & H & $0.968$ & $0.989$ & $0.978$ \\
        & M & $0.976$ & $0.932$ & $0.953$ \\
        & MP & $0.963$ & $0.937$ & $0.950$ \\
        \midrule
        \multirow{4}{*}{$\varepsilon=12$} & CP & $0.560$ & $0.524$ & $0.541$ \\
        & H & $0.872$ & $0.880$ & $0.876$ \\
        & M & $0.750$ & $0.682$ & $0.714$ \\
        & MP & $0.786$ & $0.775$ & $0.780$ \\
        \bottomrule
    \end{tabular}
\end{table}

% Image left:  pipeline_v2_single_gpu/dp_sgd/Fold_1/noDP/cm_dp_Fold_1_noDP.png
% Image right: pipeline_v2_single_gpu/dp_sgd/Fold_1/eps12/cm_dp_Fold_1_eps12.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{figs/cm_dp_Fold_1_noDP.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/cm_dp_Fold_1_eps12.png}
    \caption{Confusion matrices for Fold~1 of the DP-SGD experiment.
    Left: no DP. Right: $\varepsilon=12$. DP-SGD degrades all classes,
    with the largest impact on Chickenpox.}
    \label{fig:dp_cm}
\end{figure}

% Image left:  pipeline_v2_single_gpu/dp_sgd/Fold_1/noDP/roc_dp_Fold_1_noDP.png
% Image right: pipeline_v2_single_gpu/dp_sgd/Fold_1/eps12/roc_dp_Fold_1_eps12.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{figs/roc_dp_no_dp_fold1.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/roc_dp_eps12_fold1.png}
    \caption{Per-class ROC curves for Fold~1 of the DP-SGD experiment.
    Left: no DP. Right: $\varepsilon=12$. AUROC decreases from $0.991$
    to $0.936$, indicating reduced but non-trivial ranking ability.}
    \label{fig:dp_roc}
\end{figure}

% Image: pipeline_v2_single_gpu/fig_dp_tradeoff.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figs/fig_dp_tradeoff.png}
    \caption{DP-SGD privacy--utility trade-off. Error bars indicate
    $\pm1$ standard deviation across three folds.}
    \label{fig:dp_tradeoff}
\end{figure}

DP-SGD caused the largest performance reduction among the additional
experiments. Macro-F1 decreased from $0.922\pm0.021$ without DP to
$0.774\pm0.024$ at $\varepsilon=12$ and $0.770\pm0.029$ at
$\varepsilon=8$.

AUROC decreased less sharply, from $0.987$ without DP to $0.936$ at
$\varepsilon=12$ and $0.925$ at $\varepsilon=8$. The results therefore
show a substantial utility cost under the privacy settings used here,
although some ranking ability was retained.


% ============================================================

\subsection{External Validation}
\label{subsec:results_external}

To assess cross-dataset generalization, we additionally evaluate the model
on an external dataset. Table~\ref{tab:external} summarizes external
validation on the MPox-Vision dataset \cite{ref77}, with confidence
intervals reported using the Wilson method \cite{ref73}.
Table~\ref{tab:clf_report_external} provides the per-class classification
report, and Figures~\ref{fig:external_cm} and~\ref{fig:external_roc} show
the corresponding confusion matrices and ROC curves.

\begin{table}[htbp]
    \centering
    \caption{External validation on the MPox-Vision dataset
    (570 images; Chickenpox, Measles, and Monkeypox).}
    \label{tab:external}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Variant} &
        \textbf{Accuracy} &
        \textbf{95\% CI} &
        \textbf{Macro-F1} &
        \textbf{$n$} \\
        \midrule

        Uncorrected
        & $0.881$
        & $[0.852,0.905]$
        & $0.878$
        & 570 \\

        Class-reweighted
        & $0.888$
        & $[0.859,0.911]$
        & $0.886$
        & 570 \\

        \bottomrule
    \end{tabular}
\end{table}

% Per-class breakdown of the external validation results.
\begin{table}[htbp]
    \centering
    \small
    \caption{Per-class external validation results (uncorrected
    predictions). P: precision, R: recall, F1, S: support.}
    \label{tab:clf_report_external}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} & \textbf{S} \\
        \midrule
        Chickenpox & $0.815$ & $0.870$ & $0.842$ & $169$ \\
        Measles & $0.913$ & $0.879$ & $0.896$ & $182$ \\
        Monkeypox & $0.913$ & $0.889$ & $0.901$ & $219$ \\
        \bottomrule
    \end{tabular}
\end{table}

% Image left:  pipeline_v2_single_gpu/external_validation/roc_external_uncorrected.png
% Image right: pipeline_v2_single_gpu/external_validation/roc_external_prior_corrected.png
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]{figs/roc_external_uncorrected.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{figs/roc_external_prior_corrected.png}
    \caption{Per-class ROC curves for external validation.
    Left: uncorrected. Right: class-reweighted. AUROC remains above $0.95$
    for all classes, indicating preserved ranking ability on the external set.}
    \label{fig:external_roc}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.48\linewidth]
    {figs/cm_external_uncorrected.png}
    \hfill
    \includegraphics[width=0.48\linewidth]
    {figs/cm_external_prior_corrected.png}
    % NOTE: Filename retains legacy "prior_corrected" name; the procedure is
    % now described as class reweighting (see Methodology).
    \caption{Confusion matrices for external validation.
    Left: uncorrected predictions. Right: class-reweighted predictions.}
    \label{fig:external_cm}
\end{figure}

External validation produced an uncorrected accuracy of $88.1\%$
(95\% CI: $85.2$--$90.5\%$) and macro-F1 of $0.878$. Class reweighting
increased accuracy to $88.8\%$ and macro-F1 to $0.886$. Because the
reweighting uses the focal-loss class weights rather than a formal
prior-shift estimator, this result should be interpreted as exploratory.

The external results remained approximately $3$--$5$ percentage points
below the internal cross-validation performance (internal accuracy
$\sim92$--$93\%$, external accuracy $88.1$--$88.8\%$).
Differences in image source, acquisition conditions, and lesion
presentation may contribute to this gap. Still, the model retained close to
$88\%$ accuracy without fine-tuning on MPox-Vision.


\subsection{Discussion}
\label{subsec:discussion}

PoxCSAF-Net achieved its highest macro-F1 under uniform federated training
($0.925$), while the heterogeneous setting produced $0.911$. The
class-level results show that Chickenpox was the most sensitive to this
change, whereas Healthy and Measles remained relatively stable.

The baseline and ablation results show that ConvNeXtV2-Tiny contributes
strongly to the overall performance. PoxCSAF-Net slightly outperformed the
plain backbone in the main FL comparison, while the ECA+CBAM configuration
produced the highest mean F1 in the architecture ablation. None of the
ablation differences reached statistical significance.

The additional experiments highlight different practical trade-offs.
Temperature scaling improved calibration, FedPer remained close to FedProx,
and the poisoning experiments produced relatively small and non-monotonic
changes across the tested conditions, though FedProx was not designed as a
poisoning-defense method. In contrast, DP-SGD caused a marked loss in macro-F1.
External performance was also lower than the internal cross-validation
results, showing that cross-dataset generalization remains more difficult
than internal evaluation.


% ============================================================
\subsection{Limitations and Future Work}
\label{subsec:limitations}

The study has several limitations. The dataset contains only 1{,}463 images,
with relatively few Chickenpox and Measles samples. Statistical comparisons
are based on five folds and a single random seed, limiting the precision of
small between-model differences.

The centralized baselines also use a simpler optimization schedule than
PoxCSAF-Net. DP-SGD was evaluated on only three folds because of its
computational cost. External validation covers one dataset and only three
classes, while FedPer was evaluated on an aggregated test set rather than
per-client, which is the natural evaluation setting for personalized FL.
The aggregated evaluation may not reflect the per-client benefits of
personalization. The poisoning experiment considers label flipping on only
one client, which holds the smallest data share in the Non-IID setting.

Future work should therefore examine larger and more balanced datasets,
multiple random seeds, broader external validation, client-specific
personalization, additional DP experiments, and more varied poisoning
scenarios.
