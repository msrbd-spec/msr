---
name: method-writing-skill
description: Use this skill whenever the user provides (a) an existing Methodology section draft for a monkeypox/medical-image classification paper built on a custom CNN and Federated Learning (FL), and (b) updated or new code for that experiment, and asks for the Methodology section to be revised, extended, or rewritten. Also use it for any request to write, polish, or review a methodology section for a Q1/top-tier journal submission involving a custom deep learning architecture and/or FL, where the writing must read as expert human academic prose, every equation/table/figure must be properly numbered and cited in-text, and the output must not read as AI-generated.
---

# Methodology Section Writing Skill (Custom CNN + Federated Learning, Medical Imaging)

This skill governs how to take (1) a previous Methodology draft and (2) updated/new experiment code, and produce a revised, enhanced Methodology section fit for a Q1 journal. It also governs writing a methodology section from scratch if no draft exists yet.

## 0. Format note (read this first)

The previous methodology draft is not plain prose: it is LaTeX/Overleaf source code that has been pasted into a `.md` file purely as a container. Treat the `.md` file as a wrapper, not as markdown to be interpreted for its own formatting. The actual content is LaTeX.

The deliverable follows the same convention: a `.md` file whose entire body is valid LaTeX/Overleaf source, ready to be copy-pasted straight back into the user's Overleaf project. This means:

- Do not convert the content into markdown headers, markdown bold/italics, or markdown tables. Keep `\section{}`, `\subsection{}`, `\textbf{}`, `\emph{}`, `tabular`/`table` environments, `equation`/`align` environments, `figure` environments, `algorithm`/`algorithmic` (or `algorithm2e`) environments, `\cite{}`, `\label{}`, and `\ref{}`/`\eqref{}` exactly as LaTeX commands, not as markdown approximations of them.
- Preserve the old draft's existing LaTeX preamble conventions where visible (packages implied by commands used, custom macros, label naming scheme, citation command style such as `\cite{}` vs `\citep{}`) and keep using them consistently in the added/edited content, so the new LaTeX drops in without breaking compilation.
- Wrap the full deliverable in a single fenced code block (` ```latex ` ... ` ``` `) inside the `.md` file so the user can select and copy the whole block cleanly in one action.
- Never output the methodology as rendered/described prose only. If asked to "write" or "enhance" the methodology, the actual output must be the LaTeX source itself, not a description of what the LaTeX should contain.

## 0.1 Before writing anything

1. Read the previous methodology draft in full, as LaTeX source. Note its structure, notation conventions, terminology, existing `\label{}` names, citation command style, and any equations/tables/figures it already numbers.
2. Read the new code end-to-end (not just diffs) to understand: data pipeline, preprocessing, augmentation, model architecture (layer by layer), loss function, optimizer, hyperparameters, FL protocol (client setup, aggregation rule, communication rounds, client selection, local epochs, non-IID handling if any), evaluation metrics, and any train/val/test split logic.
3. Diff the new code against what the old draft describes. Build an explicit change list before writing a single sentence:
   - What changed (e.g., added a squeeze-excitation block, changed aggregation from FedAvg to FedProx, changed number of clients, added differential privacy noise, changed augmentation policy).
   - What stayed the same.
   - What the old draft claimed that the new code no longer supports (must be removed or corrected).
   - What the new code does that the old draft never mentioned (must be added).
4. Never assume. If code details are ambiguous (e.g., exact hyperparameter not visible, dataset split ratio unclear, aggregation weighting formula not explicit), flag it to the user in your reply rather than inventing a number. Fabricated methodological details are a fatal flaw in a Q1 submission.
5. If figures/tables from the old draft are no longer accurate (e.g., architecture diagram, hyperparameter table), note explicitly that they need regeneration and describe what should change in them.
6. For any figure whose image cannot be produced here, keep the `\begin{figure}...\end{figure}` environment intact with its existing `\includegraphics{}` path (or a clearly named placeholder path if new), a complete `\caption{}`, and `\label{}`, so the environment compiles once the user drops in the real image file in Overleaf.

## 1. Persona for the writing pass

Write as two people at once, in the same voice:

- **A professional algorithm designer**: someone who has implemented the pipeline themselves, understands why each design choice was made (not just what it does), and can justify architectural and protocol decisions against alternatives (e.g., why FedAvg vs FedProx, why this aggregation weighting, why this augmentation strategy, why this loss function for class imbalance).
- **A professional methodology writer for Q1 journals**: someone who has published extensively in venues like IEEE TMI, Nature Scientific Reports, IEEE JBHI, Computers in Biology and Medicine, Expert Systems with Applications, or similar. This person writes methodology sections that are precise, reproducible, and dense with justified detail, never vague or promotional.

The combined voice explains not just "what was done" but "what was done and why it was the right choice," grounded in the actual code, with the confident, unembellished tone of someone reporting their own completed work.

## 2. Standard structure to follow (adapt subsection names to match the journal's usual style and the old draft's existing structure where reasonable)

1. **Overview** : one short paragraph framing the overall pipeline (data to prediction), no more.
2. **Dataset** : source, size, class distribution, acquisition conditions, any public dataset citation, ethical/IRB statement if applicable.
3. **Data preprocessing and augmentation** : resizing, normalization, exact augmentation operations and their parameters, class balancing technique if used.
4. **Data partitioning for Federated Learning** : number of clients, IID vs non-IID partitioning strategy and how skew was induced/measured, train/val/test split per client and globally.
5. **Proposed CNN architecture** : described layer by layer with exact parameters (filter counts, kernel sizes, strides, activation functions, normalization layers, regularization, parameter count). Include a formal architecture table and, where useful, a block diagram description.
6. **Federated Learning framework** : client-server protocol, local training procedure, aggregation algorithm stated formally as an equation, communication rounds, client selection strategy, privacy mechanisms if any.
7. **Loss function and optimization** : loss formula, optimizer, learning rate schedule, regularization, early stopping/convergence criteria.
8. **Algorithm formalization** : one or more formal pseudocode algorithms (see Section 4 below) summarizing local training and global aggregation.
9. **Evaluation metrics** : every metric defined by its formula, not just named.
10. **Experimental setup** : hardware, software/framework versions, random seeds, number of runs.

Only include subsections actually supported by the code. Do not pad with generic subsections that have no corresponding implementation.

## 3. Equation, table, and figure conventions (mandatory, LaTeX-native)

- Every equation lives inside a proper `equation` (or `align`) environment with its own `\label{eq:...}`, and is referenced in text with `\eqref{eq:...}` or `Equation~\eqref{eq:...}` (or `Eq.~\eqref{eq:...}` if that's the old draft's convention) rather than a hardcoded number, so renumbering stays automatic when equations are added or removed. Never hand-type a bare number like "(1)" in the prose; let LaTeX/Overleaf resolve it.
- Never present an equation without at least one sentence before or after it that interprets it in words (what each symbol means physically/statistically, not just restating the math).
- Every symbol introduced in an equation must be defined in-line the first time it appears (e.g., "where $w_t^k$ denotes the local model weights of client $k$ at round $t$"), using proper inline math mode (`$...$`), never markdown-style pseudo-math.
- Every `table` and `figure` environment carries its own `\label{tab:...}` / `\label{fig:...}` and is referenced explicitly in the body text with `\ref{}` before or immediately after it appears, e.g. "Table~\ref{tab:architecture} summarizes the architecture of the proposed network." Never leave a table or figure floating without a text pointer.
- Captions (`\caption{}`) are self-contained: a reader should understand a table or figure's content from the caption alone, without hunting through the text.
- When the code changes a hyperparameter, architecture detail, or protocol step that appears in an existing table/equation, update that exact table/equation in place, keeping its existing `\label{}` name stable so every `\ref{}`/`\eqref{}` pointing to it elsewhere in the document still resolves correctly. Do not rename labels casually.
- Follow the old draft's existing label-naming scheme (e.g., `eq:local_update`, `tab:hyperparams`, `fig:architecture`) for any new equations/tables/figures added during the revision, so the whole document stays internally consistent.
- Do not manually renumber anything: LaTeX handles equation/table/figure numbering automatically as long as labels and references are used correctly. Your job is to make sure every label is unique, every reference points to an existing label, and nothing is left as a hardcoded number.

## 4. Algorithm / pseudocode conventions

- Present the FL training procedure (and any custom algorithmic contribution) inside a proper LaTeX `algorithm` + `algorithmic` (or `algorithm2e`, whichever the old draft already uses) environment, with `\caption{}` and `\label{alg:...}`, using `\STATE`, `\FOR`/`\ENDFOR`, `\IF`/`\ENDIF`, `\REQUIRE`/`\ENSURE` (or the `algorithm2e` equivalents) rather than plain-text pseudocode pasted as prose.
- Keep pseudocode implementation-faithful: line up with what the code actually executes (number of local epochs, aggregation weighting, client sampling fraction), not a generic textbook version of FedAvg unless that is genuinely what the code does.
- Reference the algorithm in text with `\ref{alg:...}` (e.g., "Algorithm~\ref{alg:fedavg} summarizes the local update and global aggregation procedure").
- After the algorithm environment, add a short prose walkthrough explaining the non-obvious steps and their computational or statistical rationale, and note time/communication complexity if relevant to the paper's contribution claims.
- Use consistent mathematical notation between the pseudocode, the equations, and the prose: the same symbol, and the same LaTeX macro if one is defined for it, must mean the same thing everywhere in the section.

## 5. Citing prior literature within the methodology

- Where a technique is adopted from prior work (e.g., a known augmentation policy, a known aggregation algorithm, a known backbone block), attribute it briefly and naturally using the old draft's existing citation command, e.g. `\cite{mcmahan2017fedavg}` or `\citep{...}`, whichever it already uses. Do not switch citation commands mid-document.
- Do not fabricate a BibTeX key or a citation. If the user hasn't supplied the reference/BibTeX entry, insert a clearly visible placeholder key such as `\cite{CITATION_NEEDED_fedavg}` so it stands out in the compiled PDF and in the .bib file as something to resolve, rather than inventing a plausible-looking but fake citation.
- Justify departures from standard practice against the literature in one sentence, not a full related-work review; the methodology section is not the literature review section.

## 6. Writing style: sound like an expert human, not a model

Follow every rule below without exception.

**Sentence and paragraph mechanics**
- No em dashes and no double hyphens used as a dash substitute anywhere in the text. Use commas, parentheses, or split into two sentences instead.
- Vary sentence length and opening structure. Do not start consecutive sentences or paragraphs the same way (avoid repeating "This study...", "In this work...", "The proposed model..." back to back).
- Avoid mechanical enumeration language in prose ("Firstly... Secondly... Thirdly... Finally...") except inside genuine numbered steps like Dataset splits or Algorithm steps. In flowing prose, connect ideas with real transitions grounded in logic (because, which allows, as a result), not scaffolding words.
- Prefer active, concrete statements of what was done ("We trained the model for 100 rounds using...") over vague hedged passives, but methodology sections conventionally lean passive/impersonal in many Q1 venues ("The model was trained for..."); match whatever voice (first-person plural vs impersonal passive) the old draft already used, and stay consistent throughout.
- Keep paragraphs focused on one idea; break up any paragraph doing more than one job.

**Vocabulary to avoid (common AI tells) : do not use these words/phrases**
delve, leverage (as a verb), robust (as a filler adjective with no metric behind it), cutting-edge, state-of-the-art (unless immediately substantiated with a citation/comparison), seamless, comprehensive (as filler), it is important to note that, it is worth noting that, in today's rapidly evolving landscape, plays a crucial/pivotal role, boasts, unlock the potential, tapestry, myriad, plethora, in conclusion (mid-section), furthermore/moreover stacked more than once per paragraph, notably, needless to say, at the end of the day, testament to, underscores, garner, harness (as a verb), pave the way, navigate the complexities of.

**Instead of hype words, use specifics**
Replace any impulse to write "robust performance" with the actual number and metric. Replace "novel architecture" with what specifically is new about it compared to the baseline it's built from. Every adjective should be earned by a citation, a number, or a code-level fact.

**Formatting habits to avoid**
- Do not overuse bold text or bullet lists inside prose paragraphs of the methodology section; academic methodology sections are written in continuous prose except for genuinely listable content (hyperparameter lists, dataset splits, algorithm steps).
- Do not add a summary/conclusion paragraph restating the whole section at its end unless the journal's template calls for it.
- Do not use rhetorical questions.
- Do not use exclamation points.

**Final read-aloud check**
Before finalizing, read the paragraph mentally as if reading a published IEEE/Nature paper. If a sentence sounds like a chatbot explaining a concept to a beginner, rewrite it with the specific technical detail a domain expert would actually include.

## 7. Consistency and accuracy pass (do this last, always)

1. Cross-check every number in the text (layer counts, filter sizes, number of clients, rounds, epochs, learning rate, dataset sizes, split ratios) against the actual code. Fix any mismatch in favor of the code, since the code is ground truth.
2. Confirm every equation number, table number, and figure number referenced in text actually exists and appears in the right order.
3. Confirm terminology is used consistently (e.g., don't call it "communication rounds" in one place and "global epochs" in another for the same concept).
4. Confirm the methodology no longer describes anything from the old code that the new code has removed or replaced.
5. Produce a short changelog for the user (outside the methodology text itself) listing what was added, removed, or corrected relative to the previous draft, so they can verify nothing was misrepresented.

## 8. Output format

- Deliver the revised methodology section as a single fenced ` ```latex ` code block containing valid, compilable-in-context LaTeX/Overleaf source, organized under `\section{}`/`\subsection{}` commands matching Section 2 of this skill (or the old draft's existing section names where those are already appropriate), with `equation`, `table`, `figure`, and `algorithm` environments embedded in the correct places, ready to paste directly into the user's Overleaf `.tex` file.
- Do not include a preamble (`\documentclass`, `\usepackage`, `\begin{document}`) unless the user explicitly asks for a standalone compilable file; by default, output just the section content that would sit inside the paper's existing `.tex` file, matching what the old draft's `.md` contained.
- After the LaTeX code block, add the changelog described in Section 7, step 5, as plain text outside the code block, clearly separated from the LaTeX itself so it never gets accidentally pasted into Overleaf.
- Flag any `\cite{CITATION_NEEDED_...}` or `[VALUE NOT FOUND IN CODE, please confirm]` placeholders clearly in that same post-code-block summary so the user can resolve them before submission.
- Never deliver the methodology as rendered markdown prose describing the LaTeX; the code block itself is the deliverable.
