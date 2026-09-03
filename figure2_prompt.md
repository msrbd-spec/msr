# AI Image Generation Prompt — Figure 2: Model Architecture (`fig:model_arch`)

## Model Name
**PoxCSAF-Net** (Pox Cross-Scale Attention Fusion Network)

---

## Target Journal Style
Top-tier Q1 journal (Elsevier, IEEE TMI, Nature Scientific Reports). Clean, professional, minimalist scientific architecture diagram. No decorative elements. White background. Vector-like flat design with subtle shadows. Sans-serif font (Helvetica or Arial). Thin clean lines. Academic publication quality. The diagram should look like it was drawn by a professional medical imaging researcher, not a graphic designer.

---

## Aspect Ratio and Size
- **Aspect ratio:** 3:4 (portrait, tall) — the architecture flows top-to-bottom
- **Resolution:** 300 DPI minimum
- **Canvas size:** 1800 × 2400 pixels (or equivalent vector)

---

## Color Palette

| Element | Color | Hex |
|---|---|---|
| Background | White | #FFFFFF |
| Input image box | Light skin-tone gradient | #FFE0BD → #D4A373 |
| Backbone stages (Stage 0–1, shallow) | Light blue | #ADD8E6 |
| Backbone stages (Stage 2–3, deep) | Medium blue | #6495ED |
| ECA module tag | Cyan | #00CED1 |
| CBAM module tag | Orange | #FF8C00 |
| ECA/CBAM attention box fill | Very light yellow | #FFFACD |
| Gated residual arrow (α) | Gray | #888888 |
| Stochastic depth (DropPath) | Dashed gray | #999999 |
| CSAH main box border | Teal | #008080 |
| CSAH main box fill | Very light teal | #E0F8F8 |
| GeM pooling box | Light green | #90EE90 |
| Projection box | Light purple | #DDA0DD |
| Token stack (T) | Medium purple | #9370DB |
| Cross-attention box | Gold | #FFD700 |
| Q/K/V labels | Dark gold | #B8860B |
| Residual connection arrow | Teal | #008080 |
| LayerNorm box | Light gray | #D3D3D3 |
| Temperature (τ) tag | Red | #DC143C |
| MLP head box | Light coral | #F08080 |
| Dropout tag | Light pink | #FFB6C1 |
| Final output (4 classes) | Dark charcoal | #222222 |
| Arrows (data flow) | Dark gray | #333333 |
| Text (labels) | Dark charcoal | #222222 |
| Text (on colored boxes) | White or dark | #FFFFFF / #222222 |
| Dividers / grid | Light gray | #CCCCCC |

---

## Overall Layout (Top-to-Bottom Flow)

The diagram flows **vertically from top to bottom** in 6 major sections:

1. **Input Image** (top)
2. **ConvNeXtV2-Tiny Backbone** (4 stages with attention)
3. **Feature Extraction Arrows** (Stages 1, 2, 3 → CSAH)
4. **CSAH (Cross-Scale Attention Head)** (the novel contribution, largest section)
5. **MLP Classification Head**
6. **Output (4 classes)** (bottom)

Each section is clearly separated with whitespace. A thin vertical centerline guides the eye downward.

---

## Section 1 — Input Image (Top)

### Content
- A rounded rectangle box (200×200px) with a skin-tone gradient fill (#FFE0BD → #D4A373)
- Inside: a simple abstract skin lesion icon (a small irregular circle shape in slightly darker skin tone)
- Label below: "Input Image" (12pt, bold, dark charcoal)
- Label below that: "384 × 384 × 3" (10pt, monospace, gray)
- Down arrow (dark gray, #333333) to Section 2

---

## Section 2 — ConvNeXtV2-Tiny Backbone (4 Stages)

### Title (left side, vertical)
"ConvNeXtV2-Tiny Backbone" (12pt, bold, medium blue #6495ED, rotated 90° or placed as a left-side bracket label)

### Content
Four stacked stage blocks, each a horizontal rounded rectangle. Stages get progressively narrower (to show spatial resolution decrease) but the same height. Each stage block contains:

**Stage 0:**
- Fill: light blue (#ADD8E6)
- Label inside: "Stage 0" (11pt, bold, white)
- Right side tag: "96 ch, 96×96" (9pt, gray)
- Below the stage block, a small attention module:
  - Small rounded box labeled "ECA" (10pt, bold, cyan #00CED1, fill very light yellow #FFFACD)
  - A small "α=0.01" tag (8pt, gray) next to a curved arrow showing gated residual
- Down arrow to Stage 1

**Stage 1:**
- Fill: light blue (#ADD8E6)
- Label inside: "Stage 1" (11pt, bold, white)
- Right side tag: "192 ch, 48×48" (9pt, gray)
- Below: "ECA" module (same style as Stage 0)
- Down arrow to Stage 2

**Stage 2:**
- Fill: medium blue (#6495ED)
- Label inside: "Stage 2" (11pt, bold, white)
- Right side tag: "384 ch, 24×24" (9pt, gray)
- Below: "CBAM" module (10pt, bold, orange #FF8C00, fill very light yellow #FFFACD)
  - Small sub-label: "Channel + Spatial Attn" (8pt, gray)
  - "DropPath p=0.10" tag (8pt, dashed gray #999999)
  - "α=0.01" gated residual tag (8pt, gray)
- Down arrow to Stage 3

**Stage 3:**
- Fill: medium blue (#6495ED)
- Label inside: "Stage 3" (11pt, bold, white)
- Right side tag: "768 ch, 12×12" (9pt, gray)
- Below: "CBAM" module (same style as Stage 2)
- Down arrow to Section 3

### Side annotation (right side of all 4 stages)
A vertical bracket spanning all 4 stages with label: "LayerNorm + GRN" (9pt, italic, gray) — indicating the normalization used throughout

---

## Section 3 — Feature Extraction Arrows (Stages → CSAH)

### Content
Three arrows branching out from Stages 1, 2, and 3 to the right, converging into the CSAH box below:

- **From Stage 1:** A curved arrow going right and down, labeled "S₁" (10pt, bold, dark charcoal) with tag "192 ch" (8pt, gray)
- **From Stage 2:** A curved arrow going right and down, labeled "S₂" (10pt, bold, dark charcoal) with tag "384 ch" (8pt, gray)
- **From Stage 3:** A straight arrow going down, labeled "S₃" (10pt, bold, dark charcoal) with tag "768 ch" (8pt, gray)

All three arrows converge into the top of the CSAH box.

### Label
"Multi-Scale Feature Maps" (10pt, italic, gray) centered above the CSAH box

---

## Section 4 — CSAH (Cross-Scale Attention Head) — Largest Section

### Main container
- Large rounded rectangle with teal (#008080) border (2px), very light teal (#E0F8F8) fill
- Title at top: "CSAH (Cross-Scale Attention Head)" (13pt, bold, teal #008080)

### Internal flow (top-to-bottom inside CSAH):

**Step 4a — GeM Pooling (3 parallel paths):**
- Three small boxes side by side, each labeled:
  - "GeM Pool" (10pt, bold, light green #90EE90 fill)
  - Below each: "p=3.0" (8pt, gray)
- Left box: tag "S₁ → p₁" (9pt, dark charcoal)
- Center box: tag "S₂ → p₂" (9pt, dark charcoal)
- Right box: tag "S₃ → p₃" (9pt, dark charcoal)
- Each box has dimensions noted: "→ 192-d", "→ 384-d", "→ 768-d" (8pt, gray)
- Down arrows from each

**Step 4b — Linear Projection (3 parallel paths):**
- Three small boxes side by side, each labeled:
  - "Proj + LN" (10pt, bold, light purple #DDA0DD fill)
  - Below each: "→ 256-d" (8pt, gray)
- Left: "t₁" (9pt, bold, dark charcoal)
- Center: "t₂" (9pt, bold, dark charcoal)
- Right: "t₃" (9pt, bold, dark charcoal)
- Down arrows from each

**Step 4c — Token Stack:**
- A single wide box labeled "T = [t₁; t₂; t₃]" (11pt, bold, medium purple #9370DB fill, white text)
- Tag: "B × 3 × 256" (8pt, gray)
- Down arrow

**Step 4d — Cross-Scale Attention (centerpiece):**
- A large box labeled "Cross-Attention" (12pt, bold, gold #FFD700 fill, dark text)
- Inside, show the attention mechanism visually:
  - Left side: "Q = t₃" (10pt, bold, dark gold #B8860B) with a small arrow
  - Center: "K, V = T" (10pt, bold, dark gold #B8860B) with a small arrow
  - Right side: formula "softmax(QKᵀ/√d)·V" (9pt, monospace, dark charcoal)
  - A small visual: 3 small circles (keys) with arrows from Q to each, showing attention weights
- Tag below: "d = 256" (8pt, gray)
- Output arrow labeled "f" (10pt, bold, dark charcoal)
- Down arrow

**Step 4e — Residual + LayerNorm + Temperature:**
- A box showing three operations stacked:
  - "f + t₃" (10pt, bold, teal #008080) with a curved residual arrow from t₃ (right side, going around the cross-attention box)
  - "LayerNorm" (10pt, bold, light gray #D3D3D3 fill)
  - "× τ" (10pt, bold, red #DC143C) with tag "τ=1.0 (learnable)" (8pt, gray)
- Output labeled "f'" (10pt, bold, dark charcoal)
- Down arrow

### Residual connection
- A curved arrow on the right side of the CSAH, going from t₃ (Step 4b) all the way down to the residual addition (Step 4e), bypassing the cross-attention. This arrow is teal (#008080), labeled "residual" (8pt, italic)

---

## Section 5 — MLP Classification Head

### Content
- A rounded rectangle box with light coral (#F08080) fill
- Title: "MLP Head" (11pt, bold, dark charcoal)
- Inside, show the flow:
  - "Dropout(0.35)" (9pt, light pink #FFB6C1 tag)
  - Down arrow
  - "Linear: 256 → 128" (9pt, monospace, gray)
  - Down arrow
  - "GELU" (9pt, bold, dark charcoal)
  - Down arrow
  - "Dropout(0.15)" (9pt, light pink #FFB6C1 tag)
  - Down arrow
  - "Linear: 128 → 4" (9pt, monospace, gray)
- Down arrow to Section 6

---

## Section 6 — Output (Bottom)

### Content
- A horizontal row of 4 small boxes, each a different color:
  - "Chickenpox" (10pt, bold, white on red #E74C3C fill)
  - "Healthy" (10pt, bold, white on green #27AE60 fill)
  - "Measles" (10pt, bold, white on blue #3498DB fill)
  - "Monkeypox" (10pt, bold, white on orange #E67E22 fill)
- Above the 4 boxes: "ŷ ∈ ℝ⁴" (10pt, italic, dark charcoal)
- Below: "4-Class Classification" (10pt, bold, dark charcoal)

---

## Global Annotations

### Top-right corner
"PoxCSAF-Net" (14pt, bold, teal #008080) — the model name

### Left side (vertical bracket spanning Stages 0–3)
"Pretrained: FCMAE + ImageNet-22k + ImageNet-1k" (9pt, italic, gray)

### Right side (near CSAH)
"~28.5M parameters" (9pt, italic, gray)

### Bottom-right
"Output: 4-class softmax" (9pt, italic, gray)

---

## Font Specifications
- **Model name:** 14pt, bold, teal
- **Section titles:** 12–13pt, bold
- **Box labels:** 10–11pt, bold
- **Sub-labels / tags:** 8–9pt, regular
- **Mathematical notation:** monospace or italic, 9–10pt
- **Class labels:** 10pt, bold, white

---

## Design Rules
1. **No 3D effects.** Flat design only.
2. **No gradients on boxes** (solid fills only), except the input image which has a skin-tone gradient.
3. **All boxes have rounded corners** (radius ~6px).
4. **All arrows have clean arrowheads**, consistent style.
5. **Consistent stroke weight** (1.5px for main boxes, 1px for arrows, 0.5px for tags).
6. **No clipart** except the abstract skin lesion icon in the input.
7. **White background** throughout.
8. **Subtle shadows** on main boxes only (offset 2px, blur 4px, opacity 8%).
9. **Text must be crisp and readable** at print size.
10. **The diagram flows strictly top-to-bottom** — no left-right zigzagging except for the 3 parallel paths inside CSAH.
11. **The 3 parallel paths** (GeM → Proj → tokens) should be visually clear as parallel columns that merge into the token stack.
12. **The residual connection** (t₃ bypassing cross-attention) should be a curved arrow on the right side, clearly visible and labeled.
13. **Attention modules (ECA, CBAM)** should be visually distinct from the backbone stages — smaller boxes attached below each stage, not inside the stage block.

---

## Summary Description for AI Model

Create a tall portrait (3:4) scientific architecture diagram for a Q1 journal paper. Model name: "PoxCSAF-Net" (teal, top-right). White background, flat design, Helvetica/Arial font, thin clean lines, professional academic style. The diagram flows strictly top-to-bottom in 6 sections:

SECTION 1 (TOP): Input image box with skin-tone gradient, labeled "Input Image 384×384×3". Down arrow.

SECTION 2: "ConvNeXtV2-Tiny Backbone" — 4 stacked horizontal stage blocks getting progressively narrower. Stage 0 (light blue, "96 ch, 96×96") and Stage 1 (light blue, "192 ch, 48×48") each have a small "ECA" attention module below them (cyan tag, light yellow fill, "α=0.01" gated residual). Stage 2 (medium blue, "384 ch, 24×24") and Stage 3 (medium blue, "768 ch, 12×12") each have a "CBAM" attention module below them (orange tag, "Channel + Spatial Attn", "DropPath p=0.10", "α=0.01"). Left bracket: "Pretrained: FCMAE + IN22k + IN1k". Right bracket: "LayerNorm + GRN".

SECTION 3: Three curved arrows from Stages 1, 2, 3 labeled "S₁ (192ch)", "S₂ (384ch)", "S₃ (768ch)" converging downward. Label: "Multi-Scale Feature Maps".

SECTION 4 (LARGEST): "CSAH (Cross-Scale Attention Head)" — large box with teal border, light teal fill. Inside, top-to-bottom: (4a) Three parallel "GeM Pool" boxes (light green, "p=3.0") for S₁/S₂/S₃, each showing input dims → output dims. (4b) Three parallel "Proj + LN" boxes (light purple, "→ 256-d") producing tokens t₁/t₂/t₃. (4c) Single "T = [t₁; t₂; t₃]" box (medium purple, "B×3×256"). (4d) "Cross-Attention" box (gold) showing "Q = t₃", "K,V = T", formula "softmax(QKᵀ/√d)·V", "d=256". (4e) Residual addition "f + t₃" (teal), "LayerNorm" (gray), "× τ" (red, "τ=1.0 learnable"). Curved teal residual arrow from t₃ bypassing cross-attention to the residual addition. Output "f'".

SECTION 5: "MLP Head" (light coral box): "Dropout(0.35) → Linear 256→128 → GELU → Dropout(0.15) → Linear 128→4".

SECTION 6 (BOTTOM): 4 class boxes in a row: "Chickenpox" (red), "Healthy" (green), "Measles" (blue), "Monkeypox" (orange). Label "ŷ ∈ ℝ⁴".

Colors: light blue #ADD8E6, medium blue #6495ED, cyan #00CED1, orange #FF8C00, teal #008080, light green #90EE90, light purple #DDA0DD, medium purple #9370DB, gold #FFD700, red #DC143C, light coral #F08080. No 3D, no gradients (except input), rounded corners, 1.5px box strokes, 1px arrows. "~28.5M parameters" (bottom-right).
