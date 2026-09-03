# AI Image Generation Prompt — Figure 1: Overall Framework (`fig:overall_framework`)

## Target Journal Style
Top-tier Q1 journal (Elsevier, IEEE TMI, Nature Scientific Reports). Clean, professional, minimalist scientific diagram. No decorative elements. White background. Vector-like flat design with subtle shadows. Sans-serif font (Helvetica or Arial). Thin clean lines. Academic publication quality.

---

## Aspect Ratio and Size
- **Aspect ratio:** 16:9 (landscape, wide)
- **Resolution:** 300 DPI minimum
- **Canvas size:** 2400 × 1350 pixels (or equivalent vector)

---

## Color Palette

| Element | Color | Hex |
|---|---|---|
| Background | White | #FFFFFF |
| Server box | Deep navy blue | #1B3A5C |
| Server box border | Navy blue | #1B3A5C |
| Client boxes (5 clients) | Light steel blue | #4682B4 |
| Client box borders | Steel blue | #4682B4 |
| Data partitioning panel (IID) | Very light blue | #E8F0FE |
| Data partitioning panel (Non-IID) | Very light amber | #FFF8E1 |
| Arrows (broadcast) | Dark gray | #333333 |
| Arrows (aggregate) | Medium gray | #555555 |
| Text (labels) | Dark charcoal | #222222 |
| Text (on dark boxes) | White | #FFFFFF |
| Accent (model name) | Teal | #008080 |
| Accent (FedProx) | Burnt orange | #CC5500 |
| Accent (FocalLoss) | Dark red | #8B0000 |
| Accent (SWA) | Forest green | #228B22 |
| Accent (TTA) | Purple | #6A0DAD |
| Grid lines / dividers | Light gray | #CCCCCC |

---

## Overall Layout (3 Panels, Left to Right)

The figure is divided into **three horizontal panels** separated by thin light-gray vertical dividers:

1. **Left Panel (25% width):** Data Partitioning Scheme
2. **Center Panel (40% width):** Client-Side Local Training Pipeline
3. **Right Panel (35% width):** Server Aggregation and Global Evaluation

A thin horizontal title bar at the top of each panel labels it.

---

## Panel 1 — Left: Data Partitioning Scheme

### Title
"Federated Data Partitioning" (centered, bold, 14pt, dark charcoal)

### Content
Two stacked sub-sections:

**Top sub-section (labeled "Run 1: IID (Uniform)"):**
- Background fill: very light blue (#E8F0FE)
- Show 5 small client icons (simple rounded rectangles) in a horizontal row, each labeled "C1" through "C5" in white text on light steel blue (#4682B4) background
- Above each client, show 4 tiny colored dots representing 4 classes (Chickenpox=red dot, Healthy=green dot, Measles=blue dot, Monkeypox=orange dot), evenly distributed and equal in number across all 5 clients
- A small legend below: 4 colored dots with labels "Chickenpox", "Healthy", "Measles", "Monkeypox" (8pt font)
- Caption text below: "≈300 images/client" (10pt, italic, gray)

**Bottom sub-section (labeled "Run 2: Non-IID (Heterogeneous)"):**
- Background fill: very light amber (#FFF8E1)
- Show 5 client icons (same style) but with different widths proportional to the weight vector [0.10, 0.15, 0.30, 0.25, 0.20]
  - C1: narrowest (10%)
  - C2: slightly wider (15%)
  - C3: widest (30%)
  - C4: medium-wide (25%)
  - C5: medium (20%)
- Each client box shows the same 4 colored dots but in different quantities proportional to the box width
- Caption text below: "w = [0.10, 0.15, 0.30, 0.25, 0.20]" (10pt, monospace, gray)
- Caption text: "7,500 total images/fold" (10pt, italic, gray)

### Arrow
A large right-pointing arrow (dark gray, #333333) from Panel 1 to Panel 2, labeled "Distribute" (10pt, italic) above the arrow.

---

## Panel 2 — Center: Client-Side Local Training Pipeline

### Title
"Client-Side Local Training" (centered, bold, 14pt, dark charcoal)

### Content
Show **one representative client** in detail (labeled "Client k") with the full training pipeline flowing top-to-bottom:

**Step 1 — Input (top):**
- A small image icon (dermatological skin lesion placeholder, simple rounded square with skin-tone gradient)
- Label: "Augmented Image (384×384)" (10pt, dark charcoal)

**Step 2 — Model box (center, largest element):**
- A large rounded rectangle box with deep navy blue (#1B3A5C) border, white fill
- Inside, show a simplified vertical pipeline of the PoxCSAF-Net architecture:
  - "ConvNeXtV2-Tiny Backbone" (12pt, bold, teal #008080)
  - Below it, 4 small stage blocks stacked vertically, each labeled "Stage 0 (ECA)", "Stage 1 (ECA)", "Stage 2 (CBAM)", "Stage 3 (CBAM)" (9pt)
  - Stages 0-1 have a small "ECA" tag in light blue
  - Stages 2-3 have a small "CBAM" tag in light orange
  - Below stages: "CSAH (Cross-Scale Attention Head)" (11pt, bold, teal #008080)
  - Below CSAH: "GeM Pool → Project → Cross-Attn → MLP" (9pt, gray)
- The entire box is labeled at the top: "PoxCSAF-Net" (13pt, bold, teal #008080)

**Step 3 — Loss (below model box):**
- A small box labeled "Focal Loss (γ=2.0)" (11pt, bold, dark red #8B0000)
- Below it: "+ FedProx Proximal Term (μ=0.01)" (10pt, burnt orange #CC5500)
- Mathematical notation: "L = L_focal + (μ/2)‖w_k − w^t‖²" (9pt, italic, gray)

**Step 4 — Optimizer (right side of loss box):**
- Small tag: "AdamW, 3 local epochs" (9pt, gray)
- Small tag: "Grad clip = 1.0" (9pt, gray)

**Step 5 — Output (bottom):**
- Arrow pointing down to a small box: "Updated w_k" (10pt, bold, dark charcoal)

### Side annotations
- On the right side of the model box, a vertical bracket with label: "50 rounds × 3 epochs" (9pt, italic, gray)
- A small green tag on the bottom-right: "SWA (rounds 41–50)" (9pt, forest green #228B22)

### Arrow
A large right-pointing arrow (dark gray, #333333) from Panel 2 to Panel 3, labeled "Transmit w_k" (10pt, italic) above the arrow.

---

## Panel 3 — Right: Server Aggregation and Global Evaluation

### Title
"Server Aggregation & Evaluation" (centered, bold, 14pt, dark charcoal)

### Content

**Top — Server box:**
- A large rounded rectangle labeled "Central Server" (13pt, bold, white text on deep navy blue #1B3A5C fill)
- Inside: "FedAvg Aggregation" (11pt, bold, white)
- Mathematical formula: "w^{t+1} = Σ (n_k/n) · w_k" (10pt, white, monospace)

**Middle — 5 client return arrows:**
- 5 thin arrows converging from the left into the server box, each labeled "w_1", "w_2", "w_3", "w_4", "w_5" (8pt, gray)
- Arrows are medium gray (#555555)

**Below server box — Broadcast arrow:**
- A single arrow pointing left (back toward Panel 2) labeled "Broadcast w^t" (9pt, italic, dark gray #333333)
- This arrow is dashed to distinguish from the solid aggregation arrows

**Bottom — Evaluation pipeline:**
- Below the server box, a downward arrow to a box labeled "Global Model w^{t+1}" (11pt, bold, dark charcoal, white fill, navy border)
- Below that, a downward arrow to a box labeled "Aggregated Validation Set" (10pt, dark charcoal, light gray fill)
- Below that, a downward arrow to a box labeled "Macro-F1 Evaluation" (10pt, bold, dark charcoal)
- To the right of the evaluation box, a small decision diamond (light gray) with:
  - "Yes" branch (green arrow): "Save Best w*" (9pt, forest green #228B22)
  - "No" branch (red arrow): "Early Stop (P=18)" (9pt, dark red #8B0000)

**Bottom-right corner — Inference pipeline:**
- A small vertical chain of boxes:
  - "Best w* + SWA" (9pt, bold)
  - Down arrow
  - "Ensemble (softmax avg)" (9pt)
  - Down arrow
  - "Temperature Scaling (T*)" (9pt, purple #6A0DAD)
  - Down arrow
  - "TTA (10-crop)" (9pt, purple #6A0DAD)
  - Down arrow
  - "Final Prediction" (9pt, bold, dark charcoal)

---

## Global Annotations

### Top-left corner
"5-Fold Cross-Validation" (10pt, bold, dark charcoal, with a small 5-fold icon: 5 pie slices)

### Top-right corner
"K = 5 Clients" (10pt, bold, dark charcoal)

### Bottom-center (spanning all 3 panels)
A thin horizontal timeline bar showing "Round t = 1, 2, ..., 50" with tick marks, and a bracket over rounds 41–50 labeled "SWA" in forest green (#228B22)

---

## Font Specifications
- **All labels:** Helvetica or Arial
- **Panel titles:** 14pt, bold
- **Box labels:** 11–13pt, bold
- **Sub-labels:** 9–10pt, regular
- **Mathematical notation:** LaTeX-style italic, 9–10pt
- **Legends:** 8pt, regular

---

## Design Rules
1. **No 3D effects.** Flat design only.
2. **No gradients on boxes** (solid fills only), except the skin lesion image placeholder which may have a subtle skin-tone gradient.
3. **All boxes have rounded corners** (radius ~8px).
4. **All arrows have clean arrowheads**, not decorative.
5. **Consistent stroke weight** (1.5px for boxes, 1px for arrows, 0.5px for dividers).
6. **No clipart or icons** except the small colored dots for class representation and the skin lesion placeholder.
7. **White background** throughout.
8. **No shadows** except very subtle ones on the main boxes (offset 2px, blur 4px, opacity 10%).
9. **Text must be crisp and readable** at print size.
10. **The diagram should read left-to-right** as a natural data flow: partition → train → aggregate.

---

## Summary Description for AI Model

Create a wide landscape (16:9) scientific diagram for a Q1 journal paper on federated learning for monkeypox classification. White background, flat design, sans-serif font (Helvetica/Arial), thin clean lines, professional academic style. The figure has three panels separated by thin light-gray vertical dividers:

LEFT PANEL (25%): "Federated Data Partitioning" — Two stacked sub-sections. Top: "Run 1: IID (Uniform)" with light blue background, showing 5 equal-width client boxes (C1–C5) in steel blue, each with 4 equal colored dots (red=Chickenpox, green=Healthy, blue=Measles, orange=Monkeypox). Bottom: "Run 2: Non-IID (Heterogeneous)" with light amber background, showing 5 client boxes of varying widths (10%, 15%, 30%, 25%, 20%) with proportional dot counts. Label "w = [0.10, 0.15, 0.30, 0.25, 0.20]".

CENTER PANEL (40%): "Client-Side Local Training" — One representative client pipeline flowing top-to-bottom: skin lesion image (384×384) → large model box labeled "PoxCSAF-Net" (teal) containing 4 stacked stage blocks (Stage 0 ECA, Stage 1 ECA, Stage 2 CBAM, Stage 3 CBAM) and "CSAH (Cross-Scale Attention Head)" → loss box "Focal Loss (γ=2.0)" (dark red) + "FedProx (μ=0.01)" (burnt orange) → "Updated w_k". Side annotations: "50 rounds × 3 epochs", "AdamW", "SWA (rounds 41–50)" (green).

RIGHT PANEL (35%): "Server Aggregation & Evaluation" — Navy blue server box "Central Server / FedAvg: w^{t+1} = Σ(n_k/n)·w_k" with 5 converging arrows (w_1–w_5) from left and a dashed broadcast arrow going left. Below: evaluation chain: "Global Model" → "Aggregated Validation" → "Macro-F1" → decision diamond (Yes: "Save Best w*" green, No: "Early Stop P=18" red). Bottom-right: inference chain "Best w* + SWA → Ensemble → Temp Scaling → TTA → Final Prediction".

Top-left: "5-Fold Cross-Validation". Top-right: "K = 5 Clients". Bottom: timeline bar "Round 1...50" with SWA bracket over rounds 41–50. Colors: navy #1B3A5C, steel blue #4682B4, teal #008080, burnt orange #CC5500, dark red #8B0000, forest green #228B22, purple #6A0DAD. No 3D, no gradients, no clipart, rounded corners, 1.5px box strokes, 1px arrows.
