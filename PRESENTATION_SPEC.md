# AI Guardian — Presentation & Accessibility Rebuild (next session)

The dashboard is a spec doc that got prettified. The next pass turns it into a
high-trust narrative that *demonstrates* accessibility rigor rather than describing
a product that has it. Nothing below changes model behaviour or the safety budget.

---

## 1. Information architecture

Reorder the page into a trust arc. Current order leads with stats before the reader
has a problem in mind, and drops the simulation mid-scroll without setup.

| # | Section | Job |
|---|---------|-----|
| 1 | Hero | One emotional hook + one mechanism sentence. No stats. |
| 2 | The problem | 3-second gut punch: why reactive description apps fail a pedestrian. |
| 3 | The mechanism | fast path / slow path / veto — strongest IP, lead visually. |
| 4 | Proof | The live simulation, framed "try it yourself", not "here's data". |
| 5 | Principles / rules | Move **up** from the footer — builds trust fast. |
| 6 | Roadmap v1/v2/v3 | Keep, simplify labels. |
| 7 | CTA | The page currently has no next action. Add exactly one. |

Stats (`2.0 s lead time`, `<2/km false alarms`) move to section 3 or later.

## 2. Copy

- Promote **"Nothing is allowed to slow the safety loop"** to the hero headline.
  It is the sharpest line on the page and is currently buried in meta description.
- Rename engineer-speak features to benefit-first:
  - Way back → **Find your way home**
  - Object memory → **Remembers what you touched last**
  - Scene on demand → **Ask what's around you**
- Kill unexplained shorthand (`spec·camera imu gps`). Replace with plain status
  badges (**Shipped** / **In development**) plus sensor *icons*, not sensor words.
- Match the rules-section tone everywhere. Cryptic UI labels ("Change the weather",
  "Where we are") become sentences a visitor parses in one pass.

## 3. Colour system (target spec)

| Role | Value | Note |
|------|-------|------|
| Base | `#0B1220` deep blue-black | precision instrument, not toy |
| Hazard accent | `#FF7A1A` amber | **reserved for real hazard states only** — never decorative |
| Safe state | desaturated green-teal | bright green reads "app"; muted reads "instrument panel" |
| Text | `#F5F6F7` off-white | never pure white on near-black — halation |

No rainbow sensor coding. One accent; icon *shape* carries the differentiation.

## 4. Aesthetic direction — HUD, not marketing site

- Monospace tabular numerals for every stat, timer and percentage.
- 1px hairline borders replace drop shadows. Shadows read consumer; hairlines read instrument.
- The simulation panel becomes the centrepiece: more vertical room, persistent
  recording indicator, and animated gauges instead of static percentages.

## 5. Icons & SVG

- Remove all emoji glyphs (📷 🎙 ◎). Ship a single-weight line set, 1.5 px stroke,
  consistent corner radius, for camera / mic / GPS / IMU.
- One signature looping SVG for **sense → predict → veto → guide**, running
  continuously in the hero so the 10 Hz loop is understood before a word is read.
- Confidence / urgency become arc gauges (speedometer), not percentage text.

## 6. Accessibility — non-negotiable for this product

- Verify real contrast ratios against WCAG AA (4.5:1 body) with a checker on the
  shipped hex values, not by eye.
- Every icon-only control carries an `aria-label`.
- **The simulation must be fully keyboard-operable and screen-reader narratable.**
  A blind visitor who cannot experience the demo is the single most damaging
  inconsistency this site could ship.
- Add a visible WCAG AA conformance badge — a trust signal specific to this audience.

## Build order

1. Accessibility pass on what exists (contrast, aria-labels, keyboard sim).
2. IA reorder + hero rewrite.
3. Colour token migration + hairline/shadow swap.
4. Icon set + hero loop SVG.
5. Arc gauges + simulation centrepiece layout.
6. CTA.
