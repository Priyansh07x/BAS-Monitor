---
name: Obsidian Control
colors:
  surface: '#111318'
  surface-dim: '#111318'
  surface-bright: '#37393e'
  surface-container-lowest: '#0c0e12'
  surface-container-low: '#1a1c20'
  surface-container: '#1e2024'
  surface-container-high: '#282a2e'
  surface-container-highest: '#333539'
  on-surface: '#e2e2e8'
  on-surface-variant: '#bac9cc'
  inverse-surface: '#e2e2e8'
  inverse-on-surface: '#2f3035'
  outline: '#849396'
  outline-variant: '#3b494c'
  surface-tint: '#00daf3'
  primary: '#c3f5ff'
  on-primary: '#00363d'
  primary-container: '#00e5ff'
  on-primary-container: '#00626e'
  inverse-primary: '#006875'
  secondary: '#ffd799'
  on-secondary: '#432c00'
  secondary-container: '#feb300'
  on-secondary-container: '#6a4800'
  tertiary: '#b1ffbf'
  on-tertiary: '#003918'
  tertiary-container: '#22ef7e'
  on-tertiary-container: '#006731'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#9cf0ff'
  primary-fixed-dim: '#00daf3'
  on-primary-fixed: '#001f24'
  on-primary-fixed-variant: '#004f58'
  secondary-fixed: '#ffdeac'
  secondary-fixed-dim: '#ffba38'
  on-secondary-fixed: '#281900'
  on-secondary-fixed-variant: '#604100'
  tertiary-fixed: '#62ff96'
  tertiary-fixed-dim: '#00e475'
  on-tertiary-fixed: '#00210b'
  on-tertiary-fixed-variant: '#005226'
  background: '#111318'
  on-background: '#e2e2e8'
  surface-variant: '#333539'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  data-lg:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
    letterSpacing: -0.01em
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  caption:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin-page: 24px
  panel-padding: 20px
  stack-sm: 8px
  stack-md: 16px
---

## Brand & Style

The design system is engineered for high-stakes experiment monitoring and mission control environments. It targets technical operators, scientists, and engineers who require immediate clarity and low cognitive load during long-duration shifts. 

The visual style is a fusion of **Modern Corporate** precision and **Glassmorphism**. It prioritizes a "Lab Equipment" aesthetic: modular, utilitarian, and sophisticated. Every interface element exists to serve data density and situational awareness. The UI should evoke a sense of calm authority, precision, and futuristic reliability.

## Colors

The palette is rooted in a deep charcoal and navy foundation to minimize eye strain in low-light control rooms. 

- **Primary (Neon Cyan):** Reserved for active states, primary actions, and "running" telemetry.
- **Secondary (Amber):** Specifically for warnings, cautionary data, or pending states.
- **Tertiary (Emerald Green):** Indicates success, completed cycles, or stable connectivity.
- **Neutral/Background:** A range of deep obsidian tones (`#0A0C10`) and dark navy-greys (`#161B22`) create the canvas.
- **Data Visualization:** Use the primary, secondary, and tertiary colors as the lead status indicators against the dark neutral backdrop.

## Typography

Typography is treated as a functional readout. **Inter** provides high legibility for prose, instructions, and navigation. **JetBrains Mono** is utilized for all numerical data, status labels, and timestamps to ensure that characters are distinct and columns of data align vertically for easy scanning.

On mobile devices, `display-lg` should scale down to `32px` to maintain hierarchy without breaking container bounds. Always prioritize contrast ratios for small-scale mono labels to ensure they remain readable against dark backgrounds.

## Layout & Spacing

This design system employs a **Fixed Grid** philosophy within a dashboard shell. The screen is divided into functional "Zones" (Sidebar, Global Header, Main Monitoring Area, and Inspector Panel).

- **Grid:** A 12-column system for the main content area.
- **Rhythm:** An 8px base unit drives all spacing, with a 4px sub-grid for tight data-rich components.
- **Responsive Behavior:** On desktop, panels are anchored. On tablet, the Inspector Panel collapses into a drawer. On mobile, the layout stacks vertically, prioritizing the most critical "Primary Metric" cards at the top.

## Elevation & Depth

Depth is achieved through **Tonal Layering** and **Glassmorphism**, rather than traditional shadows.

1.  **Level 0 (Base):** Deepest navy-black. Static background.
2.  **Level 1 (Panels):** Slightly lighter charcoal with a 1px "Ghost Border" (`#30363D`).
3.  **Level 2 (Overlays/Modals):** Semi-transparent surfaces with a `20px` backdrop blur and a more prominent inner glow or highlight on the top edge to simulate physical thickness.

Avoid heavy drop shadows. Use a subtle `40%` opacity cyan or amber outer glow exclusively for critical alerts to simulate an emitting light source on the hardware.

## Shapes

The shape language is "Soft-Industrial." Components use a `4px` (0.25rem) radius for a precise, machined look. 

- **Containers:** Standard panels use `rounded-md` (4px).
- **Interactive Elements:** Buttons and inputs follow the same 4px rule to maintain a consistent silhouette.
- **Indicator Lights:** Status pips (LEDs) are the only fully circular elements, mimicking physical hardware lights.

## Components

- **Buttons:** Low-profile with 1px borders. Primary buttons use a solid Cyan fill with black text. Secondary buttons are transparent with a Cyan border. 
- **Data Cards:** These should feature a "Header" section with a `label-caps` title and a main "Value" section using `data-lg`.
- **Status LEDs:** Small circular pips that use the palette's Cyan, Amber, and Emerald colors. Use a subtle CSS pulse animation for "Active" or "Warning" states.
- **Input Fields:** Darker than the panel background, using a 1px border that turns Cyan on focus. Monospaced text for numerical inputs.
- **Progress Bars/Gauges:** Minimalist linear tracks. The "fill" should be the status color, while the "track" is a low-opacity version of the same color.
- **Terminal/Log:** A dedicated component using JetBrains Mono, featuring a slightly different background tone to distinguish it as a raw data feed.