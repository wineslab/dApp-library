# How to Save the I/Q Samples During a Data Collection

This guide explains how to configure and run the Spectrum Sharing dApp to record
SigMF-compliant I/Q samples with ground truth labels during a data collection session.
Labels identify the RF scenario being captured (e.g. `no_rfi`, `jammer`, `radar`) and are
embedded directly into SigMF annotations at the sample index where `save_samples` is called —
making label boundaries precise to the I/Q frame, not to a flush interval.

I/Q recording is only active when `--save-iqs` is passed. Without it, `--ground-truth` and
`--show-controls` have no effect.

---

## Flag combinations and their behavior

| `--save-iqs` | `--demo-gui` | `--ground-truth <label>` | `--show-controls` | Result |
|:---:|:---:|:---:|:---:|---|
| ✗ | any | any | any | No I/Q recording, no labeling, GUI shows no label selector |
| ✓ | ✗ | ✗ | any | I/Q saved, annotations written with empty label |
| ✓ | ✗ | ✓ | any | I/Q saved, all annotations carry `<label>` for the entire run |
| ✓ | ✓ | ✗ | ✗ | GUI opens, label selector hidden (no callback wired) |
| ✓ | ✓ | ✓ | ✗ | GUI opens, label selector shown with `<label>` pre-selected as default — operator can switch live |
| ✓ | ✓ | ✓ | ✓ | Same as above plus PRB/threshold controls panel shown |

> **Note on `--ground-truth` in interactive mode:** when used together with `--demo-gui`,
> `--ground-truth <label>` sets the *default* value pre-selected in the GUI tile selector.
> The operator can switch to a different label at any time during the run; the new label
> takes effect immediately on the next `save_samples` call. The initial value is not locked.

---

## Interactive vs CI/CD

| | Interactive | CI/CD |
|---|---|---|
| **Label source** | Dashboard tile selector (GUI), with `--ground-truth` as pre-selected default | `--ground-truth <label>` fixed at launch, never changes |
| **Label changes** | Operator switches label mid-run to match the signal being transmitted | One label per run — scenario must be known in advance and kept constant |
| **When the label is logged** | On every `save_samples` call while that label is active — switching the tile writes the new label into all subsequent annotations | Same mechanism, but the label never changes so every annotation in the file carries the same label |
| **Dataset implication** | A single recording can contain multiple labeled segments (e.g. `no_rfi` → `jammer` → `no_rfi`) | Each recording is a single-class file; multiple scenarios require separate runs |
| **Recommended workflow** | Use `--demo-gui --show-controls` and coordinate with the person controlling the RF source | Parameterize `--ground-truth` and `--timed` in the pipeline job matrix — one job per scenario label |

---

## Commands

> [!NOTE]
> `--ota` sets the noise floor to 20 dB, a value calibrated for the X310 USRP. That default
> is **ignored** whenever `--noise-floor-threshold` is explicitly provided — the explicit value
> always takes precedence regardless of `--ota`.

The commands below use 53 dB as the noise floor threshold, which matches both the Colosseum
testbed and the Foxconn RU calibration. Center frequency is 3.75 GHz with 106 PRBs.

> [!IMPORTANT]
> `--ground-truth` must be set to the label of the **RFI signal being injected** (e.g.
> `jammer`, `radar`, `lte_aggressor`). Do not use `no_rfi` here: when no PRBs are detected
> above the threshold the code automatically writes `no_rfi` into the annotation itself —
> the label you provide is only stamped when interference is actually detected.

### Static threshold

The detector flags any PRB whose power exceeds a fixed absolute threshold.

**Interactive:**
```bash
hatch run python examples/spectrum_dapp.py \
  --link zmq --transport ipc \
  --save-iqs \
  --control \
  --num-prbs 106 \
  --center-freq 3.75e9 \
  --noise-floor-threshold 53 \
  --ground-truth <rfi_label> \
  --demo-gui \
  --show-controls
```

**CI/CD:**
```bash
hatch run python examples/spectrum_dapp.py \
  --link zmq --transport ipc \
  --save-iqs \
  --control \
  --num-prbs 106 \
  --center-freq 3.75e9 \
  --noise-floor-threshold 53 \
  --ground-truth <rfi_label> \
  --timed 60
```

### Adaptive noise floor

The detector estimates the noise floor dynamically and flags PRBs exceeding
`noise_floor + SNR_threshold`. Detected PRBs are held in embargo for 5 seconds
after the last detection before being released.

**Interactive:**
```bash
hatch run python examples/spectrum_dapp.py \
  --link zmq --transport ipc \
  --save-iqs \
  --control \
  --num-prbs 106 \
  --center-freq 3.75e9 \
  --use-adaptive-noise-floor \
  --noise-floor-threshold 53 \
  --embargo-timeout-secs 5 \
  --ground-truth <rfi_label> \
  --demo-gui \
  --show-controls
```

**CI/CD:**
```bash
hatch run python examples/spectrum_dapp.py \
  --link zmq --transport ipc \
  --save-iqs \
  --control \
  --num-prbs 106 \
  --center-freq 3.75e9 \
  --use-adaptive-noise-floor \
  --noise-floor-threshold 53 \
  --embargo-timeout-secs 5 \
  --ground-truth <rfi_label> \
  --timed 60
```
