![Open Source Hardware Facts](oshw_facts.svg)

# OpenMCT – DC Motor Control Trainer

This repository contains the source code and graphical user interface (GUI) for the OpenMCT kit designed for DC motor control. It includes features for real-time data acquisition, excitation signal generation, system identification, and closed-loop testing with PID controllers and controllers in difference equation form.

This started as a personal project, but with the advent of AI coding agents, the project grew exponentially, both in functionality and complexity. I have thoroughly tested all the implementations, but consider that some bugs might appear or that the code can be improved.

## Features

- **Real-time data acquisition**: Encoder reading for position and velocity measurement.
- **Excitation control**: PWM output generation for the motor driver.
- **Current measurement**: Compatible with the DRV8874 driver included in the kit.
- **Data streaming**: Serial transmission for visualization and logging.
- **Implemented controllers**: PID and z-domain coefficient controllers.
- **Intuitive graphical interface**: Python/Qt GUI for configuration, monitoring, and parameter tuning.

## Repository Files

### Firmware (Teensy / PlatformIO)
- **`src/main.cpp`**  
  Firmware code for the Teensy 4.x microcontroller. Implements:
  - Encoder reading for position and velocity measurement.
  - PWM output to the motor driver.
  - Current measurement (only compatible with the kit including DRV8874).
  - Real-time data streaming via serial port.
  - Control algorithm execution (PID and z-domain coefficients).

- **`platformio.ini`**  
  Configuration file for the PlatformIO project (VS Code extension).

> **Important note**: The GUI can be adapted to other controllers, but requires pin modifications in `main.cpp`. Otherwise, it will not work correctly.

### GUI (Python/Qt)
- **`GUI.py`**  
  Desktop application developed in Python with PyQt6 for:
  - Serial port connection to the device.
  - Real-time signal plotting.
  - Customizable excitation signal generation.
  - Data logging and export.
  - PID parameter tuning and controller coefficient adjustment.

- **`QtDesignerGUI.ui`**  
  Interface design file created with Qt Designer.

### Plotting mode

The `modo_plot` selector defaults to FOH, joining adjacent time samples with
straight lines. ZOH holds each value until the next sample and draws the jump at
that next sample, like MATLAB `stairs`. It applies to live input/output/PWM,
identification and simulation curves, and time responses in the main tabs and
enlarged dialogs. Existing curves change immediately without rerunning the model.
Root-locus branches and frequency-response curves keep their original rendering.

This selector changes visualization only. Logged samples, controller coefficients,
simulation results, and the ZOH input convention used during identification are
independent of the selected plotting mode.

### Transfer-function identification

Identification returns a continuous-time transfer function fitted to the logged
PWM input and measured output. The fitting simulation holds each PWM value until
the next sample (ZOH), uses ordinary least squares, and estimates the record's
initial conditions. Initial states are eliminated by a linear least-squares solve
for each candidate model; they are specific to that record and are not exported
as part of the transfer function. Stable multistart optimization is retained.

When timing jitter requires a uniform time grid, the PWM input is resampled with
the previous held value; only the measured output is linearly interpolated. The
identification result displays only the transfer-function fraction and the MATLAB-style fit:
`100 * (1 - norm(y - yfit) / norm(y - mean(y)))`. The existing RMSE, MAE, R2 and
NRMSE fields are computed from the same fitted response.

Regression tests compare the six recorded 5 ms / 20 ms models against the exact
coefficients and fit percentages saved in the MATLAB identification session.
Equivalent numerical fits can have small coefficient differences because the
optimizers and stopping criteria differ. See `test/README` for how to run them.

### Digital controller deployment and PWM comparison

The digital controller implements

```text
C(z) = (A + B z^-1 + C z^-2 + D z^-3 + I z^-4) /
       (1 - E z^-1 - F z^-2 - G z^-3 - H z^-4)
```

`A..D` and `I` are the numerator coefficients in increasing sample-delay order;
`E..H` are the negatives of the normalized denominator coefficients `a1..a4`.
Controller memory uses float32 values after saturation (`0..255` in speed,
`-255..255` in position). Rounding occurs only at the PWM actuator. Speed adds
the configured dead-zone compensation before rounding/clamping and resets the
controller at zero reference; position uses signed control with an unsigned PWM
magnitude and a separate direction signal, without dead-zone compensation.

On Teensy, `PWMbits` in `src/main.cpp` defaults to 12 and remains configurable
before compiling. UNO R4 keeps its 8-bit default. `PWMOUT` contains native integer
counts: after compensation and saturation in normalized 0..255 units, firmware
computes `round(level * ((1 << PWMbits) - 1) / 255)`. Controller history is never
replaced by these counts. The applied PWM reported over serial is converted back
to normalized units, preserving decimals in both live buffers and CSV files.
At 12 bits, the hardware range is 0..4095 and one count is approximately 0.06227
normalized PWM units. Identification input, gains, dead-zone compensation and GUI
limits therefore continue to use the 0..255 scale. Requested resolution remains
subject to the PWM timer's available resolution at the configured frequency.

Export from the enlarged discrete root-locus window preserves all five numerator
coefficients, uses nine significant digits for float32 round trips, and sets the
firmware sampling interval to the design period. Noncausal controllers, order
above four, coefficients outside float32 range, and periods that cannot be
expressed in whole milliseconds are reported instead of being truncated.
The runtime interval can still have hardware timing jitter.

The firmware accepts the existing 25- and 29-field CSV commands. The 30-field
format appends `I` after `RPM_filter`; all earlier fields retain their positions.
Legacy commands explicitly clear `I`. The GUI sends 29 fields when `I=0`; for
nonzero `I`, it first requires the response `#CAPS DC_MOTOR B4` to `CAPS?`.
Update the firmware before deploying a nonzero fifth numerator coefficient.
GUI-only mode can preview/export both formats without hardware.
The capability reply also includes `#PWM_BITS n`; the GUI uses it to initialize
the simulation's **PWM resolution (bits)** field. Offline simulation defaults to
12 bits, and connected legacy firmware defaults to 8 until a resolution is
reported. Changing that field changes the simulation, not the firmware setting.

In the enlarged discrete root-locus window, **Compare with integer PWM** overlays
the ideal linear response and a sample-by-sample simulation with float32 state,
mode saturation, and PWM rounding inside the feedback loop. Set the reference in
RPM or degrees and select the operating mode. The reference defaults to a unit
step (1), and PWM comparison is unchecked by default. The additional curve shows
the plant output (RPM or position). The simulation uses the selected
controller `K*C(z)` and the discrete plant shown in the root-locus tab.

Choose **Applied PWM (GUI identification)** for plants identified by this GUI,
because identification uses the measured/applied PWM as its input. Choose
**Effective PWM after dead zone** only for a plant explicitly expressed in those
units: the speed plant then receives `max(PWM - compensation, 0)`. For example,
at 8 bits an internal output of `0.5` with compensation `10` produces PWM `11`,
so these two plant conventions receive `11` and `1`, respectively. At 12 bits,
the same command produces 169 native counts, about 10.52381 normalized units;
the plant conventions then receive about 10.52381 and 0.52381. Position receives
signed PWM in either convention. The effective-input option assumes the motor
dead zone equals the configured compensation; it does not identify motor friction.

The PWM simulation requires a strictly proper plant and a supported controller;
otherwise it reports the reason while retaining the ideal curve. Encoder
quantization, RPM averaging, timing jitter, and physical motor nonlinearities are
not included. Physical validation remains necessary before treating either curve
as a measured hardware response.

### Other Files
- **`include/` and `lib/`**  
  Auxiliary libraries and headers for the firmware.
- **`test/`**  
  Test files for the project.
- **`THIRD_PARTY_NOTICES.md`**  
  Information about third-party licenses.

---

## Requirements

### Firmware
- **Development environment**: VS Code with PlatformIO extension installed.
- **Hardware**: Teensy 4.x (or the target configured in `platformio.ini`).
- **Connections**: Motor driver and encoder connected according to the code specifications.

### GUI
- **Python**: Version 3.10 or higher recommended.
- **Dependencies**: The following Python libraries are required:
  - PyQt6
  - pyqtgraph
  - pyserial
  - numpy
  - pandas
  - control
  - scipy
  - matplotlib

## Installation

### Firmware
1. Install VS Code and the PlatformIO extension.
2. Open the project in VS Code.
3. Connect the Teensy 4.x to your computer.
4. Compile and upload the firmware using PlatformIO: `pio run --target upload`.

Teensy 4.0 remains the default environment. Both configured targets can be built
without uploading with `pio run -e teensy40 -e uno_r4_wifi`. Board pin selection
follows the selected environment automatically; UNO R4 WiFi uses the H-bridge
pin mapping in `src/main.cpp`, while Teensy uses the kit's DIR/!SLEEP mapping.

### GUI
1. Ensure Python 3.10+ is installed.
2. Install the dependencies by running the following command in PowerShell or terminal:
   ```
   pip install PyQt6 pyqtgraph pyserial numpy pandas control scipy matplotlib
   ```
3. Run the GUI with:
   ```
   python GUI.py
   ```

## Usage

1. **Hardware setup**: Connect the DC motor, driver, and encoder.
2. **Upload firmware**: Flash the code to the Teensy using PlatformIO.
3. **Run the GUI**: Launch the application and connect to the Teensy's serial port.
4. **Configure parameters**: Adjust PID parameters or input z-domain coefficients.
5. **Run tests**: Generate excitation signals, monitor responses, and log data.

The enlarged continuous root-locus window can export the selected **K*C(s)**
using **Close and export to continuous controller tab**. The **Continuous Contr**
page also accepts numerator and denominator coefficients in descending powers
of s. Use the existing **Start** and **Update parameters** buttons. Conversion
uses standard Tustin and the main **Sampling time (ms)** setting; changing that
period recomputes the deployed coefficients on the next update. The separate
Discretization page's prewarp setting does not affect this controller.

Continuous compensators must be proper and of order at most four. All five
numerator coefficients are preserved; firmware with B4 support is required when
the fifth discrete numerator coefficient is nonzero. The existing general
difference equation executes the converted controller, so no firmware change is
needed. PID parameters and manually entered discrete coefficients remain separate.
The arbitrary-controller path uses saturated difference-equation memory, as in
the discrete-controller page; the PID page retains its own anti-windup algorithms.

Both controller pages provide **Back to root locus design** after a design is
exported. The button restores the source plant, compensator, gain and, for a
discrete design, its sample period and PWM comparison settings. Returning sends
no motor command; changes take effect when the controller is exported again.
These source designs are saved with the GUI configuration and restored on restart.

The PID Tuner uses a continuous second-order response target. **Response speed**
reduces the target response time from 15 to 0.5 plant time constants;
**Transient behavior** reduces target overshoot from 35% to 0%. This is a response
objective, not a guaranteed upper bound. P/PI/PD/PID structure and the configured
derivative-filter time can limit how closely the actual response matches it.
The detailed view reports the actual step-response metrics for comparison.

`pid_tuning.py` fits the filtered parallel controller to that reference with
nonnegative normalized gains and a strictly convex regularized objective. P/PD
also account for their steady-state tracking error. Each request is independent
of previous slider positions, plants and views. Unstable results and unsupported
plants produce an error; previous gains are not substituted. The enlarged view
passes its settings directly to the tuner without changing the main sliders.
These are ideal continuous-time designs; hardware still deploys the existing
discretized PID with its saturation and anti-windup behavior.

## Editing the interface in Qt Designer

The static interface is defined in Designer forms:

- `QtDesignerGUI.ui`: the main window and all thirteen tabs, including both controller pages.
- `ui/RootLocusDialog.ui`: continuous root-locus designer.
- `ui/DiscreteRootLocusDialog.ui`: discrete root-locus designer and PWM comparison controls.
- `ui/PIDTunerDialog.ui`: enlarged PID tuner.
- `ui/PortSelectDialog.ui`: serial-port selection.

Open these files directly in Qt Designer. Preserve the `objectName` of controls
used by `GUI.py` when moving or styling them. Changes to labels, sizes, layouts
and button placement are loaded when the application starts; no Python UI code
needs to be generated.

Graphs and their standard Matplotlib toolbars are promoted widgets (`PlotCanvas`,
`PlotToolbar` from `ui_widgets`, and `PlotWidget` from `pyqtgraph`). Designer shows
their editable placeholders; the application renders the actual plots at runtime.
Keep the promotion definitions in the forms. `GUI.py` handles signals, state,
runtime enable/disable logic and plot configuration rather than constructing
static controls.

The layout keeps the live motor plots wide, limits numeric fields to useful
widths, and treats the PID/root-locus plots as previews that open their detailed
windows on double click. Diagram labels use `AspectRatioLabel` from `ui_widgets`
to retain their original proportions. The original `Implemented_H(z).png` is
preserved; `Implemented_H_complete.svg` includes the fifth numerator coefficient
used by the firmware.

Continuous-controller fields are promoted to `CoefficientEdit` from
`coefficient_edit`: unfocused fields show two decimal places, exact zeros show
their gray power placeholders, and focusing exposes the full value for editing.
Calculations, serial export and saved settings retain the original precision.
Page order can be changed in Designer; firmware IDs follow the page object names.
Root-locus plant fields use the same editor, with two decimal places for the
continuous plant and five for the discrete plant. Their detailed root-locus
windows open maximized. The continuous design page has root-locus and step
previews; the block-reduction page shows its series and feedback results in two
columns. The START/STOP label is independent of the motor's running state.

The form before the proportion adjustments, together with its Python files,
is preserved in `backups/ui_20261105/`. Keep layouts on edited groups so resizing
the application continues to arrange their controls.

## Contributing

This project is open source. If you wish to contribute:
- Report issues on GitHub.
- Submit pull requests with improvements or fixes.
- Check `THIRD_PARTY_NOTICES.md` for license information.

## License

CC BY 4.0
You can share, copy and modify this dataset so long as you give appropriate credit, provide a link to the CC BY license, and indicate if changes were made, but you may not do so in a way that suggests the rights holder has endorsed you or your use of the dataset. Note that further permission may be required for any content within the dataset that is identified as belonging to a third party.
