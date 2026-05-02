![Open Source Hardware Facts](oshw_facts.svg)

# OpenMCT – DC Motor Control Trainer (Control Theory I/II)

This repository contains the source code and graphical user interface (GUI) for the OpenMCT kit designed for DC motor control. It includes features for real-time data acquisition, excitation signal generation, system identification, and closed-loop testing with PID controllers and controllers in difference equation form.

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


## Contributing

This project is open source. If you wish to contribute:
- Report issues on GitHub.
- Submit pull requests with improvements or fixes.
- Check `THIRD_PARTY_NOTICES.md` for license information.

## License

This project is licensed under the [MIT](LICENSE) license (assuming a LICENSE file exists; otherwise, specify the appropriate license).
