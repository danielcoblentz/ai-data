# Opportunistic Robot Controller

This project implements a multi-layered robot controller that navigates a simulated environment. The robot uses an opportunistic strategy to visit locations, always selecting the nearest unvisited location from its current position. The project also includes a "trap" scenario to test the controller's behavior in challenging environments.

## Features
- **Opportunistic Navigation**: The robot selects the closest unvisited location to optimize its path.
- **Multi-Layered Architecture**:
  - **Top Layer**: High-level decision-making and planning.
  - **Middle Layer**: Handles movement commands and retries.
  - **Body Layer**: Interacts with the simulated environment.
- **Trap Scenario**: A challenging environment to test the robot's ability to navigate obstacles.

## File Structure
- `agentTop.py`: Implements the top layer of the robot controller.
- `agentMiddle.py`: Implements the middle layer of the robot controller.
- `agentEnv.py`: Defines the robot's body and the simulated environment.
- `display.py`: Provides utilities for displaying debug information.

Reorganized layout (added on 2025-09-30):

- `src/` - Python package containing modules and utilities. Reusable modules were moved to `src/utils/`.
- `notebooks/` - Jupyter notebooks (moved from repository root).
- `data/` - datasets used by the notebooks and scripts (unchanged).
- `models/`, `scripts/`, `docs/`, `tests/` - reserved for future organization.

Compatibility shims were added at the repository root (e.g., `utilities.py`, `display.py`, `variable.py`) that re-export implementations from `src.utils` so existing import paths continue to work.

## How to Run
1. Ensure Python 3 is installed on your system.
2. Run the main script:
   ```bash
   python agentTop.py
   ```
3. Example usage:
   - Run the default example:
     ```python
     rob_ex()
     ```
   - Use opportunistic behavior:
     ```python
     rob_ex(use_opportunistic=True)
     ```
   - Test the trap scenario:
     ```python
     robot_trap()
     ```

## Example Output
The robot will navigate the environment and print its progress:
```
Goal o109 True
Goal storage True
Goal o103 True
```

## Dependencies
- Python 3.x
- No external libraries are required.

## Notes
- The `robot_trap` function demonstrates a scenario where the robot may struggle to escape due to obstacles.
- Modify the `World` object to create custom environments for testing.
