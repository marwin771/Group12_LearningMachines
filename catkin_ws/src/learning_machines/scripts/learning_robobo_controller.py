#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, run_task0_actions, run_task1_actions


if __name__ == "__main__":
    # You can do better argument parsing than this!
    print(sys.argv)
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
    
    print(sys.argv[2])

    if len(sys.argv) >= 3 and sys.argv[2] == "--train":
        run_all_actions()
    elif len(sys.argv) >= 3 and sys.argv[2] == "--test":
        if len(sys.argv) >= 4:
            run_task1_actions(rob, sys.argv[3])
        else:
            run_task1_actions(rob)

    # run_all_actions(rob)
