import cv2

from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())

def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.get_position())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())

def move_until_obstacle(rob: IRobobo):
    try:
        while True:
            # Start moving forward for a short duration to continually check sensors
            rob.move(25, 25, 500)
            #rob.move(75, 75, 1000)

            # Read IR sensor data
            ir_data = rob.read_irs()
            print("IR sensor data:", ir_data)

            # Check if any of the IR sensors detect an obstacle
            #if any(sensor > 150 for sensor in ir_data):
            #if ir_data[4] > 5:
            if ir_data[4] > 150:

                # Stop the robot
                rob.move(0, 0, 0)
                rob.talk("Obstacle detected")
                break

            rob.sleep(0.1)

        # Start moving backwards until another obstacle is detected
        while True:
            rob.move(-25, -25, 500)
            #rob.move(-75, -75, 1000)

            # Read IR sensor data
            ir_data = rob.read_irs()
            print("IR sensor data while moving backwards:", ir_data)

            # Check if any of the IR sensors detect an obstacle
            #if any(sensor > 150 for sensor in ir_data):
            #if ir_data[6] > 10:
            if ir_data[6] > 150:

                # Stop the robot
                rob.move(0, 0, 0)
                rob.talk("Another obstacle detected")
                break

            rob.sleep(0.1)

        # Turn left
        rob.move(0, 75, 800)
        #rob.move(0, 10, 3800)
        rob.talk("Turning left")

    except KeyboardInterrupt:
        rob.move(0, 0, 0)

def move(rob: IRobobo):
    rob.move(10, 10, 10000)


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    #test_emotions(rob)
    #test_sensors(rob)
    #test_move_and_wheel_reset(rob)
    move_until_obstacle(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)

    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)

    #test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()