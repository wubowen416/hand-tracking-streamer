
import json

from tcp_client import TCPClient

def make_mh_hand_msg(side, motion_frame, interval_s):
    data = {
        "id": "robot_hand_motion",
        "priority": 0,
        "isRelative": False,
        "isGloablCoordinates": True,
        "motionPartName": f"{side}HandController",
        "thoughSafetyStopoverPoint": False,
        "motionHandData": [
            {
                "id": "robot_hand_motion",
                "motionTowardObject": "BodyController",  # Spine, seems the same as BodyController
                "targetMotionMode": 1,
                "transitionCoordinate": 0,
                "targetPoint": {
                    "x": motion_frame[f"{side}_hand_cartesian_pos"][0],
                    "y": motion_frame[f"{side}_hand_cartesian_pos"][1],
                    "z": motion_frame[f"{side}_hand_cartesian_pos"][2],
                },
                "translateSpeed": -1,
                "translateTime": interval_s * 1000,
                "rotationCoordinate": 9,
                "targetRotation": {
                    "x": motion_frame[f"{side}_hand_euler_angle"][0],
                    "y": motion_frame[f"{side}_hand_euler_angle"][1],
                    "z": motion_frame[f"{side}_hand_euler_angle"][2],
                },
                "rotateSpeed": -1,
                "rotateTime": interval_s * 1000,
                "keepTime": 0,
                "mode": 2,
                "gazeTracking": True,
                "priority": 0,
                "tracking": True,
                "fingerData": [
                    {
                        "motionPartName": "RightFingers",
                        "targetAngle": 0,
                        "springValue": 10,
                    },
                    {
                        "motionPartName": "LeftFingers",
                        "targetAngle": 0,
                        "springValue": 10,
                    },
                ],
            }
        ],
    }
    msg = "playthishandmotion={}".format(json.dumps(data))
    return msg


# motion_frame = {
#     "Right_hand_cartesian_pos": [0.2, 1.1, 0.5],
#     # "Right_hand_cartesian_pos": [0.6, 1.1, 0.0],
#     "Right_hand_euler_angle": [-90.0, 0.0, 0.0],
# }

motion_frame = {
    "Left_hand_cartesian_pos": [-0.2, 1.1, 0.5],
    "Left_hand_euler_angle": [0.0, 0.0, 0.0],
}

client = TCPClient("localhost", 21000)
client.connect()


msg = make_mh_hand_msg("Left", motion_frame, 1.0)
client.send_message(msg)