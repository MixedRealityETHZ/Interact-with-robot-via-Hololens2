# Interacting with the Robot via Hololens 2
This repo contains the source code for Interacting with the Robot via Hololens 2 project. We implemented several intuitive and immersive methods for robot control, including joystick, hand-gestures and Azure Spatial Anchors.

## Setup
### Environment preparation
We build our app on Unity 2020.3.40f1, Visual Studio 2022.
### Clone source code
```
git clone https://github.com/MixedRealityETHZ/Interact-with-robot-via-Hololens2.git
```
### Import toolkits
Open the [MixedRealityFeatureTool](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/unity/welcome-to-mr-feature-tool) and select the folder that you just cloned. Choose "Restore Features".
### Import project
Open Unity Hub, Select the drop down box near the "Open" and choose "Add project from disk".

After opening the unity, drag the three scenes from Scenes folder and delete the original empty scene.

Select "File"->"Build Settings"->"Universal Windows Platform"->"Switch Platform"->"Build", and this will generate a sln file.

Open the sln file with visual studio 2022, choose build mode as "Release", target as "Device". Connect Hololens to the computer with cable and wait for building.

## Detailed functions
### Joystick
The joystick is a cube and the Spot robot moves by interacting with it. For example if we rotate the cube it sends angular velocity commands to Spot and it rotates, while if you move the cube up and down, the robot moves forward and backward respectively by receiving linear velocity commands.
[Watch the video](https://drive.google.com/file/d/10NkD0m7RoxhTW9849UQwYLM7zUaB9oYj/view?usp=sharing)

### Hand-gestures
This is a system for detecting and recognizing hand gestures to control a Spot robot. We use the MRTK (Mixed Reality Toolkit) to detect hands and obtain hand joint data, which we then pass to a hand gesture recognition model. We convert the model output into velocity commands, which we send to the Spot computer.
[Watch the video](https://drive.google.com/file/d/19ZoH12Rgbq7-o9xvbA3--_vqvGkIGc7G/view?usp=sharing)

Hand gesture recognition training code and `README` are in the `MR_model_training` folder.

### Azure Spatial Anchors
Azure Spatial Anchor is an online service to map 3D spaces, we can use it to colocalize Spot with Hololens and know their relative position with each other. When a user places a spatial anchor (represented by a sphere) anywhere in the room, the robots moves to that location. The orientation of the Spot robot is such that it always faces the user.
[Watch the video](https://drive.google.com/file/d/1tp7by5bSnSeOmuDXBCE8HjAfZAqygAAl/view?usp=sharing)

### Using spot

```rosrun image_transport republish raw in:=/spot/camera/hand_color/image compressed out:=/spot/camera/hand_color```

Capture from HoloLens Portal video for higher quality

```rosservice call /asa_ros/create_anchor '{anchor_in_target_frame: {header: {frame_id: odom}}}'```



Run Spot driver: ```roslaunch spot_driver driver.launch```

Run TCP Endpoint: ```roslaunch ros_tcp_endpoint endpoint.launch```

Run ASA: ```roslaunch asa_ros asa_ros.launch```

Run go2anchor.py: ```python3 src/go2anchor/go2anchor.py```


Create a dummy anchor in spot: rosservice call /asa_ros/create_anchor '{anchor_in_target_frame: {header: {frame_id: odom}}}’

Run image to joystick scene: ```rosrun image_transport republish raw in:=/spot/camera/hand_color/image compressed out:=/spot/camera/hand_color```


