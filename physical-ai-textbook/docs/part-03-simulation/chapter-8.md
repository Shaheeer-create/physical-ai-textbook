---
title: "Chapter 8: Robot Modeling in Gazebo"
description: "Creating and simulating robot models in Gazebo environment"
sidebar_position: 2
---

# Chapter 8: Robot Modeling in Gazebo

## Learning Objectives

After completing this chapter, you should be able to:

- Create URDF models and convert them to SDF for Gazebo
- Define physics properties and constraints for robot models
- Implement sensor simulation in Gazebo
- Configure actuator models for realistic simulation

## Converting URDF to SDF

### Understanding URDF vs. SDF

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) serve different purposes:
- **URDF**: Primarily for kinematic and geometric description of robots
- **SDF**: Comprehensive format for simulation including physics, sensors, and plugins

Gazebo can load URDF models, but internally converts them to SDF for simulation.

### Basic URDF to Gazebo Integration

A robot model designed in URDF needs additional Gazebo-specific tags to function properly in simulation:

```xml
<!-- In your URDF file -->
<robot name="my_robot">
  <!-- Links and joints as usual -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Gazebo-specific tags -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
    <enable_wind>false</enable_wind>
    <gravity>true</gravity>
  </gazebo>

  <!-- Joint definition -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Gazebo properties for the joint -->
  <gazebo reference="base_to_wheel">
    <disableFixedJointLumping>true</disableFixedJointLumping>
  </gazebo>
</robot>
```

### Advanced Gazebo-Specific Tags

#### Visual Properties
```xml
<gazebo reference="link_name">
  <material>
    <ambient>0.1 0.1 0.1 1</ambient>
    <diffuse>0.7 0.7 0.7 1</diffuse>
    <specular>0.01 0.01 0.01 1</specular>
    <emission>0 0 0 1</emission>
  </material>
</gazebo>
```

#### Physics Properties
```xml
<gazebo reference="link_name">
  <mu1>0.2</mu1>  <!-- Primary friction coefficient -->
  <mu2>0.2</mu2>  <!-- Secondary friction coefficient -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>  <!-- Contact damping -->
  <max_vel>100.0</max_vel>  <!-- Maximum contact correction velocity -->
  <min_depth>0.001</min_depth>  <!-- Minimum contact depth -->
  <fdir1>0 0 1</fdir1>  <!-- Friction direction -->
</gazebo>
```

## Physics Properties and Constraints

### Mass and Inertia Considerations

Accurate mass and inertia properties are crucial for realistic simulation:

#### Calculating Inertias
For common geometries:
- **Box** (width x depth x height): `Ixx = 1/12 * m * (d² + h²)`
- **Cylinder** (radius r, height h): `Izz = 1/2 * m * r²`, `Ixx = Iyy = 1/12 * m * (3*r² + h²)`
- **Sphere** (radius r): `Ixx = Iyy = Izz = 2/5 * m * r²`

#### Setting Inertias in URDF
```xml
<inertial>
  <origin xyz="0.012 0 -0.009" rpy="0 0 0"/>
  <mass value="0.142455"/>
  <inertia ixx="0.0000388146" ixy="0" ixz="0.000000952346" 
           iyy="0.0000595356" iyz="-0.000000199477" izz="0.0000663686"/>
</inertial>
```

### Joint Constraints and Dynamics

#### Joint Friction and Damping
```xml
<joint name="example_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  
  <!-- Dynamics properties -->
  <dynamics damping="0.1" friction="0.0"/>
</joint>

<!-- Gazebo-specific joint properties -->
<gazebo reference="example_joint">
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

#### Joint Safety Limits
```xml
<gazebo reference="example_joint">
  <provideFeedback>true</provideFeedback>
  <joint>
    <type>revolute</type>
    <axis>
      <xyz>0 0 1</xyz>
      <limit>
        <lower>-1.57</lower>
        <upper>1.57</upper>
        <effort>10</effort>
        <velocity>1</velocity>
      </limit>
      <dynamics>
        <damping>1.0</damping>
        <friction>0.1</friction>
      </dynamics>
    </axis>
  </joint>
</gazebo>
```

## Sensor Simulation

### Types of Sensors in Gazebo

Gazebo supports simulation of various sensor types:
- **Camera sensors**: RGB, depth, stereo cameras
- **Range sensors**: Laser range finders, sonars
- **IMU (Inertial Measurement Unit)**: Acceleration, angular velocity
- **Force/Torque sensors**: Joint forces and torques
- **GPS sensors**: Position and velocity in world coordinates
- **Contact sensors**: Detect contact between objects

### Camera Sensors

#### RGB Camera
```xml
<gazebo reference="camera_link">
  <sensor name="camera1" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>  <!-- 80 degrees -->
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

#### Depth Camera
```xml
<gazebo reference="depth_camera_link">
  <sensor name="depth_camera" type="depth">
    <update_rate>30</update_rate>
    <camera name="depth">
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.1</stddev>
      </noise>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>camera_ir</cameraName>
      <imageTopicName>/camera/image_raw</imageTopicName>
      <depthImageTopicName>/camera/depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>/camera/depth/points</pointCloudTopicName>
      <cameraInfoTopicName>/camera/camera_info</cameraInfoTopicName>
      <depthImageCameraInfoTopicName>/camera/depth/camera_info</depthImageCameraInfoTopicName>
      <frameName>camera_depth_optical_frame</frameName>
      <pointCloudCutoff>0.5</pointCloudCutoff>
      <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <CxPrime>0.0</CxPrime>
      <Cx>320.5</Cx>
      <Cy>240.5</Cy>
      <focalLength>320.0</focalLength>
      <hackBaseline>0.0718</hackBaseline>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Simulation

```xml
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
          <max_angle>1.570796</max_angle>    <!-- 90 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
      <topicName>/scan</topicName>
      <frameName>laser_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <topic>__default_topic__</topic>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <bodyName>imu_link</bodyName>
      <topicName>imu</topicName>
      <serviceName>imu_service</serviceName>
      <gaussianNoise>0.01</gaussianNoise>
      <updateRateHZ>100.0</updateRateHZ>
    </plugin>
  </sensor>
</gazebo>
```

## Actuator Modeling

### Motor Dynamics

Realistic motor simulation is important for accurate robot behavior:

```xml
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>
```

### Joint Control Types

Gazebo supports various control approaches:

#### Position Control
- Direct control of joint positions
- Good for precise position control
- May cause oscillations if gains are too high

#### Velocity Control
- Control of joint velocities
- Good for smooth motion
- Requires careful tuning of velocity controller

#### Effort Control
- Direct control of joint torques/forces
- Most physically realistic
- Requires sophisticated controllers to be effective

## Practical Example: Complete Robot Model

Here's a complete example of a simple wheeled robot with sensors:

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.035" iyz="0" izz="0.045"/>
    </inertial>
  </link>

  <!-- Gazebo material -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Wheels -->
  <xacro:macro name="wheel" params="suffix parent x y z">
    <joint name="${suffix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${suffix}_wheel"/>
      <origin xyz="${x} ${y} ${z}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${suffix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.02"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 0.8"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.02"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0004"/>
      </inertial>
    </link>

    <gazebo reference="${suffix}_wheel">
      <mu1>100.0</mu1>
      <mu2>100.0</mu2>
      <kp>10000000.0</kp>
      <kd>1.0</kd>
      <fdir1>1 0 0</fdir1>
    </gazebo>
  </xacro:macro>

  <!-- Create two wheels -->
  <xacro:wheel suffix="left" parent="base_link" x="0" y="0.2" z="0"/>
  <xacro:wheel suffix="right" parent="base_link" x="0" y="-0.2" z="0"/>

  <!-- Casters -->
  <joint name="front_caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="front_caster"/>
    <origin xyz="0.15 0 -0.05" rpy="0 0 0"/>
  </joint>

  <link name="front_caster">
    <visual>
      <geometry>
        <sphere radius="0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>

  <gazebo reference="front_caster">
    <mu1>0.01</mu1>
    <mu2>0.01</mu2>
  </gazebo>

  <!-- Sensors -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Gazebo sensor plugin -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Troubleshooting Common Issues

### Model Instability

**Problem**: Robot model wobbles or falls apart in simulation
**Solutions**:
1. Check that all inertial properties are correctly defined
2. Verify that center of mass is properly located
3. Increase physics solver iterations
4. Use smaller time steps
5. Verify that collision geometry matches visual geometry

### Sensor Issues

**Problem**: Sensor data appears incorrect or noisy
**Solutions**:
1. Check sensor configuration in URDF/SDF
2. Verify plugin parameters
3. Adjust noise parameters if too much or too little
4. Ensure proper frame transformations

### Joint Limitations

**Problem**: Joints move beyond intended limits
**Solutions**:
1. Define joint limits in URDF
2. Add Gazebo joint limits
3. Implement software position limit controllers

## Exercises

1. **Model Creation Exercise**: Create a URDF model for a simple differential drive robot with two wheels and a caster, including proper inertial properties.

2. **Sensor Integration Exercise**: Add a camera sensor to your robot model and configure the Gazebo plugin to publish data to a ROS 2 topic.

3. **Analysis Exercise**: Compare the behavior of your robot model with different physics parameters (time steps, solver iterations) and explain how these affect the simulation.

## Summary

This chapter covered the process of creating realistic robot models for simulation in Gazebo. We explored how to convert URDF models to work effectively in Gazebo, defined appropriate physics properties and constraints, implemented various sensor types, and configured actuator models for realistic simulation behavior.

The key takeaways include:
- Proper inertial properties are essential for realistic dynamics
- Gazebo-specific tags enhance URDF models for simulation
- Sensor simulation requires careful configuration for realistic behavior
- Physics parameters significantly impact model stability and performance

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For sensor systems, see [Chapter 2: The Robotic Sensorium](../part-01-foundations/chapter-2). For details on Gazebo simulation environment, see [Chapter 7: Gazebo Simulation Environment](../part-03-simulation/chapter-7).