---
title: "Chapter 4: Building ROS 2 Packages"
description: "Creating and structuring ROS 2 packages with Python"
sidebar_position: 2
---

# Chapter 4: Building ROS 2 Packages

## Learning Objectives

After completing this chapter, you should be able to:

- Create ROS 2 packages using Python (rclpy)
- Understand the proper package structure and organization
- Implement launch files and parameter management
- Apply best practices for error handling and debugging in ROS 2

## Package Structure

A ROS 2 package follows a standardized structure that enables consistent build and execution across different systems and platforms.

### Technical Diagrams

<img
  src="/img/ros2-package-structure.svg"
  alt="Diagram showing the file structure of a ROS 2 package with directories for source code, launch files, configuration, and tests"
  width={600}
  height={400}
/>

> **Figure 1**: ROS 2 Package Structure. This diagram illustrates the standard organization of a ROS 2 package, showing the relationship between the manifest file, source code directories, launch files, configuration files, and test directories. The hierarchical structure enables modular and maintainable code organization.

### Basic Package Layout

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml            # Package manifest
├── setup.cfg              # Python installation configuration
├── setup.py               # Python package setup
├── resource/my_package    # Package resource files
├── my_package/            # Main Python package directory
│   ├── __init__.py
│   ├── my_node.py         # Example node implementation
│   └── my_service.py      # Example service implementation
├── launch/                # Launch files
│   └── my_launch.py
├── config/                # Configuration files
│   └── params.yaml
├── test/                  # Test files
│   └── test_my_node.py
└── README.md              # Package documentation
```

### Package Manifest (package.xml)

The `package.xml` file contains metadata about your package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package for my robot</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Creating a ROS 2 Package with Python

### Using ros2 pkg Command

The easiest way to create a new package is using the `ros2 pkg create` command:

```bash
# Create a new package named 'my_robot_controller' using Python
ros2 pkg create --build-type ament_python my_robot_controller

# Create a new package with dependencies
ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs geometry_msgs my_robot_controller
```

### Python Package Structure

Inside the main package directory, you'll find the Python module structure:

```python
# my_robot_controller/my_robot_controller/__init__.py
# Empty file to make this directory a Python package

# my_robot_controller/my_robot_controller/robot_controller.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create publisher
        self.publisher = self.create_publisher(String, 'controller_status', 10)
        
        # Create subscriber
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        
        # Timer for periodic execution
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        self.get_logger().info('Robot controller node initialized')

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f'Received velocity command: linear={msg.linear.x}, angular={msg.angular.z}')

    def timer_callback(self):
        msg = String()
        msg.data = f'Controller status: OK at {self.get_clock().now().nanoseconds}'
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    
    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files

Launch files allow you to start multiple nodes with specific configurations simultaneously.

### Python-based Launch Files

```python
# my_robot_package/launch/controller_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directory
    package_dir = get_package_share_directory('my_robot_package')
    
    # Create nodes
    robot_controller = Node(
        package='my_robot_package',
        executable='robot_controller',  # This should match your entry point
        name='robot_controller',
        parameters=[
            os.path.join(package_dir, 'config', 'params.yaml')
        ],
        output='screen'
    )
    
    sensor_processor = Node(
        package='my_robot_package',
        executable='sensor_processor',
        name='sensor_processor',
        output='screen'
    )
    
    return LaunchDescription([
        robot_controller,
        sensor_processor
    ])
```

### Entry Points Configuration

To make your nodes executable, you need to configure entry points in `setup.py`:

```python
# setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Example ROS 2 package for my robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = my_robot_package.robot_controller:main',
            'sensor_processor = my_robot_package.sensor_processor:main',
        ],
    },
)
```

## Parameter Management

Parameters allow you to configure node behavior without changing code. ROS 2 provides multiple ways to manage parameters.

### YAML Parameter Files

```yaml
# config/params.yaml
my_robot_controller:
  ros__parameters:
    update_rate: 50.0
    max_velocity: 1.0
    safety_distance: 0.5
    topics:
      cmd_vel_topic: "/cmd_vel"
      odom_topic: "/odom"
    debug_mode: false
```

### Using Parameters in Code

```python
# my_robot_controller/my_robot_controller/robot_controller.py
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Declare parameters with default values
        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)
        
        # Get parameter values
        self.update_rate = self.get_parameter('update_rate').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        
        # Create publisher
        self.publisher = self.create_publisher(String, 'controller_status', 10)
        
        # Create timer based on parameter
        self.timer = self.create_timer(1.0/self.update_rate, self.timer_callback)
        
        self.get_logger().info(f'Controller initialized with update rate: {self.update_rate}Hz')
        
        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
    
    def parameter_callback(self, params):
        """Callback for parameter changes"""
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.max_velocity = param.value
                self.get_logger().info(f'Max velocity updated to: {self.max_velocity}')
        
        return SetParametersResult(successful=True)
```

## Error Handling and Debugging

### Best Practices for Error Handling

1. **Handle exceptions gracefully**:
```python
def process_sensor_data(self, sensor_msg):
    try:
        # Process sensor data
        processed_data = self.transform_sensor_data(sensor_msg)
        self.publish_processed_data(processed_data)
    except ValueError as e:
        self.get_logger().error(f'Invalid sensor data: {e}')
    except Exception as e:
        self.get_logger().error(f'Unexpected error in sensor processing: {e}')
```

2. **Use appropriate logging levels**:
```python
# For important information
self.get_logger().info('Node started successfully')

# For warnings
self.get_logger().warn('Sensor calibration may be outdated')

# For errors that don't stop execution
self.get_logger().error('Failed to connect to sensor')

# For detailed debugging
self.get_logger().debug('Processing sensor reading: ' + str(reading))
```

3. **Implement node lifecycle management**:
```python
def destroy_node(self):
    """Clean up resources before node destruction"""
    # Stop timers
    if self.timer:
        self.timer.cancel()
    
    # Shutdown publishers/subscribers
    # Clean up other resources
    
    super().destroy_node()
```

### Debugging Tools

#### ROS 2 Command Line Tools

```bash
# Check the status of all nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>

# Echo messages from a topic
ros2 topic echo /topic_name std_msgs/msg/String

# List all available topics
ros2 topic list

# Call a service
ros2 service call /service_name std_srvs/srv/Empty

# Get information about services
ros2 service list
```

#### Launch File Debugging

```bash
# Launch with debug output
ros2 launch my_robot_package controller_launch.py --debug

# Launch specific nodes for testing
ros2 run my_robot_package robot_controller
```

## Best Practices for Package Development

### Code Structure

1. **Follow ROS 2 Python style guide**:
   - Use snake_case for function and variable names
   - Use PascalCase for class names
   - Add docstrings for all public methods

2. **Keep nodes focused**:
   - Each node should have a single primary responsibility
   - Break complex functionality into multiple nodes
   - Use composition rather than inheritance when possible

3. **Organize code modularly**:
```python
# my_robot_controller/
#   __init__.py
#   robot_controller.py      # Main node
#   sensor_processor.py      # Sensor processing node
#   kinematics.py           # Forward/inverse kinematics
#   trajectory_planner.py   # Path planning algorithms
#   utils.py                # Helper functions
```

### Testing

Include unit tests for your components:

```python
# test/test_robot_controller.py
import unittest
import rclpy
from my_robot_controller.robot_controller import RobotController

class TestRobotController(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = RobotController()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_parameter_initialization(self):
        """Test that parameters are properly initialized"""
        self.assertEqual(self.node.update_rate, 50.0)
        self.assertEqual(self.node.max_velocity, 1.0)

if __name__ == '__main__':
    unittest.main()
```

## Exercises

1. **Implementation Exercise**: Create a ROS 2 package that implements a simple robot controller node which subscribes to velocity commands and publishes robot status messages. Include a launch file and parameter configuration.

2. **Analysis Exercise**: Review the provided parameter callback function. Explain how it could be enhanced to validate parameter bounds and trigger specific actions when parameters change.

3. **Troubleshooting Exercise**: You have a ROS 2 node that's not receiving messages from a topic you expect. List the debugging steps you would take to identify and resolve the issue.

## Summary

This chapter covered the essentials of creating and structuring ROS 2 packages with Python. We explored the proper package structure, learned how to create launch files for starting multiple nodes, implemented parameter management, and discussed best practices for error handling and debugging. These skills are fundamental for developing robust and maintainable ROS 2 applications.

The key takeaways include:
- Proper package structure is essential for maintainability and reuse
- Launch files simplify the process of starting complex multi-node systems
- Parameter management allows runtime configuration of node behavior
- Effective error handling and debugging practices are crucial for robot systems
- Following ROS 2 best practices leads to more reliable and maintainable code

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For sensor systems that ROS 2 can integrate, see [Chapter 2: The Robotic Sensorium](../part-01-foundations/chapter-2). For architectural concepts of ROS 2, see [Chapter 3: ROS 2 Architecture Fundamentals](../part-02-ros2/chapter-3).