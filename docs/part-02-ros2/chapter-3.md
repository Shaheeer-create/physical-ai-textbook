---
title: "Chapter 3: ROS 2 Architecture Fundamentals"
description: "Understanding ROS 2 core concepts and architecture"
sidebar_position: 1
---

# Chapter 3: ROS 2 Architecture Fundamentals

## Learning Objectives

After completing this chapter, you should be able to:

- Explain the core concepts of ROS 2 architecture
- Describe the differences between ROS 1 and ROS 2
- Identify key ROS 2 components: nodes, topics, services, and actions
- Set up a basic ROS 2 development environment

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Why ROS 2?

ROS 2 addresses limitations in the original ROS and provides features for real-world applications:

- **Real-time support**: For time-critical applications
- **Multi-robot systems**: Better coordination between multiple robots
- **Security**: Authentication and encryption of communications
- **Deterministic builds**: Reproducible builds across platforms
- **Quality of Service (QoS)**: Configurable reliability and performance settings

## Core Concepts of ROS 2

### Technical Diagrams

import Image from '@theme/IdealImage';

<Image
  img={require('../../static/img/ros2-architecture.svg').default}
  alt="ROS 2 architecture diagram showing nodes communicating through topics, services, and actions, with DDS middleware managing the communication"
  align="center"
/>

> **Figure 1**: ROS 2 Architecture. This diagram shows the core components of a ROS 2 system: nodes communicating through topics (publishers/subscribers), services (clients/servers), and actions (action clients/servers). The DDS middleware manages all communication between nodes, providing Quality of Service controls and enabling multi-robot communication.

### Nodes

Nodes are the fundamental units of computation in ROS. They encapsulate the functionality of the robot and communicate with other nodes through messages.

**Key characteristics of nodes**:
- Each node typically performs a specific task
- Nodes are organized by process for fault isolation
- Multiple nodes can run in the same process if desired
- Nodes can be started and stopped independently

```python
# Example: Creating a minimal ROS 2 node in Python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.get_clock().now().nanoseconds
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are the data packets sent between nodes subscribed/publishing to a topic.

**Topic characteristics**:
- Unidirectional communication (publisher → subscriber)
- Multiple publishers and subscribers can use the same topic
- Anonymous publishing and subscribing
- Asynchronous message passing

### Services

Services provide a request/reply communication pattern. A service client sends a request message and waits for a reply message from a service server.

**Service characteristics**:
- Synchronous communication
- Request-response pattern
- One client to one server at a time
- Blocking call until response is received

### Actions

Actions are used for long-running tasks that may take some time to complete. They allow clients to send a goal to an action server, get feedback during the goal's execution, and receive a result when the goal completes.

**Action characteristics**:
- For long-running goals
- Goal → Feedback → Result pattern
- Goal cancellation
- Goal preemption

## ROS 2 vs ROS 1

### Key Differences

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Communication | ROS Middleware (Custom) | DDS (Data Distribution Service) |
| Build System | rosbuild → catkin | colcon (ament) |
| Quality of Service | No QoS | Configurable QoS |
| Real-time Support | Limited | Full real-time support |
| Security | No security | Authentication & encryption |
| Multi-robot Systems | Challenging | Improved support |
| Licensing | Mixed | Apache 2.0 |

### Migration Considerations

When moving from ROS 1 to ROS 2, consider:
- Rewrite for new build system (ament/colcon)
- Rethink network topology (DDS replaces roscore)
- Implement security configurations
- Review QoS settings for your application

## Setting Up a ROS 2 Development Environment

### Prerequisites

- Ubuntu Linux (20.04 recommended) or Windows 10/11 with WSL2
- Python 3.6 or higher
- Git
- Basic understanding of Linux command line

### Installation Steps

1. **Add the ROS 2 repository key and repository**:
```bash
# Add the repository key
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -

# Add the repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/key] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

2. **Update apt and install ROS 2 packages**:
```bash
sudo apt update
sudo apt install ros-humble-desktop
```

3. **Install colcon build system**:
```bash
sudo apt install python3-colcon-common-extensions
```

4. **Setup environment variables**:
```bash
source /opt/ros/humble/setup.bash
```

### Workspace Setup

Create a ROS 2 workspace for your custom packages:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash
```

## Quality of Service (QoS) Profiles

QoS profiles control the behavior of publishers, subscribers, services, and clients. They define parameters like reliability, durability, and history policies.

### Common QoS Settings

- **Reliability**: Best effort vs Reliable
- **Durability**: Volatile vs Transient local
- **History**: Keep last vs Keep all
- **Lifespan**: Time-based message expiration

```python
# Example: Setting QoS for a publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Create a QoS profile with reliable delivery
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE
)

publisher = self.create_publisher(String, 'topic', qos_profile)
```

## Exercises

1. **Conceptual Question**: Explain the key differences between ROS 1 and ROS 2 architecture, and why these changes were necessary.

2. **Implementation Question**: Create a ROS 2 node that publishes a counter value at 1 Hz and another node that subscribes to this value and logs the received messages.

3. **Analysis Question**: For a safety-critical robot application, which QoS settings would you choose for sensor data publishing, and why?

## Summary

This chapter introduced the fundamental concepts of ROS 2 architecture, including nodes, topics, services, and actions. We explored the key differences between ROS 1 and ROS 2, discussed the new DDS-based communication layer, and covered the setup of a development environment. Understanding these concepts is essential for developing robotic applications with ROS 2.

The key takeaways include:
- ROS 2 is built on DDS for more robust communication
- Quality of Service settings allow fine-tuning of communication behavior
- The new build system offers better reproducibility and cross-platform support
- Security and real-time support are integrated into the architecture

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For sensor systems that ROS 2 can manage, see [Chapter 2: The Robotic Sensorium](../part-01-foundations/chapter-2). For detailed package implementation, see [Chapter 4: Building ROS 2 Packages](../part-02-ros2/chapter-4).