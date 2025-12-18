---
title: "Glossary of Terms"
description: "Definitions of technical terms used in Physical AI and Humanoid Robotics"
sidebar_position: 100
---

# Glossary of Terms

This glossary provides definitions for technical terms used throughout the Physical AI & Humanoid Robotics textbook.

## A

**Actuator**: A component of a robot that converts energy into physical movement. Common types include electric motors, hydraulic cylinders, and pneumatic systems.

**AI (Artificial Intelligence)**: The simulation of human intelligence processes by machines, especially computer systems. In robotics, AI techniques enable perception, decision-making, and learning.

## B

**Behavior Tree**: A hierarchical structure used in robotics and AI to organize and execute complex behaviors. Behavior trees provide a structured alternative to finite state machines.

## C

**Computer Vision**: A field of AI that enables computers to interpret and understand visual information from the world. In robotics, computer vision allows robots to recognize objects, navigate, and interact with their environment.

## D

**DDS (Data Distribution Service)**: A middleware protocol and API standard for real-time, scalable, dependable, performant, and interoperable data exchanges. It's the foundation for ROS 2 communication.

**Deep Learning**: A subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.

## E

**Embodied Intelligence**: Intelligence that is grounded in the physical world through interaction with the environment. Unlike abstract AI, embodied intelligence must deal with real-world physics and sensorimotor contingencies.

## F

**Forward Kinematics**: The use of joint parameters to compute the position and orientation of the robot's end-effector. It's fundamental for robot control and motion planning.

**Force/Torque Sensor**: A device that measures the forces and torques applied to a robot at particular points, often used for precise manipulation and interaction with objects.

## G

**Gazebo**: A 3D simulation environment for robotics that provides realistic physics simulation and rendering capabilities for testing robot algorithms in a virtual environment.

## H

**Humanoid Robot**: A robot with a physical form that generally resembles that of a human, typically having a head, torso, two arms, and two legs.

## I

**IMU (Inertial Measurement Unit)**: A device that measures and reports a body's specific force, angular rate, and sometimes the magnetic field surrounding the body, using a combination of accelerometers, gyroscopes, and magnetometers.

**Inverse Kinematics**: The mathematical process of calculating the joint parameters needed to position a robot's end-effector at a desired location and orientation.

## K

**Kinematics**: The study of motion without considering the forces that cause the motion. In robotics, it deals with the relationship between joint angles and the position of the robot's end-effector.

## L

**LiDAR (Light Detection and Ranging)**: A remote sensing method that uses light in the form of a pulsed laser to measure distances to objects. It's commonly used in robotics for mapping and navigation.

**Locomotion**: The ability to move from one place to another. In humanoid robotics, this typically refers to walking bipedally.

## M

**Machine Learning**: A type of AI that enables systems to automatically learn and improve from experience without being explicitly programmed.

**Middleware**: Software that provides common services and capabilities to applications beyond what's offered by the operating system. In ROS 2, the DDS implementation serves as the middleware.

## N

**Node**: In ROS (Robot Operating System), a process that performs computation. Nodes are the fundamental building blocks of a ROS program.

## P

**Proprioception**: The sense of the relative position of one's own parts of the body and strength of effort being employed in movement. In robotics, this refers to sensors that measure the robot's internal state.

**Path Planning**: The computational problem of finding a valid sequence of configurations to a robot to move from a start to a goal position while avoiding obstacles.

## Q

**QoS (Quality of Service)**: In ROS 2, a set of policies that define the behavior of publishers and subscribers, including reliability, durability, and history of messages.

## R

**Real-time**: A property of a system where the correctness of the system depends not only on the logical correctness of the computation but also on the time at which the results are produced.

**ROS (Robot Operating System)**: A flexible framework for writing robot software that provides a collection of tools, libraries, and conventions for creating robot applications.

**ROS 2**: The second generation of the Robot Operating System, designed to be suitable for production environments with improved security, real-time support, and multi-robot systems.

## S

**Sensor Fusion**: The process of combining sensory data or data derived from disparate sources such that the resulting information has less uncertainty than would be possible when these sources were used individually.

**SLAM (Simultaneous Localization and Mapping)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

## T

**Topic**: In ROS, a named channel over which nodes exchange messages. Topics enable asynchronous message passing between nodes.

## V

**Vision System**: A system that uses cameras and computer vision algorithms to perceive and interpret visual information from the environment.

## W

**Waypoint**: A set of coordinates that identify a point in physical space, typically used for navigation purposes to define a path for a robot to follow.

## X, Y, Z

**X, Y, Z Axes**: The three-dimensional coordinate system used to define positions and orientations in physical space. X typically represents left/right, Y represents forward/backward, and Z represents up/down.

**Zero Moment Point (ZMP)**: A concept used in robotics to assess the dynamic stability of a walking robot, representing the point on the ground where the sum of all moments of the active forces equals zero.

## References

For more complete definitions of these terms in the context of Physical AI and robotics:
- Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics*.
- Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.