---
title: "Chapter Template"
description: "Template for textbook chapters"
sidebar_position: 1
---

# Chapter Title

## Learning Objectives

After completing this chapter, you should be able to:

- Objective 1
- Objective 2
- Objective 3

## Content

Your chapter content goes here...

### Code Examples

```python
# Example Python code for ROS 2
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
```

### Technical Diagrams

import Image from '@theme/IdealImage';

<Image
  img={require('../static/img/robot-architecture.svg').default}
  alt="Robot Architecture Diagram"
  align="center"
/>

## Exercises

1. **Exercise 1**: Question goes here
2. **Exercise 2**: Question goes here

## Summary

Brief summary of key concepts covered in this chapter.