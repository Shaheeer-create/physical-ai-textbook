// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Part 1: Foundations',
      link: {
        type: 'generated-index',
        title: 'Part 1: Foundations',
        description: 'Introduction to Physical AI and the robotic sensorium',
        slug: '/part-01-foundations',
      },
      items: [
        {
          type: 'doc',
          id: 'part-01-foundations/chapter-1',
          label: 'Chapter 1: Introduction to Physical AI'
        },
        {
          type: 'doc',
          id: 'part-01-foundations/chapter-2',
          label: 'Chapter 2: The Robotic Sensorium'
        }
      ],
    },
    {
      type: 'category',
      label: 'Part 2: The Robotic Nervous System - ROS 2',
      link: {
        type: 'generated-index',
        title: 'Part 2: The Robotic Nervous System - ROS 2',
        description: 'ROS 2 architecture fundamentals and packages',
        slug: '/part-02-ros2',
      },
      items: [
        {
          type: 'doc',
          id: 'part-02-ros2/chapter-3',
          label: 'Chapter 3: ROS 2 Architecture Fundamentals'
        },
        {
          type: 'doc',
          id: 'part-02-ros2/chapter-4',
          label: 'Chapter 4: Building ROS 2 Packages'
        }
      ],
    },
    {
      type: 'category',
      label: 'Part 3: Digital Twin Simulation',
      link: {
        type: 'generated-index',
        title: 'Part 3: Digital Twin Simulation',
        description: 'Gazebo simulation and Unity integration',
        slug: '/part-03-simulation',
      },
      items: [
        {
          type: 'doc',
          id: 'part-03-simulation/chapter-7',
          label: 'Chapter 7: Gazebo Simulation Environment'
        },
        {
          type: 'doc',
          id: 'part-03-simulation/chapter-8',
          label: 'Chapter 8: Robot Modeling in Gazebo'
        },
        {
          type: 'doc',
          id: 'part-03-simulation/chapter-9',
          label: 'Chapter 9: Unity Integration for High-Fidelity Visualization'
        }
      ],
    },
    {
      type: 'category',
      label: 'Part 4: The AI-Robot Brain - NVIDIA Isaac',
      link: {
        type: 'generated-index',
        title: 'Part 4: The AI-Robot Brain - NVIDIA Isaac',
        description: 'NVIDIA Isaac platform and perception algorithms',
        slug: '/part-04-isaac',
      },
      items: [
        {
          type: 'doc',
          id: 'part-04-isaac/chapter-10',
          label: 'Chapter 10: NVIDIA Isaac Platform Overview'
        },
        {
          type: 'doc',
          id: 'part-04-isaac/chapter-11',
          label: 'Chapter 11: Advanced Perception with Isaac'
        },
        {
          type: 'doc',
          id: 'part-04-isaac/chapter-12',
          label: 'Chapter 12: Reinforcement Learning for Robot Control'
        },
        {
          type: 'doc',
          id: 'part-04-isaac/chapter-13',
          label: 'Chapter 13: Navigation and Path Planning'
        }
      ],
    },
    {
      type: 'category',
      label: 'Part 5: Vision-Language-Action Integration',
      link: {
        type: 'generated-index',
        title: 'Part 5: Vision-Language-Action Integration',
        description: 'Voice-to-action systems and cognitive planning',
        slug: '/part-05-vla',
      },
      items: [
        {
          type: 'doc',
          id: 'part-05-vla/chapter-14',
          label: 'Chapter 14: Voice-to-Action Systems'
        },
        {
          type: 'doc',
          id: 'part-05-vla/chapter-15',
          label: 'Chapter 15: Cognitive Planning with LLMs'
        },
        {
          type: 'doc',
          id: 'part-05-vla/chapter-16',
          label: 'Chapter 16: Computer Vision Integration'
        }
      ],
    },
    {
      type: 'category',
      label: 'Part 6: Capstone Integration',
      link: {
        type: 'generated-index',
        title: 'Part 6: Capstone Integration',
        description: 'Autonomous humanoid project and system integration',
        slug: '/part-06-capstone',
      },
      items: [
        {
          type: 'doc',
          id: 'part-06-capstone/chapter-17',
          label: 'Chapter 17: The Autonomous Humanoid Project'
        },
        {
          type: 'doc',
          id: 'part-06-capstone/chapter-18',
          label: 'Chapter 18: Object Manipulation'
        },
        {
          type: 'doc',
          id: 'part-06-capstone/chapter-19',
          label: 'Chapter 19: System Integration and Testing'
        },
        {
          type: 'doc',
          id: 'part-06-capstone/chapter-20',
          label: 'Chapter 20: Future of Physical AI'
        }
      ],
    }
  ],
};

module.exports = sidebars;