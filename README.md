# Physical AI & Humanoid Robotics Textbook

This is an interactive textbook on Physical AI & Humanoid Robotics, designed to bridge the gap between digital AI knowledge and physical robot control. The textbook provides students with practical knowledge to apply AI concepts in controlling humanoid robots in both simulated and real-world environments.

## Features

- Comprehensive coverage of Physical AI and Humanoid Robotics concepts
- Interactive examples and code snippets
- Integration with ROS 2, NVIDIA Isaac, and simulation environments
- Accessibility-first design (WCAG 2.1 AA compliant)
- Responsive layout for all device sizes
- AI-assisted content generation

## Technologies Used

- [Docusaurus](https://docusaurus.io/): Static site generator optimized for documentation
- [React](https://reactjs.org/): JavaScript library for building user interfaces
- [Node.js](https://nodejs.org/): JavaScript runtime environment
- [Python](https://www.python.org/): For backend services and AI integration

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- Git
- Python 3.11 (for any backend services)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-org]/physical-ai-textbook.git
   cd physical-ai-textbook
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the result.

## Project Structure

```
physical-ai-textbook/
├── docs/                    # Textbook content organized by parts
│   ├── part-01-foundations/ # Foundations content (Chapters 1-2)
│   ├── part-02-ros2/        # ROS 2 content (Chapters 3-6)
│   ├── part-03-simulation/  # Simulation content (Chapters 7-9)
│   ├── part-04-isaac/       # Isaac content (Chapters 10-13)
│   ├── part-05-vla/         # Vision-Language-Action content (Chapters 14-16)
│   └── part-06-capstone/    # Capstone content (Chapters 17-20)
├── src/                     # Custom React components and theme
│   ├── components/          # Reusable React components
│   ├── pages/               # Standalone pages
│   └── theme/               # Custom theme components
├── static/                  # Static assets (images, documents, etc.)
├── docusaurus.config.js     # Main Docusaurus configuration
├── sidebars.js              # Navigation sidebar configuration
└── package.json             # Project metadata and dependencies
```

## Contributing

We welcome contributions to enhance this textbook! Please read our contributing guidelines before making changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.