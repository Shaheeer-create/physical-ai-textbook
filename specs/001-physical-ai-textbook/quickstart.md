# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- Git
- Python 3.11 (for any backend services)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/[your-org]/physical-ai-textbook.git
cd physical-ai-textbook
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory with the following:
```
# Docusaurus Configuration
BABEL_ENV=development
NODE_ENV=development

# For authentication (if implemented)
AUTH_GITHUB_CLIENT_ID=your_github_client_id
AUTH_GITHUB_CLIENT_SECRET=your_github_client_secret

# For analytics
GA_TRACKING_ID=your_google_analytics_id

# For CDN if self-hosting assets
CDN_URL=https://your-cdn-url.com
```

### 4. Start the Development Server
```bash
npm start
```

This command starts a local development server and opens the application in your browser. Most changes are reflected live without having to restart the server.

### 5. Build for Production
```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static hosting service.

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

## Creating New Content

### Adding a New Chapter
1. Determine which part the chapter belongs to
2. Create a new `.md` or `.mdx` file in the appropriate part directory
3. Use the following frontmatter template:

```markdown
---
title: "Example Chapter Title"
description: "Brief description of the chapter content"
sidebar_position: 1  # Position in the sidebar for this part
---

# Example Chapter Title

## Learning Objectives
- Objective 1
- Objective 2
- Objective 3

## Content
Your chapter content goes here...
```

### Adding Code Examples
For code examples in the textbook, use Docusaurus' built-in code block syntax:

```python
# Example Python code for ROS 2
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
```

### Adding Technical Diagrams
Place diagrams in the `static/img/` directory and reference them using the following syntax:

```markdown
import Image from '@theme/IdealImage';

<Image
  img={require('./img/robot-architecture.png').default}
  alt="Robot Architecture Diagram"
  align="center"
/>
```

## Running Tests

### Frontend Component Tests
```bash
npm test
```

### Content Validation
```bash
npm run check:content
```

### Accessibility Testing
```bash
npm run check:accessibility
```

## Deployment

### To GitHub Pages
```bash
GIT_USER=<Your GitHub username> npm run deploy
```

### To Vercel
1. Push your code to a GitHub repository
2. Connect Vercel to your repository
3. Vercel will automatically build and deploy any changes

### To Netlify
1. Push your code to a GitHub repository
2. Connect Netlify to your repository
3. Netlify will automatically build and deploy any changes

## Common Tasks

### Adding a New Part
1. Create a new directory in `/docs/` with the naming convention `part-XX-name`
2. Update `sidebars.js` to include the new part
3. Create an introductory document for the part

### Updating Navigation
Edit `sidebars.js` to modify the navigation structure of the textbook.

### Custom Styling
- For global styles: Edit `src/css/custom.css`
- For component-specific styles: Use CSS modules or styled components within the component files