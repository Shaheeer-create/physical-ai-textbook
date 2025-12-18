<!--

Sync Impact Report:
- Version change: N/A → 1.0.0
- Added sections: All principles and sections based on user input
- Templates requiring updates: ⚠ pending (need to check .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md)
- Follow-up TODOs: None

-->
# Physical AI & Humanoid Robotics Textbook Constitution
<!-- Example: Spec Constitution, TaskFlow Constitution, etc. -->

## Core Principles

### AI-First Development Approach
- **Spec-Driven Methodology**: All features must be specified before implementation using Spec-Kit Plus workflow
- **Claude Code Integration**: Leverage AI for code generation, review, and optimization
- **Subagent Architecture**: Create reusable intelligence components for scalable development
- **Continuous AI Enhancement**: Regularly evaluate and improve AI-generated components

### Educational Excellence Standards
- **Accuracy First**: All technical content must be peer-reviewed and factually accurate
- **Progressive Complexity**: Content must follow logical learning progression from basics to advanced concepts
- **Practical Application**: Every concept must include real-world examples and hands-on exercises
- **Multimodal Learning**: Support text, code, diagrams, and interactive simulations

### Technical Architecture Standards
- **Modern Stack**: Use cutting-edge but stable technologies
- **Cloud-Native Design**: Build for scalability and reliability
- **API-First Approach**: All functionality must be accessible via well-documented APIs
- **Microservices Architecture**: Separate concerns for maintainability and scaling

### User Experience Excellence
- **Accessibility First**: WCAG 2.1 AA compliance minimum
- **Responsive Design**: Perfect experience on all devices
- **Performance Optimization**: &lt;2s page load time, &lt;100ms API response time
- **Intuitive Navigation**: Users should find content within 3 clicks maximum

### Personalization & Localization
- **Background-Aware Content**: Adapt content based on user's software/hardware expertise
- **Dynamic Difficulty Adjustment**: Content complexity adapts to user comprehension
- **Cultural Sensitivity**: Ensure content is appropriate for global audience
- **Urdu Translation Quality**: Professional-grade translation with technical accuracy

### Quality Standards
- **TypeScript Strict Mode**: All frontend code must pass strict TypeScript checking
- **Python Type Hints**: All backend code must include comprehensive type annotations
- **Test Coverage**: Minimum 80% code coverage for all critical paths
- **Code Reviews**: All code must be AI-reviewed and human-verified before merge


## Technology Governance
- **Approved Technology Stack**:
  - **Frontend**: Docusaurus (latest stable), React, TypeScript
  - **Backend**: FastAPI (Python), OpenAI Agents SDK
  - **Database**: Neon Serverless Postgres, Qdrant Cloud (Free Tier)
  - **Authentication**: Better Auth with custom profiling
  - **AI/ML**: OpenAI GPT models, Claude Code, custom subagents
  - **Deployment**: Vercel/Netlify for frontend, cloud-native backend
  - **Monitoring**: Comprehensive logging and analytics
- **Development Workflow**: Specification → Planning → Implementation → Testing → Review → Documentation phases using Spec-Kit Plus tools

## Feature Development Guidelines
- **RAG Chatbot Development**: Context awareness, selective text analysis, citation system, fallback mechanisms
- **Personalization System**: Dynamic content generation, learning progression tracking, skill assessment, content recommendations
- **Translation System**: Technical accuracy in Urdu translations, cultural adaptation, real-time translation, translation memory

## Governance
This constitution serves as the foundation for all development decisions and must be referenced in every specification, planning, and implementation phase using the Spec-Kit Plus workflow. All features must follow the specified development workflow and comply with the technology governance guidelines.

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->