# Data Model: Physical AI & Humanoid Robotics Textbook

## Core Entities

### Textbook Chapter
**Description**: Represents a unit of educational content with learning objectives, concepts, examples, and exercises

**Fields**:
- `id` (string): Unique identifier for the chapter (e.g., "ch01", "ch02")
- `title` (string): Title of the chapter
- `partId` (string): Reference to the part this chapter belongs to
- `number` (integer): Sequential number of the chapter in the textbook (1-20)
- `content` (string): Markdown content of the chapter
- `learningObjectives` (array of string): List of learning objectives for the chapter
- `codeExamples` (array of object): Code examples in the chapter
  - `language` (string): Programming language (e.g., "Python", "C++")
  - `code` (string): The actual code snippet
  - `description` (string): Explanation of what the code does
- `diagrams` (array of object): Technical diagrams in the chapter
  - `id` (string): Unique identifier for the diagram
  - `altText` (string): Description for accessibility
  - `caption` (string): Figure caption
- `exercises` (array of object): End-of-chapter exercises
  - `id` (string): Unique identifier for the exercise
  - `question` (string): The exercise question
  - `difficulty` (string): "beginner", "intermediate", "advanced"
- `prerequisites` (array of string): Chapter IDs that should be completed before this one
- `relatedChapters` (array of string): IDs of related chapters for cross-referencing
- `lastModified` (datetime): Timestamp of last content update
- `version` (string): Content version for tracking changes

**Relationships**:
- Belongs to one "Part"
- Can reference multiple other "Chapter" entities (via prerequisites and relatedChapters)
- Contains multiple "CodeExample" and "Diagram" elements

**Validation Rules**:
- Title must not be empty
- Number must be between 1 and 20
- Content must not be empty
- Learning objectives array must contain at least one objective

### Part
**Description**: Represents a major section of the textbook that contains multiple chapters

**Fields**:
- `id` (string): Unique identifier for the part (e.g., "part1", "part2")
- `title` (string): Title of the part
- `number` (integer): Sequential number of the part in the textbook (1-6)
- `description` (string): Brief description of what the part covers
- `chapters` (array of string): IDs of chapters in this part
- `learningOutcomes` (array of string): High-level learning outcomes for the part

**Relationships**:
- Contains multiple "Chapter" entities
- Belongs to the overall textbook

**Validation Rules**:
- Title must not be empty
- Number must be between 1 and 6
- Must contain at least one chapter

### Course Path
**Description**: Represents a curated sequence of chapters designed for specific learning goals or audiences

**Fields**:
- `id` (string): Unique identifier for the course path
- `title` (string): Title of the course path
- `description` (string): Description of what this path is designed to achieve
- `targetAudience` (string): "student", "educator", "developer", or custom string
- `chapters` (array of object): Ordered list of chapters in the path
  - `chapterId` (string): Reference to the chapter
  - `required` (boolean): Whether this chapter is required (vs optional/recommended)
- `estimatedDuration` (string): Estimated time to complete the path (e.g., "4 weeks", "60 hours")
- `prerequisites` (array of string): Knowledge or chapters required before starting
- `learningObjectives` (array of string): Overall objectives of the path

**Relationships**:
- References multiple "Chapter" entities
- Uses the "targetAudience" to determine appropriate content selection

**Validation Rules**:
- Title must not be empty
- Must contain at least one chapter
- At least 70% of referenced chapters must exist

### User
**Description**: Represents a person using the textbook platform

**Fields**:
- `id` (string): Unique identifier for the user
- `email` (string): User's email address (for authentication)
- `role` (string): "student", "educator", or "developer"
- `profile` (object): User profile information
  - `name` (string): User's display name
  - `institution` (string): Educational institution or company (for educators/developers)
  - `background` (string): Technical background (for personalization)
- `createdAt` (datetime): Account creation timestamp
- `lastActive` (datetime): Last activity timestamp
- `preferences` (object): User preferences
  - `theme` (string): "light" or "dark"
  - `language` (string): Preferred language
  - `fontSize` (string): Preferred font size setting

**Relationships**:
- Can have multiple "UserProgress" records
- Can create multiple "CoursePath" entities (for educators)

**Validation Rules**:
- Email must be a valid email format
- Role must be one of the allowed values
- Name must not be empty

### UserProgress
**Description**: Tracks a user's progress through chapters and exercises

**Fields**:
- `id` (string): Unique identifier for the progress record
- `userId` (string): Reference to the user
- `chapterId` (string): Reference to the chapter
- `status` (string): "not-started", "in-progress", "completed"
- `lastAccessed` (datetime): When the user last accessed this chapter
- `completionPercentage` (number): 0-100 percentage of chapter completed
- `exerciseProgress` (array of object): Progress on chapter exercises
  - `exerciseId` (string): Reference to the exercise
  - `completed` (boolean): Whether the exercise was completed
  - `attempts` (array of object): Record of user attempts
    - `solution` (string): User's solution
    - `feedback` (string): Any feedback provided
    - `completedAt` (datetime): When the attempt was made
- `timeSpent` (number): Time spent on the chapter in minutes

**Relationships**:
- Belongs to one "User"
- Belongs to one "Chapter"
- References exercises within that chapter

**Validation Rules**:
- Status must be one of the allowed values
- Completion percentage must be between 0 and 100
- Time spent must be non-negative

### ContentUpdate
**Description**: Tracks changes to textbook content for version management

**Fields**:
- `id` (string): Unique identifier for the update
- `chapterId` (string): Reference to the chapter that was updated
- `version` (string): New version identifier
- `changelog` (string): Description of what changed
- `updatedAt` (datetime): When the update was made
- `author` (string): Who made the update
- `reviewed` (boolean): Whether the update has been reviewed for technical accuracy

**Relationships**:
- References one "Chapter"

**Validation Rules**:
- Changelog must not be empty
- UpdatedAt must not be in the future