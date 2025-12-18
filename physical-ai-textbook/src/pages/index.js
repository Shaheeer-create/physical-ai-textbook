import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="text--center padding-horiz--md">
          <img src="img/logo.svg" alt="Physical AI & Humanoid Robotics Logo" className={styles.heroLogo} />
          <h1 className={clsx('hero__title', styles.hero__title_text)}>{siteConfig.title}</h1>
        </div>
        <p className={clsx('hero__subtitle', styles.hero__subtitle)}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/part-01-foundations/chapter-1">
            Start Reading - Complete Guide ⏱️
          </Link>
          <Link
            className="button button--primary button--lg"
            to="/part-02-ros2/chapter-3">
            Explore ROS 2 Architecture
          </Link>
        </div>
      </div>
    </header>
  );
}

function PartCard({ title, chapters, slug }) {
  return (
    <div className={clsx('col col--4', styles.cardContainer)}>
      <div className={clsx('card', styles.card, styles.partCard)}>
        <h3 className={styles.partTitle}>{title}</h3>
        <ul className={styles.chapterList}>
          {chapters.map((chapter, index) => (
            <li key={index} className={styles.chapterItem}>
              <Link to={chapter.to}>{chapter.label}</Link>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function TextbookStructure() {
  const parts = [
    {
      title: "Part 1: Foundations",
      chapters: [
        { to: '/part-01-foundations/chapter-1', label: 'Chapter 1: Introduction to Physical AI' },
        { to: '/part-01-foundations/chapter-2', label: 'Chapter 2: The Robotic Sensorium' }
      ]
    },
    {
      title: "Part 2: ROS 2 Architecture",
      chapters: [
        { to: '/part-02-ros2/chapter-3', label: 'Chapter 3: ROS 2 Architecture Fundamentals' },
        { to: '/part-02-ros2/chapter-4', label: 'Chapter 4: Building ROS 2 Packages' },
        { to: '/part-02-ros2/chapter-5', label: 'Chapter 5: Bridging AI to Robot Control' },
        { to: '/part-02-ros2/chapter-6', label: 'Chapter 6: Robot Description with URDF' }
      ]
    },
    {
      title: "Part 3: Simulation Environments",
      chapters: [
        { to: '/part-03-simulation/chapter-7', label: 'Chapter 7: Gazebo Simulation Environment' },
        { to: '/part-03-simulation/chapter-8', label: 'Chapter 8: Robot Modeling in Gazebo' },
        { to: '/part-03-simulation/chapter-9', label: 'Chapter 9: Unity Integration for High-Fidelity Visualization' }
      ]
    },
    {
      title: "Part 4: NVIDIA Isaac Platform",
      chapters: [
        { to: '/part-04-isaac/chapter-10', label: 'Chapter 10: NVIDIA Isaac Platform Overview' },
        { to: '/part-04-isaac/chapter-11', label: 'Chapter 11: Advanced Perception with Isaac' },
        { to: '/part-04-isaac/chapter-12', label: 'Chapter 12: Reinforcement Learning for Robot Control' },
        { to: '/part-04-isaac/chapter-13', label: 'Chapter 13: Navigation and Path Planning' }
      ]
    },
    {
      title: "Part 5: Vision-Language-Action Systems",
      chapters: [
        { to: '/part-05-vla/chapter-14', label: 'Chapter 14: Voice-to-Action Systems' },
        { to: '/part-05-vla/chapter-15', label: 'Chapter 15: Cognitive Planning with LLMs' },
        { to: '/part-05-vla/chapter-16', label: 'Chapter 16: Computer Vision Integration' }
      ]
    },
    {
      title: "Part 6: Capstone Integration",
      chapters: [
        { to: '/part-06-capstone/chapter-17', label: 'Chapter 17: The Autonomous Humanoid Project' },
        { to: '/part-06-capstone/chapter-18', label: 'Chapter 18: Object Manipulation' },
        { to: '/part-06-capstone/chapter-19', label: 'Chapter 19: System Integration and Testing' },
        { to: '/part-06-capstone/chapter-20', label: 'Chapter 20: Future of Physical AI' }
      ]
    }
  ];

  return (
    <section className={styles.chaptersSection}>
      <div className="container">
        <Heading as="h2">Textbook Structure</Heading>
        <div className="row">
          {parts.slice(0, 3).map((part, index) => (
            <div key={index} className={clsx('col col--4', styles.cardContainer)}>
              <div className={clsx('card', styles.card, styles.partCard)}>
                <h3 className={styles.partTitle}>{part.title}</h3>
                <ul className={styles.chapterList}>
                  {part.chapters.map((chapter, chapterIndex) => (
                    <li key={chapterIndex} className={styles.chapterItem}>
                      <Link to={chapter.to}>{chapter.label}</Link>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
        <div className="row">
          {parts.slice(3).map((part, index) => (
            <div key={index+3} className={clsx('col col--4', styles.cardContainer)}>
              <div className={clsx('card', styles.card, styles.partCard)}>
                <h3 className={styles.partTitle}>{part.title}</h3>
                <ul className={styles.chapterList}>
                  {part.chapters.map((chapter, chapterIndex) => (
                    <li key={chapterIndex} className={styles.chapterItem}>
                      <Link to={chapter.to}>{chapter.label}</Link>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics Textbook`}
      description="Complete textbook on bridging digital AI and physical robot control">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <TextbookStructure />
      </main>
    </Layout>
  );
}
