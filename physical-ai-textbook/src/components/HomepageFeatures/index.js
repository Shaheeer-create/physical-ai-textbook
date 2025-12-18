import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Physical AI Foundations',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Learn the fundamentals of bridging digital AI and physical robot control,
        understanding the unique challenges of embodied intelligence.
      </>
    ),
  },
  {
    title: 'Modern Robotics Stack',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Master ROS 2, NVIDIA Isaac, Unity simulation, and Vision-Language-Action systems
        that power today's advanced robotics applications.
      </>
    ),
  },
  {
    title: 'Complete Learning Path',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        From basic concepts to capstone projects, follow a structured curriculum
        designed to build expertise in humanoid robotics.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
