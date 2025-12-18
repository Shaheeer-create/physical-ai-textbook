// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'Bridging the gap between digital AI and physical robot control',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://shaheer-create.github.io',
  baseUrl: '/physical-ai-textbook/',

  organizationName: 'Shaheer-Create',
  projectName: 'physical-ai-textbook',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'], // Urdu-ready
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/', // 📘 Docs at root (TEXTBOOK MODE)
          editUrl:
            'https://github.com/Shaheer-Create/physical-ai-textbook/edit/main/physical-ai-textbook/',
        },
        blog: false, // ❌ Disable blog (TEXTBOOK)
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  // 🖼️ Image optimization
  plugins: [
    '@docusaurus/plugin-ideal-image',
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',

      colorMode: {
        respectPrefersColorScheme: true,
      },

      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'textbookSidebar',
            label: 'Textbook',
            position: 'left',
          },
          {
            href: 'https://github.com/Shaheer-Create/physical-ai-textbook',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },

      footer: {
        style: 'dark',
        links: [
          {
            title: 'Textbook',
            items: [
              {
                label: 'Part 1: Foundations',
                to: '/part-01-foundations/chapter-1',
              },
              {
                label: 'Part 2: ROS 2',
                to: '/part-02-ros2/chapter-3',
              },
              {
                label: 'Part 3: Simulation',
                to: '/part-03-simulation/chapter-7',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discord',
                href: 'https://discord.gg/robotics',
              },
              {
                label: 'Robotics Stack Exchange',
                href: 'https://robotics.stackexchange.com/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Shaheer-Create/physical-ai-textbook',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook.`,
      },

      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'json', 'yaml'],
      },
    }),
};

export default config;
