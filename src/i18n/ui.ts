// Central bilingual UI dictionary (FR default / EN).
// Consumed via the <T> helper: <T fr={ui.nav.work.fr} en={ui.nav.work.en} />
// or directly with t(key, lang) in scripts.

export type Lang = "fr" | "en";

type Str = { fr: string; en: string };

export const ui = {
  nav: {
    home: { fr: "Accueil", en: "Home" },
    work: { fr: "Projets", en: "Work" },
    about: { fr: "À propos", en: "About" },
    blog: { fr: "Blog", en: "Blog" },
    contact: { fr: "Contact", en: "Contact" },
  },
  hero: {
    greeting: { fr: "Bonjour, je suis", en: "Hi, I'm" },
    available: { fr: "Disponible", en: "Available" },
    ctaWork: { fr: "Voir mes projets", en: "View my work" },
    ctaContact: { fr: "Me contacter", en: "Get in touch" },
    scroll: { fr: "Défiler", en: "Scroll" },
  },
  home: {
    stackTitle: { fr: "Ma stack technique", en: "My tech stack" },
    doTitle: { fr: "Ce que je fais", en: "What I do" },
    doSubtitle: {
      fr: "Des solutions de bout en bout, du mobile à la donnée.",
      en: "End-to-end solutions, from mobile to data.",
    },
    featuredTitle: { fr: "Projets phares", en: "Featured work" },
    featuredSubtitle: {
      fr: "Une sélection de ce que j'ai construit récemment.",
      en: "A selection of things I've built recently.",
    },
    seeAll: { fr: "Tous les projets", en: "See all work" },
  },
  services: {
    mobile: { fr: "Développement mobile", en: "Mobile development" },
    mobileDesc: {
      fr: "Applications multi-plateformes performantes avec Flutter & Dart.",
      en: "High-performance cross-platform apps with Flutter & Dart.",
    },
    backend: { fr: "Backend & API", en: "Backend & APIs" },
    backendDesc: {
      fr: "Services robustes en NestJS, Django REST et Node.js.",
      en: "Robust services with NestJS, Django REST and Node.js.",
    },
    data: { fr: "Data engineering", en: "Data engineering" },
    dataDesc: {
      fr: "Pipelines de données, IA & Big Data pour la prise de décision.",
      en: "Data pipelines, AI & Big Data for decision-making.",
    },
  },
  about: {
    title: { fr: "À propos de moi", en: "About me" },
    subtitle: { fr: "Mon parcours en bref", en: "My story in short" },
    experience: { fr: "Expérience", en: "Experience" },
    education: { fr: "Formation", en: "Education" },
    skills: { fr: "Compétences", en: "Skills" },
    languages: { fr: "Langues", en: "Languages" },
    interests: { fr: "Centres d'intérêt", en: "Interests" },
    present: { fr: "Présent", en: "Present" },
    showMore: { fr: "Voir plus", en: "Show more" },
    showLess: { fr: "Voir moins", en: "Show less" },
    summaryLabel: { fr: "Résumé", en: "Summary" },
    responsibilitiesLabel: { fr: "Responsabilités", en: "Responsibilities" },
    achievementsLabel: { fr: "Réalisations", en: "Achievements" },
  },
  work: {
    title: { fr: "Mes projets", en: "My work" },
    subtitle: {
      fr: "Applications, sites et expérimentations.",
      en: "Apps, sites and experiments.",
    },
    all: { fr: "Tout", en: "All" },
    liveLabel: { fr: "Voir le live", en: "Live demo" },
    codeLabel: { fr: "Code source", en: "Source code" },
    experienceTitle: { fr: "Expérience professionnelle", en: "Work experience" },
  },
  blog: {
    title: { fr: "Blog", en: "Blog" },
    subtitle: {
      fr: "Notes sur le data engineering, le machine learning et l'IA.",
      en: "Notes on data engineering, machine learning and AI.",
    },
    minRead: { fr: "min de lecture", en: "min read" },
    readArticle: { fr: "Lire l'article", en: "Read article" },
    back: { fr: "Retour au blog", en: "Back to blog" },
    empty: {
      fr: "Aucun article pour le moment. Reviens bientôt !",
      en: "No posts yet. Check back soon!",
    },
    updatedOn: { fr: "Mis à jour le", en: "Updated on" },
    soon: { fr: "Bientôt disponible", en: "Coming soon" },
    desc: {
      fr: "J'écris bientôt sur le développement mobile, le data engineering et l'IA. Reviens vite !",
      en: "I'll soon write about mobile development, data engineering and AI. Check back soon!",
    },
    cta: { fr: "Me contacter en attendant", en: "Get in touch meanwhile" },
  },
  contact: {
    title: { fr: "Travaillons ensemble", en: "Let's work together" },
    subtitle: {
      fr: "Une idée, un projet, une opportunité ? Écris-moi.",
      en: "Got an idea, a project or an opportunity? Drop me a line.",
    },
    name: { fr: "Nom", en: "Name" },
    namePlaceholder: { fr: "Votre nom", en: "Your name" },
    subject: { fr: "Sujet", en: "Subject" },
    subjectPlaceholder: { fr: "Sujet de votre message", en: "Subject of your message" },
    message: { fr: "Message", en: "Message" },
    messagePlaceholder: { fr: "Votre message...", en: "Your message..." },
    send: { fr: "Envoyer le message", en: "Send message" },
    orReach: { fr: "Ou joignez-moi directement", en: "Or reach me directly" },
    copy: { fr: "Copier", en: "Copy" },
    copied: { fr: "Copié !", en: "Copied!" },
  },
  footer: {
    builtWith: { fr: "Conçu avec Astro", en: "Built with Astro" },
    rights: { fr: "Tous droits réservés", en: "All rights reserved" },
    palette: { fr: "Ouvrir la palette", en: "Open command palette" },
  },
  a11y: {
    toggleTheme: { fr: "Changer de thème", en: "Toggle theme" },
    toggleLang: { fr: "Changer de langue", en: "Toggle language" },
    menu: { fr: "Menu", en: "Menu" },
  },
} satisfies Record<string, Record<string, Str>>;

export type Ui = typeof ui;
