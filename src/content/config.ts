import { defineCollection, z } from "astro:content";

// Canonical (FR) posts — drive URLs, card metadata and the FR body.
const blog = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    title_en: z.string().optional(),
    description: z.string(),
    description_en: z.string().optional(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    author: z.string().default("d'ALMEIDA Hyacinthe"),
    draft: z.boolean().default(false),
  }),
});

// English bodies — matched to a FR post by identical slug. Not routed.
const blogEn = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string().optional(),
    description: z.string().optional(),
  }),
});

export const collections = { blog, "blog-en": blogEn };
