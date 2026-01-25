import { writeFile } from 'fs/promises';
import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  Tensor,
} from '@huggingface/transformers';

// Category types for photos
type PhotoCategory = 'general' | 'face' | 'group' | 'food' | 'landscape' | 'screenshot' | 'drawing';

// Paired quality dimension - each dimension is normalized via sigmoid(pos - neg)
interface QualityDimension {
  name: string;
  positive: string;
  negative: string;
  weight: number;
  categories: PhotoCategory[]; // Which categories this dimension applies to
}

// Category anchor for detection
interface CategoryAnchor {
  category: PhotoCategory;
  anchors: string[]; // Multiple prompts to detect this category
}

// ============================================================================
// CATEGORY DETECTION ANCHORS
// ============================================================================
const categoryAnchors: CategoryAnchor[] = [
  {
    category: 'face',
    anchors: [
      'a close-up portrait of a single person',
      'a selfie photograph',
      'a headshot of one person',
    ],
  },
  {
    category: 'group',
    anchors: [
      'a group photo of multiple people',
      'a family photo with several people',
      'a crowd of people together',
    ],
  },
  {
    category: 'food',
    anchors: [
      'a photograph of food on a plate',
      'a picture of a meal or dish',
      'food photography',
    ],
  },
  {
    category: 'landscape',
    anchors: [
      'a landscape photograph of nature',
      'a scenic outdoor photograph',
      'a photograph of mountains, beach, or countryside',
    ],
  },
  {
    category: 'screenshot',
    anchors: [
      'a screenshot of a phone or computer screen',
      'a screen capture with text and interface',
      'a digital screenshot',
    ],
  },
  {
    category: 'drawing',
    anchors: [
      'a drawing or illustration',
      'digital art or painting',
      'a sketch or artwork',
    ],
  },
  {
    category: 'general',
    anchors: [
      'a photograph',
      'a casual snapshot',
      'an everyday photo',
    ],
  },
];

// ============================================================================
// QUALITY DIMENSIONS
// ============================================================================
const qualityDimensions: QualityDimension[] = [
  // ---------------------------
  // UNIVERSAL TECHNICAL QUALITY
  // ---------------------------
  {
    name: 'sharpness',
    positive: 'a sharp, crisp, in-focus photograph',
    negative: 'a blurry, soft, out of focus photograph',
    weight: 1.5,
    categories: ['general', 'face', 'group', 'food', 'landscape'],
  },
  {
    name: 'exposure',
    positive: 'a well-exposed photograph with good brightness',
    negative: 'a poorly exposed, too dark or washed out photograph',
    weight: 1.3,
    categories: ['general', 'face', 'group', 'food', 'landscape'],
  },
  {
    name: 'noise',
    positive: 'a clean photograph with smooth tones',
    negative: 'a noisy, grainy photograph with visible grain',
    weight: 1.0,
    categories: ['general', 'face', 'group', 'food', 'landscape'],
  },
  {
    name: 'composition',
    positive: 'a well-composed, balanced photograph',
    negative: 'a poorly framed, awkwardly cropped photograph',
    weight: 1.1,
    categories: ['general', 'face', 'group', 'food', 'landscape'],
  },
  {
    name: 'colors',
    positive: 'a photograph with pleasing, natural colors',
    negative: 'a photograph with ugly, unnatural color cast',
    weight: 0.9,
    categories: ['general', 'face', 'group', 'food', 'landscape'],
  },

  // ---------------------------
  // FACE/PORTRAIT SPECIFIC
  // ---------------------------
  {
    name: 'face_expression',
    positive: 'a person with a natural, pleasant expression',
    negative: 'a person with an awkward, unflattering expression',
    weight: 1.4,
    categories: ['face', 'group'],
  },
  {
    name: 'eyes_open',
    positive: 'a person with eyes open, looking alert',
    negative: 'a person with eyes closed or blinking',
    weight: 1.5,
    categories: ['face', 'group'],
  },
  {
    name: 'face_lighting',
    positive: 'a portrait with flattering, soft lighting on face',
    negative: 'a portrait with harsh shadows on face',
    weight: 1.2,
    categories: ['face'],
  },
  {
    name: 'smile',
    positive: 'a person smiling genuinely',
    negative: 'a person frowning or looking unhappy',
    weight: 1.0,
    categories: ['face', 'group'],
  },
  {
    name: 'face_angle',
    positive: 'a flattering angle showing the face well',
    negative: 'an unflattering angle, double chin, or distorted face',
    weight: 1.1,
    categories: ['face'],
  },

  // ---------------------------
  // GROUP PHOTO SPECIFIC
  // ---------------------------
  {
    name: 'everyone_visible',
    positive: 'a group photo with everyone clearly visible',
    negative: 'a group photo with people cut off or hidden',
    weight: 1.4,
    categories: ['group'],
  },
  {
    name: 'group_attention',
    positive: 'everyone in the group looking at camera',
    negative: 'people in the group looking away or distracted',
    weight: 1.2,
    categories: ['group'],
  },
  {
    name: 'group_arrangement',
    positive: 'a well-arranged group photo with good spacing',
    negative: 'a messy group photo with people overlapping awkwardly',
    weight: 1.0,
    categories: ['group'],
  },

  // ---------------------------
  // FOOD SPECIFIC
  // ---------------------------
  {
    name: 'appetizing',
    positive: 'delicious, appetizing looking food',
    negative: 'unappetizing, gross looking food',
    weight: 1.5,
    categories: ['food'],
  },
  {
    name: 'food_presentation',
    positive: 'beautifully presented food on a nice plate',
    negative: 'messy, sloppy food presentation',
    weight: 1.2,
    categories: ['food'],
  },
  {
    name: 'food_lighting',
    positive: 'food with appetizing warm lighting',
    negative: 'food with unflattering harsh or cold lighting',
    weight: 1.1,
    categories: ['food'],
  },
  {
    name: 'food_freshness',
    positive: 'fresh, vibrant looking food',
    negative: 'stale, wilted, old looking food',
    weight: 1.0,
    categories: ['food'],
  },

  // ---------------------------
  // LANDSCAPE SPECIFIC
  // ---------------------------
  {
    name: 'scenic_beauty',
    positive: 'a breathtaking, beautiful scenic view',
    negative: 'a boring, unremarkable view',
    weight: 1.4,
    categories: ['landscape'],
  },
  {
    name: 'natural_lighting',
    positive: 'beautiful natural light, golden hour',
    negative: 'flat, dull lighting conditions',
    weight: 1.3,
    categories: ['landscape'],
  },
  {
    name: 'depth_interest',
    positive: 'a landscape with interesting depth and layers',
    negative: 'a flat, uninteresting landscape',
    weight: 1.0,
    categories: ['landscape'],
  },
  {
    name: 'weather_conditions',
    positive: 'pleasant weather and atmospheric conditions',
    negative: 'poor visibility, hazy, or bad weather',
    weight: 0.8,
    categories: ['landscape'],
  },

  // ---------------------------
  // SCREENSHOT SPECIFIC
  // ---------------------------
  {
    name: 'text_clarity',
    positive: 'a screenshot with clear, readable text',
    negative: 'a screenshot with blurry, unreadable text',
    weight: 1.5,
    categories: ['screenshot'],
  },
  {
    name: 'content_relevance',
    positive: 'a useful screenshot showing important content',
    negative: 'a screenshot with mostly empty space or irrelevant content',
    weight: 1.0,
    categories: ['screenshot'],
  },
  {
    name: 'screen_quality',
    positive: 'a clean screenshot without glare or reflections',
    negative: 'a photo of a screen with glare and reflections',
    weight: 1.2,
    categories: ['screenshot'],
  },

  // ---------------------------
  // DRAWING/ART SPECIFIC
  // ---------------------------
  {
    name: 'artistic_skill',
    positive: 'skillfully drawn artwork with good technique',
    negative: 'poorly drawn, amateur artwork',
    weight: 1.3,
    categories: ['drawing'],
  },
  {
    name: 'artistic_completion',
    positive: 'a finished, complete artwork',
    negative: 'an unfinished, incomplete sketch',
    weight: 1.0,
    categories: ['drawing'],
  },
  {
    name: 'visual_appeal',
    positive: 'visually striking, appealing artwork',
    negative: 'visually unappealing, ugly artwork',
    weight: 1.2,
    categories: ['drawing'],
  },

  // ---------------------------
  // FUN / MEMORABILITY (UNIVERSAL)
  // ---------------------------
  {
    name: 'memorable',
    positive: 'a memorable, special moment captured',
    negative: 'a forgettable, mundane moment',
    weight: 0.8,
    categories: ['general', 'face', 'group', 'landscape'],
  },
  {
    name: 'interesting',
    positive: 'an interesting, engaging photograph',
    negative: 'a boring, uninteresting photograph',
    weight: 0.7,
    categories: ['general', 'face', 'group', 'food', 'landscape'],
  },
  {
    name: 'emotion',
    positive: 'a photo capturing genuine emotion or joy',
    negative: 'a photo with no emotional impact',
    weight: 0.8,
    categories: ['face', 'group'],
  },
  {
    name: 'action',
    positive: 'an exciting action shot or dynamic moment',
    negative: 'a static, lifeless photograph',
    weight: 0.6,
    categories: ['general', 'group'],
  },
  {
    name: 'uniqueness',
    positive: 'a unique, one-of-a-kind photograph',
    negative: 'a generic, common photograph',
    weight: 0.5,
    categories: ['general', 'face', 'group', 'food', 'landscape'],
  },
];

// Use the same model as imageAnalysis.ts (from Hugging Face)
const MODEL_ID = 'plhery/mobileclip2-onnx';
// Available model sizes: 's0', 's2', 'b', 'l14'
const MODEL_SIZE = 's2';

function tensorToArray(t: Tensor): number[] {
  return Array.from(t.data as Float32Array);
}

async function generateEmbedding(
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  textModel: InstanceType<typeof CLIPTextModelWithProjection>,
  text: string
): Promise<number[]> {
  const input = tokenizer(text, { padding: 'max_length', truncation: true, max_length: 77 });
  const outputs = await textModel({ text: input.input_ids, input_ids: input.input_ids });
  const textEmbeds = (outputs.text_embeds ?? outputs.unnorm_text_features) as Tensor;
  const normed = textEmbeds.normalize(2, -1);
  return tensorToArray(normed);
}

async function generateAverageEmbedding(
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  textModel: InstanceType<typeof CLIPTextModelWithProjection>,
  texts: string[]
): Promise<number[]> {
  const embeddings = await Promise.all(texts.map(t => generateEmbedding(tokenizer, textModel, t)));

  // Average the embeddings
  const dim = embeddings[0].length;
  const avg = new Array(dim).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      avg[i] += emb[i] / embeddings.length;
    }
  }

  // Normalize the averaged embedding
  const norm = Math.sqrt(avg.reduce((s, v) => s + v * v, 0));
  return avg.map(v => v / norm);
}

(async () => {
  console.log(`⏬  Loading MobileCLIP2-${MODEL_SIZE.toUpperCase()} text tower from Hugging Face...`);
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, {
    device: 'cpu',
    dtype: 'fp32',
    model_file_name: `${MODEL_SIZE}/text_model`,
  });

  // Generate category detection embeddings
  console.log('🏷️  Generating category detection embeddings...');
  const categories: Record<string, number[]> = {};
  for (const anchor of categoryAnchors) {
    console.log(`   - ${anchor.category}`);
    categories[anchor.category] = await generateAverageEmbedding(tokenizer, textModel, anchor.anchors);
  }

  // Generate quality dimension embeddings
  console.log('📊 Generating quality dimension embeddings...');
  const dimensions: Array<{
    name: string;
    positive: number[];
    negative: number[];
    weight: number;
    categories: PhotoCategory[];
  }> = [];

  for (const dim of qualityDimensions) {
    console.log(`   - ${dim.name}`);
    const positive = await generateEmbedding(tokenizer, textModel, dim.positive);
    const negative = await generateEmbedding(tokenizer, textModel, dim.negative);
    dimensions.push({
      name: dim.name,
      positive,
      negative,
      weight: dim.weight,
      categories: dim.categories,
    });
  }

  const output = {
    version: 2,
    categories,
    dimensions,
    calibration: {
      temperature: 10,      // Sigmoid temperature for dimension scoring
      categoryThreshold: 0.15, // Min category confidence to apply category-specific dimensions
    },
  };

  await writeFile('src/data/qualityEmbeds.json', JSON.stringify(output, null, 2));
  console.log('✅  Wrote src/data/qualityEmbeds.json');
  console.log(`    - ${Object.keys(categories).length} categories`);
  console.log(`    - ${dimensions.length} quality dimensions`);
})();
