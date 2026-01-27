import { writeFile } from 'fs/promises';
import {
  AutoTokenizer,
  SiglipTextModel,
  Tensor,
} from '@huggingface/transformers';

// Category types for photos
type PhotoCategory = 'general' | 'face' | 'group' | 'food' | 'landscape' | 'screenshot' | 'drawing' | 'pet' | 'document' | 'night';

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
// More specific, visually-descriptive prompts work better with CLIP
// ============================================================================
const categoryAnchors: CategoryAnchor[] = [
  {
    category: 'face',
    anchors: [
      'a close-up photo of one person\'s face',
      'a selfie of a single person looking at the camera',
      'a portrait headshot showing someone\'s face clearly',
      'a photo focused on one human face',
      'one person posing for a photo, face visible',
    ],
  },
  {
    category: 'group',
    anchors: [
      'a photo of multiple people standing together',
      'a group of friends posing for a picture',
      'several people in a photo together smiling',
      'a family gathered together for a photo',
      'many faces visible in a group photograph',
    ],
  },
  {
    category: 'food',
    anchors: [
      'a close-up photo of food on a plate',
      'a delicious meal photographed from above',
      'a dish of prepared food ready to eat',
      'restaurant food photography showing a meal',
      'a plate of food with visible ingredients',
    ],
  },
  {
    category: 'landscape',
    anchors: [
      'a wide landscape photo of mountains and sky',
      'a scenic nature photograph with trees and horizon',
      'an outdoor vista showing natural scenery',
      'a photograph of the beach and ocean',
      'a wide-angle photo of countryside or wilderness',
    ],
  },
  {
    category: 'screenshot',
    anchors: [
      'a screenshot of a mobile phone screen with apps',
      'a computer desktop screenshot showing windows',
      'a screen capture of a website or application',
      'a digital screenshot with user interface elements',
      'a phone screenshot showing a chat conversation',
    ],
  },
  {
    category: 'drawing',
    anchors: [
      'a hand-drawn illustration or sketch on paper',
      'digital artwork created on a computer',
      'a cartoon or animated style drawing',
      'an artistic painting or illustration',
      'a pencil sketch or colored drawing',
    ],
  },
  {
    category: 'general',
    anchors: [
      'a photo of an object or thing',
      'a photograph of indoor room or furniture',
      'a picture of a building or architecture',
      'a casual photograph of everyday items',
      'a photograph of a vehicle or machine',
    ],
  },
  {
    category: 'pet',
    anchors: [
      'a close-up photo of a pet dog or cat',
      'an animal looking at the camera',
      'a cute pet portrait photograph',
      'a photo of a furry animal companion',
      'a dog or cat posing for a picture',
    ],
  },
  {
    category: 'document',
    anchors: [
      'a photo of a paper document or receipt',
      'a photograph of printed text on paper',
      'a scanned document or form',
      'a photo of handwritten notes',
      'a photographed invoice or bill',
    ],
  },
  {
    category: 'night',
    anchors: [
      'a nighttime photograph with city lights',
      'a low-light photo taken at night',
      'a photograph of stars or night sky',
      'a dark atmospheric photograph',
      'a night scene with artificial lighting',
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
    categories: ['general', 'face', 'group', 'food', 'landscape', 'pet'],
  },
  {
    name: 'exposure',
    positive: 'a well-exposed photograph with good brightness',
    negative: 'a poorly exposed, too dark or washed out photograph',
    weight: 1.3,
    categories: ['general', 'face', 'group', 'food', 'landscape', 'pet'],
  },
  {
    name: 'noise',
    positive: 'a clean photograph with smooth tones',
    negative: 'a noisy, grainy photograph with visible grain',
    weight: 1.0,
    categories: ['general', 'face', 'group', 'food', 'landscape', 'pet'],
  },
  {
    name: 'composition',
    positive: 'a well-composed, balanced photograph',
    negative: 'a poorly framed, awkwardly cropped photograph',
    weight: 1.1,
    categories: ['general', 'face', 'group', 'food', 'landscape', 'pet', 'night'],
  },
  {
    name: 'colors',
    positive: 'a photograph with pleasing, natural colors',
    negative: 'a photograph with ugly, unnatural color cast',
    weight: 0.9,
    categories: ['general', 'face', 'group', 'food', 'landscape', 'pet'],
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
    weight: 1.3,
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
    weight: 1.3,
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
    weight: 1.7,
    categories: ['group'],
  },
  {
    name: 'group_attention',
    positive: 'everyone in the group looking at camera',
    negative: 'people in the group looking away or distracted',
    weight: 1.5,
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
  // PET/ANIMAL SPECIFIC
  // ---------------------------
  {
    name: 'pet_attention',
    positive: 'a pet looking at the camera, alert and engaged',
    negative: 'a pet looking away, distracted or sleeping',
    weight: 1.4,
    categories: ['pet'],
  },
  {
    name: 'pet_expression',
    positive: 'a pet with a cute, expressive face',
    negative: 'a pet with dull, uninteresting expression',
    weight: 1.3,
    categories: ['pet'],
  },
  {
    name: 'pet_posture',
    positive: 'a pet in a flattering, photogenic position',
    negative: 'a pet in an awkward or unflattering position',
    weight: 1.0,
    categories: ['pet'],
  },
  {
    name: 'pet_fur',
    positive: 'a pet with clean, well-groomed fur or coat',
    negative: 'a pet with messy, dirty, or matted fur',
    weight: 0.8,
    categories: ['pet'],
  },

  // ---------------------------
  // DOCUMENT SPECIFIC
  // ---------------------------
  {
    name: 'document_flat',
    positive: 'a flat, well-aligned document photograph',
    negative: 'a skewed, crumpled, or curved document photo',
    weight: 1.4,
    categories: ['document'],
  },
  {
    name: 'document_complete',
    positive: 'a complete document with all edges visible',
    negative: 'a cropped document with cut-off text or edges',
    weight: 1.3,
    categories: ['document'],
  },
  {
    name: 'document_legible',
    positive: 'a document photo with clear, readable text',
    negative: 'a document photo with blurry, illegible text',
    weight: 1.5,
    categories: ['document'],
  },
  {
    name: 'document_lighting',
    positive: 'a document with even, shadow-free lighting',
    negative: 'a document with shadows, glare, or uneven lighting',
    weight: 1.1,
    categories: ['document'],
  },

  // ---------------------------
  // NIGHT PHOTOGRAPHY SPECIFIC
  // ---------------------------
  {
    name: 'night_exposure',
    positive: 'a well-exposed night photograph with visible details',
    negative: 'an underexposed, too dark night photograph',
    weight: 1.3,
    categories: ['night'],
  },
  {
    name: 'night_sharpness',
    positive: 'a sharp night photograph without camera shake',
    negative: 'a blurry night photograph with motion blur from camera shake',
    weight: 1.4,
    categories: ['night'],
  },
  {
    name: 'night_lights',
    positive: 'beautiful light trails or bokeh in night photography',
    negative: 'blown out, overexposed lights in night photography',
    weight: 1.0,
    categories: ['night'],
  },
  {
    name: 'night_atmosphere',
    positive: 'atmospheric, moody night photograph',
    negative: 'flat, uninteresting night photograph',
    weight: 1.1,
    categories: ['night'],
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
const MODEL_ID = 'onnx-community/siglip2-base-patch16-512-ONNX';

function tensorToArray(t: Tensor): number[] {
  return Array.from(t.data as Float32Array);
}

async function generateEmbedding(
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  textModel: InstanceType<typeof SiglipTextModel>,
  text: string
): Promise<number[]> {
  const input = tokenizer(text, { padding: 'max_length', truncation: true, max_length: 64 });
  const outputs = await textModel(input);
  // SigLIP uses pooler_output
  const textEmbeds = (outputs.pooler_output ?? outputs.text_embeds ?? outputs.last_hidden_state) as Tensor;
  const normed = textEmbeds.normalize ? textEmbeds.normalize(2, -1) : textEmbeds;
  return tensorToArray(normed);
}

async function generateAverageEmbedding(
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  textModel: InstanceType<typeof SiglipTextModel>,
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
  console.log(`⏬  Loading SigLIP2 text model from Hugging Face...`);
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const textModel = await SiglipTextModel.from_pretrained(MODEL_ID, {
    device: 'cpu',
    dtype: 'fp32',
  });

  // Generate category detection embeddings (store all anchors, not averaged)
  console.log('🏷️  Generating category detection embeddings...');
  const categories: Record<string, number[][]> = {};
  for (const anchor of categoryAnchors) {
    console.log(`   - ${anchor.category} (${anchor.anchors.length} anchors)`);
    const anchorEmbeddings = await Promise.all(
      anchor.anchors.map(text => generateEmbedding(tokenizer, textModel, text))
    );
    categories[anchor.category] = anchorEmbeddings;
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
      temperature: 18,           // Sigmoid temperature for dimension scoring (higher = more spread from 0.5)
      categoryTemperature: 20,   // Softmax temperature for category detection (higher = more peaked)
      categoryThreshold: 0.20,   // Min category confidence to apply category-specific dimensions
    },
  };

  await writeFile('src/data/qualityEmbeds.json', JSON.stringify(output, null, 2));
  console.log('✅  Wrote src/data/qualityEmbeds.json');
  console.log(`    - ${Object.keys(categories).length} categories`);
  console.log(`    - ${dimensions.length} quality dimensions`);
})();
