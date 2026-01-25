import { writeFile } from 'fs/promises';
import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  Tensor,
  env,
} from '@huggingface/transformers';
import path from 'path';

// Weighted prompts for general photo quality
// Weight indicates importance: higher = more impact on final score
interface WeightedPrompt {
  text: string;
  weight: number;
}

const generalPositivePrompts: WeightedPrompt[] = [
  // Technical quality (highest weights)
  { text: 'a sharp, in-focus photograph', weight: 1.5 },
  { text: 'a well-exposed photograph with good lighting', weight: 1.3 },
  { text: 'a high resolution, detailed photo', weight: 1.2 },
  { text: 'a clear photo with no blur', weight: 1.4 },
  // Composition & aesthetics
  { text: 'a well-composed photograph', weight: 1.1 },
  { text: 'a photo with balanced colors', weight: 1.0 },
  { text: 'a visually appealing image', weight: 1.0 },
  { text: 'a professional quality photograph', weight: 1.2 },
  // Scene quality
  { text: 'a clean, uncluttered background', weight: 0.8 },
  { text: 'a vibrant, colorful photo', weight: 0.9 },
];

const generalNegativePrompts: WeightedPrompt[] = [
  // Technical problems (highest weights)
  { text: 'a blurry, out of focus photo', weight: 1.5 },
  { text: 'a motion-blurred photograph', weight: 1.4 },
  { text: 'an underexposed, too dark photo', weight: 1.3 },
  { text: 'an overexposed, washed out photo', weight: 1.3 },
  { text: 'a noisy, grainy photograph', weight: 1.2 },
  // Composition issues
  { text: 'a poorly framed photo', weight: 1.0 },
  { text: 'a cluttered, messy background', weight: 0.9 },
  { text: 'a low quality amateur snapshot', weight: 1.1 },
  // Artifacts
  { text: 'a pixelated, low resolution image', weight: 1.2 },
  { text: 'a photo with compression artifacts', weight: 1.0 },
];

// Face-specific prompts - used when a face is detected in the image
const facePositivePrompts: WeightedPrompt[] = [
  // Face/portrait quality (highest priority)
  { text: 'a sharp portrait with face in perfect focus', weight: 1.6 },
  { text: 'a well-lit portrait with flattering lighting', weight: 1.4 },
  { text: 'a portrait with natural, pleasant expression', weight: 1.3 },
  { text: 'a person with eyes open and looking at camera', weight: 1.5 },
  { text: 'a smiling, happy person', weight: 1.2 },
  // Technical (faces)
  { text: 'a portrait with sharp facial features', weight: 1.4 },
  { text: 'a clear photo of a person', weight: 1.1 },
  { text: 'a professional headshot', weight: 1.2 },
  // Expression & pose
  { text: 'a candid, natural-looking photo of a person', weight: 1.0 },
  { text: 'an attractive, flattering portrait', weight: 1.0 },
];

const faceNegativePrompts: WeightedPrompt[] = [
  // Face problems (highest priority)
  { text: 'a blurry face, out of focus portrait', weight: 1.6 },
  { text: 'a person with closed eyes or blinking', weight: 1.5 },
  { text: 'an unflattering photo of a person', weight: 1.3 },
  { text: 'a person making an awkward expression', weight: 1.2 },
  { text: 'a badly lit face with harsh shadows', weight: 1.4 },
  // Technical (faces)
  { text: 'a person looking away from camera', weight: 1.0 },
  { text: 'a portrait with red-eye effect', weight: 1.2 },
  { text: 'a face partially cut off or cropped badly', weight: 1.3 },
  // General portrait issues
  { text: 'a person with motion blur', weight: 1.4 },
  { text: 'a dark, underexposed portrait', weight: 1.3 },
];

const MODEL_ID = 'mobileclip2-b';

function tensorToNested(t: Tensor): number[][] {
  const [rows, cols] = t.dims;
  const out: number[][] = [];
  const data = t.data as Float32Array;
  for (let r = 0; r < rows; r++) {
    out.push(Array.from(data.slice(r * cols, (r + 1) * cols)));
  }
  return out;
}

async function generateEmbeddings(
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  textModel: InstanceType<typeof CLIPTextModelWithProjection>,
  prompts: WeightedPrompt[]
): Promise<{ embeddings: number[][]; weights: number[] }> {
  const embeddings: number[][] = [];
  const weights = prompts.map(p => p.weight);

  for (const prompt of prompts) {
    const input = tokenizer(prompt.text, { padding: 'max_length', truncation: true, max_length: 77 });
    const outputs = await textModel({ text: input.input_ids });
    const textEmbeds = (outputs.text_embeds ?? outputs.unnorm_text_features) as Tensor;
    const normed = textEmbeds.normalize(2, -1);
    embeddings.push(tensorToNested(normed)[0]);
  }

  return { embeddings, weights };
}

(async () => {
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = path.resolve(process.cwd(), 'public', 'models');

  console.log('⏬  loading local MobileCLIP2-B text tower…');
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { local_files_only: true });
  const textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, {
    device: 'cpu',
    dtype: 'fp32',
    local_files_only: true,
  });

  console.log('🔧 Generating general quality embeddings...');
  const generalPositive = await generateEmbeddings(tokenizer, textModel, generalPositivePrompts);
  const generalNegative = await generateEmbeddings(tokenizer, textModel, generalNegativePrompts);

  console.log('🔧 Generating face quality embeddings...');
  const facePositive = await generateEmbeddings(tokenizer, textModel, facePositivePrompts);
  const faceNegative = await generateEmbeddings(tokenizer, textModel, faceNegativePrompts);

  const json = {
    general: {
      positive: generalPositive.embeddings,
      positiveWeights: generalPositive.weights,
      negative: generalNegative.embeddings,
      negativeWeights: generalNegative.weights,
    },
    face: {
      positive: facePositive.embeddings,
      positiveWeights: facePositive.weights,
      negative: faceNegative.embeddings,
      negativeWeights: faceNegative.weights,
    },
    // Calibration parameters - can be tuned
    calibration: {
      slope: 12,        // Spread of score distribution
      offset: 0.5,      // Center point adjustment
      faceWeight: 0.6,  // How much face quality contributes when face detected (0-1)
    }
  };

  await writeFile('src/data/qualityEmbeds.json', JSON.stringify(json, null, 2));
  console.log('✅  Wrote src/data/qualityEmbeds.json');
})();
