import { Photo, PhotoGroup, PhotoMetadata } from '../types';
import {
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  Tensor,
  matmul,
} from '@huggingface/transformers';
import ExifReader from 'exifreader';
import qualityEmbedsData from '../data/qualityEmbeds.json';

// Category types for photos
export type PhotoCategory = 'general' | 'face' | 'group' | 'food' | 'landscape' | 'screenshot' | 'drawing';

// Quality dimension with paired positive/negative embeddings
interface QualityDimension {
  name: string;
  positive: Tensor;
  negative: Tensor;
  weight: number;
  categories: PhotoCategory[];
}

interface PreparedQualityEmbeddings {
  version: number;
  categories: Map<PhotoCategory, Tensor>;
  dimensions: QualityDimension[];
  calibration: {
    temperature: number;
    categoryThreshold: number;
  };
}

/* ---------- Error handling helper -------------------------------------- */
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  if (typeof error === 'object' && error !== null) {
    const obj = error as Record<string, unknown>;
    if ('message' in obj && typeof obj.message === 'string') {
      return obj.message;
    }
    if ('error' in obj && typeof obj.error === 'string') {
      return obj.error;
    }
  }
  return `Unknown error (${typeof error}): ${String(error)}`;
}

/* ---------- Tensor disposal utilities ---------------------------------- */
function disposeTensor(tensor: Tensor | null | undefined): void {
  if (tensor && typeof tensor.dispose === 'function') {
    try {
      tensor.dispose();
    } catch (e) {
      console.warn('Failed to dispose tensor:', e);
    }
  }
}

function disposeTensors(...tensors: (Tensor | null | undefined)[]): void {
  tensors.forEach(disposeTensor);
}

/* ---------- Model lazy‑loading ----------------------------------------- */
type VisionDevice = 'auto' | 'wasm' | 'webgpu' | 'cpu';
type ModelDtype = 'fp32' | 'fp16' | 'q8' | 'q4';

let visionModelPromise: Promise<InstanceType<typeof CLIPVisionModelWithProjection>> | null = null;
let processorPromise: Promise<InstanceType<typeof AutoProcessor>> | null = null;
let currentDevice: VisionDevice | null = null;
let currentDtype: ModelDtype | null = null;

function getDefaultDevice(): VisionDevice {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    return 'webgpu';
  }
  if (typeof window !== 'undefined') {
    return 'wasm';
  }
  return 'cpu';
}

function getDefaultDtype(_device: VisionDevice): ModelDtype {
  return 'fp32';
}

/**
 * Preheat the model by loading it and running a dummy inference.
 * Call this on app startup to eliminate first-image delay.
 */
export async function preheatModel(): Promise<void> {
  let pixelValues: Tensor | null = null;
  let outputs: any = null;

  try {
    const { processor, vision_model } = await getModels();

    // Create a small dummy image (1x1 pixel, will be resized to 224x224 by processor)
    const dummyImage = new RawImage(new Uint8ClampedArray([128, 128, 128, 255]), 1, 1, 4);
    const image_inputs = await (processor as unknown as (images: RawImage[]) => Promise<{ pixel_values: Tensor }>)([dummyImage]);
    pixelValues = image_inputs.pixel_values;

    // Run inference to trigger shader compilation
    outputs = await vision_model({ pixel_values: pixelValues });

    console.log('Model preheated successfully');
  } catch (error) {
    console.warn('Model preheat failed:', getErrorMessage(error));
  } finally {
    disposeTensor(pixelValues);
    if (outputs && typeof outputs.dispose === 'function') {
      outputs.dispose();
    }
  }
}

const HUGGINGFACE_MODEL_ID = 'plhery/mobileclip2-onnx';
// Available model sizes: 's0' (43MB), 's2' (136MB), 'b' (330MB), 'l14' (1.1GB)
type ModelSize = 's0' | 's2' | 'b' | 'l14';

function isMobileDevice(): boolean {
  if (typeof navigator === 'undefined') return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function getModelSize(): ModelSize {
  // Use smaller S0 model (43MB) on mobile, S2 (136MB) on desktop
  return isMobileDevice() ? 's0' : 's2';
}

const MODEL_SIZE = getModelSize();

async function getModels(deviceOverride?: VisionDevice) {
  const device = deviceOverride ?? currentDevice ?? getDefaultDevice();
  const dtype = currentDtype ?? getDefaultDtype(device);

  if (!visionModelPromise || currentDevice !== device || currentDtype !== dtype) {
    currentDevice = device;
    currentDtype = dtype;

    console.log(`Loading model ${MODEL_SIZE} with device=${device}, dtype=${dtype}`);
    processorPromise = processorPromise || AutoProcessor.from_pretrained(HUGGINGFACE_MODEL_ID, {
      config_file_name: `${MODEL_SIZE}/preprocessor_config.json`,
    });
    visionModelPromise = CLIPVisionModelWithProjection.from_pretrained(HUGGINGFACE_MODEL_ID, {
      device,
      dtype,
      model_file_name: `${MODEL_SIZE}/vision_model`,
    });
  }

  const [processor, vision_model] = await Promise.all([processorPromise!, visionModelPromise!]);
  return { processor, vision_model, device };
}

export async function extractFeatures(photo: Photo): Promise<number[]> {
  // Step 1: Read image
  let image: RawImage;
  try {
    image = await RawImage.read(photo.nonHeicFile);
  } catch (error) {
    throw new Error(`Failed to read image: ${getErrorMessage(error)}`);
  }

  let pixelValues: Tensor | null = null;
  let outputs: any = null;
  let imageEmbeds: Tensor | null = null;
  let normalized: Tensor | null = null;

  try {
    // Step 2: Process image
    const { processor } = await getModels();
    const image_inputs = await (processor as unknown as (images: RawImage[]) => Promise<{ pixel_values: Tensor }>)([image]);
    pixelValues = image_inputs.pixel_values;

    // Step 3: Run inference (with WebGPU -> WASM fallback)
    try {
      const { vision_model } = await getModels();
      outputs = await vision_model({ pixel_values: pixelValues });
    } catch (error) {
      if (currentDevice === 'webgpu') {
        console.warn('WebGPU inference failed, falling back to WASM:', getErrorMessage(error));
        visionModelPromise = null;
        const { vision_model } = await getModels('wasm');
        outputs = await vision_model({ pixel_values: pixelValues });
      } else {
        throw new Error(`Inference failed: ${getErrorMessage(error)}`);
      }
    }

    imageEmbeds = (outputs.image_embeds ?? outputs.unnorm_image_features) as Tensor;
    normalized = imageEmbeds.normalize(2, -1);

    // Extract data before disposal
    const embedding = Array.from(normalized.data);

    return embedding;
  } finally {
    // Always cleanup tensors
    disposeTensors(pixelValues, imageEmbeds, normalized);
    // outputs may contain other tensors - dispose if it has dispose method
    if (outputs && typeof outputs.dispose === 'function') {
      outputs.dispose();
    }
  }
}

function createTensorFromArray(data: number[]): Tensor {
  return new Tensor('float32', Float32Array.from(data), [1, data.length]);
}

export async function prepareQualityEmbeddings(): Promise<PreparedQualityEmbeddings> {
  const data = qualityEmbedsData as Record<string, unknown>;

  // Check for version 2 format (paired dimensions with categories)
  if ('version' in data && (data.version as number) === 2) {
    const categoriesData = data.categories as Record<string, number[]>;
    const dimensionsData = data.dimensions as Array<{
      name: string;
      positive: number[];
      negative: number[];
      weight: number;
      categories: PhotoCategory[];
    }>;
    const calibration = data.calibration as {
      temperature: number;
      categoryThreshold: number;
    };

    // Convert category embeddings to tensors
    const categories = new Map<PhotoCategory, Tensor>();
    for (const [category, embedding] of Object.entries(categoriesData)) {
      categories.set(category as PhotoCategory, createTensorFromArray(embedding));
    }

    // Convert dimension embeddings to tensors
    const dimensions: QualityDimension[] = dimensionsData.map(dim => ({
      name: dim.name,
      positive: createTensorFromArray(dim.positive),
      negative: createTensorFromArray(dim.negative),
      weight: dim.weight,
      categories: dim.categories,
    }));

    return { version: 2, categories, dimensions, calibration };
  }

  // Legacy format compatibility - convert old format to new structure
  console.warn('Using legacy quality embeddings format. Run npm run generate-embeddings to upgrade.');

  // Create minimal compatible structure from old format
  const categories = new Map<PhotoCategory, Tensor>();
  const dimensions: QualityDimension[] = [];

  return {
    version: 1,
    categories,
    dimensions,
    calibration: {
      temperature: 10,
      categoryThreshold: 0.15,
    },
  };
}

/**
 * Compute cosine similarity between two tensors.
 */
async function cosineSimilarity(a: Tensor, b: Tensor): Promise<number> {
  let transposed: Tensor | null = null;
  let result: Tensor | null = null;

  try {
    transposed = b.transpose(1, 0);
    result = await matmul(a, transposed);
    return (result.data as Float32Array)[0];
  } finally {
    disposeTensors(transposed, result);
  }
}

/**
 * Detect photo categories using softmax over category similarities.
 * Returns a map of category -> confidence (0-1).
 */
async function detectCategories(
  imageEmbedding: Tensor,
  categoryEmbeddings: Map<PhotoCategory, Tensor>
): Promise<Map<PhotoCategory, number>> {
  const similarities: [PhotoCategory, number][] = [];

  for (const [category, embedding] of categoryEmbeddings) {
    const sim = await cosineSimilarity(imageEmbedding, embedding);
    similarities.push([category, sim]);
  }

  // Apply softmax with temperature to get probabilities
  const temperature = 5; // Lower = more peaked distribution
  const maxSim = Math.max(...similarities.map(([, s]) => s));
  const expScores = similarities.map(([cat, sim]) => [cat, Math.exp((sim - maxSim) * temperature)] as [PhotoCategory, number]);
  const sumExp = expScores.reduce((sum, [, exp]) => sum + exp, 0);

  const confidences = new Map<PhotoCategory, number>();
  for (const [cat, exp] of expScores) {
    confidences.set(cat, exp / sumExp);
  }

  return confidences;
}

/**
 * Calculate quality score using paired contrastive dimensions.
 * Each dimension is normalized via sigmoid(temperature * (pos - neg)) before weighting.
 */
async function calculateDimensionScore(
  imageEmbedding: Tensor,
  dimension: QualityDimension,
  temperature: number
): Promise<number> {
  const posSim = await cosineSimilarity(imageEmbedding, dimension.positive);
  const negSim = await cosineSimilarity(imageEmbedding, dimension.negative);

  // Sigmoid normalization: maps (pos - neg) to 0-1 range
  const diff = posSim - negSim;
  return 1 / (1 + Math.exp(-temperature * diff));
}

/* ---------- Quality + metadata analysis -------------------------------- */
export async function analyzeImage(
  photo: Photo,
  embedding: number[] | undefined,
  qualityEmbeddings: PreparedQualityEmbeddings | null
): Promise<{ quality: number; metadata: PhotoMetadata; hasFace?: boolean; detectedCategory?: PhotoCategory }> {

  let quality = 0;
  let hasFace = false;
  let detectedCategory: PhotoCategory = 'general';

  let imageEmbedding: Tensor | null = null;

  try {
    if (embedding && embedding.length > 0 && qualityEmbeddings && qualityEmbeddings.dimensions.length > 0) {
      try {
        imageEmbedding = new Tensor(
          'float32',
          Float32Array.from(embedding),
          [1, embedding.length]
        );

        const { categories, dimensions, calibration } = qualityEmbeddings;

        // Step 1: Detect photo categories
        const categoryConfidences = await detectCategories(imageEmbedding, categories);

        // Find the dominant category (excluding 'general')
        let maxConfidence = 0;
        for (const [cat, conf] of categoryConfidences) {
          if (cat !== 'general' && conf > maxConfidence) {
            maxConfidence = conf;
            detectedCategory = cat;
          }
        }

        // If no specific category is confident enough, use 'general'
        if (maxConfidence < calibration.categoryThreshold) {
          detectedCategory = 'general';
        }

        // Detect face presence from category
        hasFace = detectedCategory === 'face' || detectedCategory === 'group';

        // Step 2: Calculate quality score using applicable dimensions
        let weightedSum = 0;
        let totalWeight = 0;

        for (const dimension of dimensions) {
          // Check if this dimension applies to the detected category
          // Also always include dimensions for 'general' category
          const applies = dimension.categories.includes(detectedCategory) ||
            dimension.categories.includes('general');

          if (!applies) continue;

          // Calculate dimension-specific category weight
          // If dimension is specific to detected category, full weight
          // If dimension is general-only, reduce weight for specific categories
          let effectiveWeight = dimension.weight;
          if (!dimension.categories.includes(detectedCategory) && detectedCategory !== 'general') {
            effectiveWeight *= 0.5; // General dimensions get reduced weight for specific categories
          }

          const dimScore = await calculateDimensionScore(imageEmbedding, dimension, calibration.temperature);
          weightedSum += dimScore * effectiveWeight;
          totalWeight += effectiveWeight;
        }

        // Normalize to 0-100 range
        // dimScore is already 0-1 from sigmoid, so weighted average is 0-1
        const normalizedScore = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
        quality = Math.max(0, Math.min(100, Math.round(normalizedScore * 100)));

      } catch (error) {
        console.error("Error calculating image quality:", error);
        quality = 0;
      }
    } else if (!qualityEmbeddings || qualityEmbeddings.dimensions.length === 0) {
      console.warn("Quality embeddings not loaded or empty. Run 'npm run generate-embeddings' to generate them.");
      quality = 50; // Default to neutral score
    } else {
      console.warn("Embedding not provided, setting quality to 0.");
      quality = 0;
    }

    // EXIF reading - create separate scope for ArrayBuffer
    let metadata: PhotoMetadata;
    {
      const arrayBuffer = await photo.file.arrayBuffer();
      const exifTags = ExifReader.load(arrayBuffer);
      const dateFromExif = exifTags?.['DateTimeOriginal']?.description || exifTags?.['DateTime']?.description;
      metadata = {
        captureDate: new Date(dateFromExif?.replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3") || photo.file.lastModified),
      };
      // arrayBuffer goes out of scope here and can be GC'd
    }

    return { quality, metadata, hasFace, detectedCategory };
  } finally {
    disposeTensor(imageEmbedding);
  }
}

/* ---------- Union-Find data structure ---------------------------------- */
class UnionFind {
  private parent: number[];
  private rank: number[];

  constructor(size: number) {
    this.parent = Array.from({ length: size }, (_, i) => i);
    this.rank = new Array(size).fill(0);
  }

  find(x: number): number {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // Path compression
    }
    return this.parent[x];
  }

  union(x: number, y: number): void {
    const rootX = this.find(x);
    const rootY = this.find(y);
    if (rootX === rootY) return;

    // Union by rank
    if (this.rank[rootX] < this.rank[rootY]) {
      this.parent[rootX] = rootY;
    } else if (this.rank[rootX] > this.rank[rootY]) {
      this.parent[rootY] = rootX;
    } else {
      this.parent[rootY] = rootX;
      this.rank[rootX]++;
    }
  }

  getGroups(): Map<number, number[]> {
    const groups = new Map<number, number[]>();
    for (let i = 0; i < this.parent.length; i++) {
      const root = this.find(i);
      if (!groups.has(root)) {
        groups.set(root, []);
      }
      groups.get(root)!.push(i);
    }
    return groups;
  }
}

/**
 * Compute the effective similarity threshold based on temporal distance.
 * - If either date is missing, require higher similarity (+0.1)
 * - Photos > 2 hours apart are never grouped (returns Infinity)
 * - Photos within 1 minute get a lower threshold (-0.05)
 * - Photos 1-10 minutes apart use the base threshold
 * - Photos 10min-2h apart need higher similarity (+0.05)
 */
function getEffectiveThreshold(
  dateA: Date | undefined,
  dateB: Date | undefined,
  baseThreshold: number
): number {
  if (!dateA || !dateB) {
    return Math.min(1, baseThreshold + 0.1);
  }

  const diffMinutes = Math.abs(dateA.getTime() - dateB.getTime()) / 60000;

  if (diffMinutes > 120) return Infinity;
  if (diffMinutes <= 1) return Math.max(0, baseThreshold - 0.05);
  if (diffMinutes > 10) return Math.min(1, baseThreshold + 0.05);
  return baseThreshold;
}

export async function groupSimilarPhotos(
  photos: Photo[],
  similarityThreshold: number = 0.7
): Promise<{ groups: PhotoGroup[]; uniquePhotos: Photo[] }> {
  const photosWithEmbeddings = photos.filter(p => p.embedding && p.embedding.length > 0);
  const photosWithoutEmbeddings = photos.filter(p => !p.embedding || p.embedding.length === 0);

  if (photosWithEmbeddings.length < 2) {
    photos.sort((a, b) => {
      const dateA = a.metadata?.captureDate?.getTime() ?? 0;
      const dateB = b.metadata?.captureDate?.getTime() ?? 0;
      return dateB - dateA;
    });
    return { groups: [], uniquePhotos: photos };
  }

  const n = photosWithEmbeddings.length;
  const embeddingDim = photosWithEmbeddings[0].embedding!.length;

  let embeddingsTensor: Tensor | null = null;
  let transposed: Tensor | null = null;
  let similarityMatrix: Tensor | null = null;

  try {
    embeddingsTensor = new Tensor(
      'float32',
      Float32Array.from(photosWithEmbeddings.flatMap(p => p.embedding!)),
      [n, embeddingDim]
    );

    transposed = embeddingsTensor.transpose(1, 0);
    similarityMatrix = await matmul(embeddingsTensor, transposed);
    const similarities = similarityMatrix.data as Float32Array;

    // Use Union-Find for proper connected-component clustering
    const uf = new UnionFind(n);
    const pairSimilarities = new Map<string, number>();

    for (let i = 0; i < n; i++) {
      const photoA = photosWithEmbeddings[i];
      for (let j = i + 1; j < n; j++) {
        const photoB = photosWithEmbeddings[j];

        const threshold = getEffectiveThreshold(
          photoA.metadata?.captureDate,
          photoB.metadata?.captureDate,
          similarityThreshold
        );

        if (threshold === Infinity) continue;

        const similarity = similarities[i * n + j];
        if (similarity >= threshold) {
          uf.union(i, j);
          const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
          pairSimilarities.set(key, similarity);
        }
      }
    }

    // Extract groups from Union-Find
    const ufGroups = uf.getGroups();
    const groups: PhotoGroup[] = [];
    const uniquePhotos: Photo[] = [...photosWithoutEmbeddings];

    for (const indices of ufGroups.values()) {
      if (indices.length > 1) {
        const groupPhotos = indices.map(idx => photosWithEmbeddings[idx]);

        // Calculate minimum similarity within the group
        let minSimilarity = 1.0;
        for (let i = 0; i < indices.length; i++) {
          for (let j = i + 1; j < indices.length; j++) {
            const key = `${Math.min(indices[i], indices[j])}-${Math.max(indices[i], indices[j])}`;
            const sim = pairSimilarities.get(key);
            if (sim !== undefined) {
              minSimilarity = Math.min(minSimilarity, sim);
            }
          }
        }

        const sortedPhotos = [...groupPhotos].sort((a, b) => (b.quality ?? 0) - (a.quality ?? 0));
        const groupDate = sortedPhotos[0].metadata?.captureDate ?? new Date();
        groups.push({
          id: `${sortedPhotos[0].id}-group`,
          title: getGroupTitle(sortedPhotos[0]),
          date: groupDate,
          photos: sortedPhotos,
          similarity: minSimilarity,
          similarityThreshold,
        });
      } else {
        uniquePhotos.push(photosWithEmbeddings[indices[0]]);
      }
    }

    // Sort results by date
    groups.sort((a, b) => b.date.getTime() - a.date.getTime());
    uniquePhotos.sort((a, b) => {
      const dateA = a.metadata?.captureDate?.getTime() ?? 0;
      const dateB = b.metadata?.captureDate?.getTime() ?? 0;
      return dateB - dateA;
    });

    return { groups, uniquePhotos };
  } finally {
    // Critical: dispose O(n²) similarity matrix!
    disposeTensors(embeddingsTensor, transposed, similarityMatrix);
  }
}

/* ---------- Incremental grouping --------------------------------------- */
export async function addPhotoIncrementally(
  newPhoto: Photo,
  existingGroups: PhotoGroup[],
  existingUniquePhotos: Photo[],
  similarityThreshold: number = 0.7
): Promise<{ groups: PhotoGroup[]; uniquePhotos: Photo[]; needsFullRegroup: boolean }> {
  if (!newPhoto.embedding || newPhoto.embedding.length === 0) {
    const uniquePhotos = [...existingUniquePhotos, newPhoto].sort((a, b) => {
      const dateA = a.metadata?.captureDate?.getTime() ?? 0;
      const dateB = b.metadata?.captureDate?.getTime() ?? 0;
      return dateB - dateA;
    });
    return { groups: existingGroups, uniquePhotos, needsFullRegroup: false };
  }

  const newEmbedding = new Float32Array(newPhoto.embedding);
  const newDate = newPhoto.metadata?.captureDate;

  const computeSimilarity = (embedding: number[]): number => {
    let sum = 0;
    for (let i = 0; i < embedding.length; i++) {
      sum += newEmbedding[i] * embedding[i];
    }
    return sum;
  };

  const getThreshold = (otherDate?: Date): number => {
    return getEffectiveThreshold(newDate, otherDate, similarityThreshold);
  };

  const matchingGroupIndices: number[] = [];
  const matchingUniqueIndices: number[] = [];

  for (let i = 0; i < existingGroups.length; i++) {
    const group = existingGroups[i];
    const representative = group.photos[0];
    if (!representative.embedding) continue;

    const threshold = getThreshold(representative.metadata?.captureDate);
    if (threshold === Infinity) continue;

    const similarity = computeSimilarity(representative.embedding);
    if (similarity >= threshold) {
      matchingGroupIndices.push(i);
    }
  }

  for (let i = 0; i < existingUniquePhotos.length; i++) {
    const photo = existingUniquePhotos[i];
    if (!photo.embedding) continue;

    const threshold = getThreshold(photo.metadata?.captureDate);
    if (threshold === Infinity) continue;

    const similarity = computeSimilarity(photo.embedding);
    if (similarity >= threshold) {
      matchingUniqueIndices.push(i);
    }
  }

  const totalMatches = matchingGroupIndices.length + matchingUniqueIndices.length;

  if (totalMatches === 0) {
    const uniquePhotos = [...existingUniquePhotos, newPhoto].sort((a, b) => {
      const dateA = a.metadata?.captureDate?.getTime() ?? 0;
      const dateB = b.metadata?.captureDate?.getTime() ?? 0;
      return dateB - dateA;
    });
    return { groups: existingGroups, uniquePhotos, needsFullRegroup: false };
  }

  if (matchingGroupIndices.length > 1 || (matchingGroupIndices.length >= 1 && matchingUniqueIndices.length >= 1)) {
    return { groups: existingGroups, uniquePhotos: existingUniquePhotos, needsFullRegroup: true };
  }

  if (matchingGroupIndices.length === 1 && matchingUniqueIndices.length === 0) {
    const groupIdx = matchingGroupIndices[0];
    const groups = [...existingGroups];
    const group = { ...groups[groupIdx] };

    const representative = group.photos[0];
    const similarity = computeSimilarity(representative.embedding!);

    const photos = [...group.photos, newPhoto].sort((a, b) => (b.quality ?? 0) - (a.quality ?? 0));
    group.photos = photos;
    group.similarity = Math.min(group.similarity, similarity);

    if (photos[0].id === newPhoto.id) {
      group.title = getGroupTitle(newPhoto);
      group.date = newPhoto.metadata?.captureDate ?? group.date;
      group.id = `${newPhoto.id}-group`;
    }
    groups[groupIdx] = group;

    groups.sort((a, b) => b.date.getTime() - a.date.getTime());

    return { groups, uniquePhotos: existingUniquePhotos, needsFullRegroup: false };
  }

  if (matchingUniqueIndices.length >= 1 && matchingGroupIndices.length === 0) {
    const matchedPhotos = matchingUniqueIndices.map(i => existingUniquePhotos[i]);
    const allGroupPhotos = [newPhoto, ...matchedPhotos];

    let minSimilarity = 1.0;
    for (const photo of matchedPhotos) {
      if (photo.embedding) {
        minSimilarity = Math.min(minSimilarity, computeSimilarity(photo.embedding));
      }
    }

    const sortedPhotos = allGroupPhotos.sort((a, b) => (b.quality ?? 0) - (a.quality ?? 0));
    const bestPhoto = sortedPhotos[0];

    const newGroup: PhotoGroup = {
      id: `${bestPhoto.id}-group`,
      title: getGroupTitle(bestPhoto),
      date: bestPhoto.metadata?.captureDate ?? new Date(),
      photos: sortedPhotos,
      similarity: minSimilarity,
      similarityThreshold,
    };

    const remainingUnique = existingUniquePhotos.filter((_, i) => !matchingUniqueIndices.includes(i));
    const groups = [...existingGroups, newGroup].sort((a, b) => b.date.getTime() - a.date.getTime());

    return { groups, uniquePhotos: remainingUnique, needsFullRegroup: false };
  }

  return { groups: existingGroups, uniquePhotos: existingUniquePhotos, needsFullRegroup: true };
}

/* ---------- Utility helpers -------------------------------------------- */
function getGroupTitle(photo: Photo): string {
  const date = photo.metadata?.captureDate ?? new Date();
  const formattedDate = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
  const formattedTime = date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
  return `${formattedDate} at ${formattedTime}`;
}
