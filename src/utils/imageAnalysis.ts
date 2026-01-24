import { Photo, PhotoGroup, PhotoMetadata } from '../types';
import {
  AutoProcessor,
  AutoTokenizer,
  CLIPVisionModelWithProjection,
  RawImage,
  Tensor,
  matmul,
} from '@huggingface/transformers';
import ExifReader from 'exifreader';
import qualityEmbedsData from '../data/qualityEmbeds.json';

// Type for the new structured quality embeddings
interface QualityEmbeddingsSet {
  embeddings: Tensor;
  weights: Float32Array;
}

interface PreparedQualityEmbeddings {
  general: {
    positive: QualityEmbeddingsSet;
    negative: QualityEmbeddingsSet;
  };
  face: {
    positive: QualityEmbeddingsSet;
    negative: QualityEmbeddingsSet;
  };
  calibration: {
    slope: number;
    offset: number;
    faceWeight: number;
  };
}

/* ---------- Model lazy‑loading ----------------------------------------- */
let visionModelPromise: Promise<InstanceType<typeof CLIPVisionModelWithProjection>> | null = null;
let processorPromise: Promise<InstanceType<typeof AutoProcessor>> | null = null;
let tokenizerPromise: Promise<InstanceType<typeof AutoTokenizer>> | null = null;

async function getModels() {
  if (!visionModelPromise) {
    const modelId = 'jinaai/jina-clip-v1';
    const processorId = 'xenova/clip-vit-base-patch32';

    processorPromise = processorPromise || AutoProcessor.from_pretrained(processorId, { device: 'auto' });
    visionModelPromise = visionModelPromise || CLIPVisionModelWithProjection.from_pretrained(modelId, { device: 'auto', dtype: 'fp32' });
    tokenizerPromise = tokenizerPromise || AutoTokenizer.from_pretrained(modelId);
  }
  const [processor, vision_model, tokenizer] = await Promise.all([
    processorPromise!,
    visionModelPromise!,
    tokenizerPromise!,
  ]);
  return { processor, vision_model, tokenizer };
}

export async function extractFeatures(photo: Photo): Promise<number[]> {
  const { processor, vision_model } = await getModels();
  const image = await RawImage.read(photo.nonHeicFile);
  const image_inputs = await (processor as { (images: RawImage[]): Promise<Record<string, unknown>> })([image]);
  const { image_embeds } = await vision_model(image_inputs);
  return Array.from(image_embeds.normalize(2, -1).data);
}

function createEmbeddingsSet(
  embeddingsData: number[][],
  weightsData: number[]
): QualityEmbeddingsSet {
  const n = embeddingsData.length;
  const embeddingSize = embeddingsData[0].length;

  const embeddings = new Tensor(
    'float32',
    Float32Array.from(embeddingsData.flat()),
    [n, embeddingSize]
  );

  const weights = Float32Array.from(weightsData);

  return { embeddings, weights };
}

export async function prepareQualityEmbeddings(): Promise<PreparedQualityEmbeddings> {
  // Handle both old format (for backwards compatibility) and new format
  const data = qualityEmbedsData as Record<string, unknown>;

  // Check if it's the new format with 'general' and 'face' keys
  if ('general' in data && 'face' in data) {
    const generalData = data.general as {
      positive: number[][];
      positiveWeights: number[];
      negative: number[][];
      negativeWeights: number[];
    };
    const faceData = data.face as {
      positive: number[][];
      positiveWeights: number[];
      negative: number[][];
      negativeWeights: number[];
    };
    const calibration = data.calibration as {
      slope: number;
      offset: number;
      faceWeight: number;
    };

    return {
      general: {
        positive: createEmbeddingsSet(generalData.positive, generalData.positiveWeights),
        negative: createEmbeddingsSet(generalData.negative, generalData.negativeWeights),
      },
      face: {
        positive: createEmbeddingsSet(faceData.positive, faceData.positiveWeights),
        negative: createEmbeddingsSet(faceData.negative, faceData.negativeWeights),
      },
      calibration,
    };
  }

  // Old format compatibility - treat all as general with equal weights
  const positiveEmbeddingsData = (data as { positive: number[][] }).positive;
  const negativeEmbeddingsData = (data as { negative: number[][] }).negative;

  const equalPosWeights = new Array(positiveEmbeddingsData.length).fill(1.0);
  const equalNegWeights = new Array(negativeEmbeddingsData.length).fill(1.0);

  const generalSet = {
    positive: createEmbeddingsSet(positiveEmbeddingsData, equalPosWeights),
    negative: createEmbeddingsSet(negativeEmbeddingsData, equalNegWeights),
  };

  return {
    general: generalSet,
    face: generalSet, // Use same embeddings for face in old format
    calibration: {
      slope: 15,
      offset: 1,
      faceWeight: 0.5,
    },
  };
}

/**
 * Calculate weighted average similarity score.
 */
async function calculateWeightedSimilarity(
  imageEmbedding: Tensor,
  embeddingsSet: QualityEmbeddingsSet
): Promise<number> {
  const { embeddings, weights } = embeddingsSet;

  // Compute similarities: [1, n_prompts]
  const similarities = await matmul(imageEmbedding, embeddings.transpose(1, 0));
  const simData = similarities.data as Float32Array;

  // Weighted average
  let weightedSum = 0;
  let totalWeight = 0;
  for (let i = 0; i < weights.length; i++) {
    weightedSum += simData[i] * weights[i];
    totalWeight += weights[i];
  }

  return weightedSum / totalWeight;
}

/* ---------- Quality + metadata analysis -------------------------------- */
export async function analyzeImage(
  photo: Photo,
  embedding: number[] | undefined,
  qualityEmbeddings: PreparedQualityEmbeddings | null
): Promise<{ quality: number; metadata: PhotoMetadata; hasFace?: boolean }> {

  let quality = 0;
  let hasFace = false;

  if (embedding && embedding.length > 0 && qualityEmbeddings) {
    try {
      const v = new Tensor(
        'float32',
        Float32Array.from(embedding),
        [1, embedding.length]
      );

      const { general, face, calibration } = qualityEmbeddings;

      // Calculate general quality score
      const generalPosScore = await calculateWeightedSimilarity(v, general.positive);
      const generalNegScore = await calculateWeightedSimilarity(v, general.negative);
      const generalRawScore = generalPosScore - generalNegScore;

      // Calculate face quality score
      const facePosScore = await calculateWeightedSimilarity(v, face.positive);
      const faceNegScore = await calculateWeightedSimilarity(v, face.negative);
      const faceRawScore = facePosScore - faceNegScore;

      // Heuristic face confidence from similarity margin
      const faceMargin = facePosScore - faceNegScore;
      const faceConfidence = 1 / (1 + Math.exp(-faceMargin * 8));
      const clampedFaceConfidence = Math.min(1, Math.max(0, faceConfidence));
      hasFace = clampedFaceConfidence > 0.4;

      // Blend scores based on face presence
      // If face detected, give more weight to face-specific quality
      const blendWeight = clampedFaceConfidence * calibration.faceWeight;
      const rawScore = generalRawScore * (1 - blendWeight) + faceRawScore * blendWeight;

      // Apply calibration
      // Maps raw score (typically -0.2 to 0.2) to 0-100 range
      const normalizedScore = (rawScore * calibration.slope + calibration.offset);
      quality = Math.max(0, Math.min(100, Math.round(normalizedScore * 100)));

    } catch (error) {
      console.error("Error calculating image quality:", error);
      quality = 0;
    }
  } else {
    console.warn("Embedding or quality embeddings not provided, setting quality to 0.");
    quality = 0;
  }

  const arrayBuffer = await photo.file.arrayBuffer();
  const exifTags = ExifReader.load(arrayBuffer);

  const dateFromExif = exifTags?.['DateTimeOriginal']?.description || exifTags?.['DateTime']?.description

  const metadata: PhotoMetadata = {
    captureDate: new Date(dateFromExif?.replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3") || photo.file.lastModified),
  };
  return { quality, metadata, hasFace };
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
  const embeddingsTensor = new Tensor(
    'float32',
    Float32Array.from(photosWithEmbeddings.flatMap(p => p.embedding!)),
    [n, embeddingDim]
  );
  const similarityMatrix = await matmul(embeddingsTensor, embeddingsTensor.transpose(1, 0));
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
