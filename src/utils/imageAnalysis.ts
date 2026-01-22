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

export async function prepareQualityEmbeddings(): Promise<{
  positiveEmbeddings: Tensor;
  negativeEmbeddings: Tensor;
}> {
  // Load embeddings from the JSON file instead of generating them
  const positiveEmbeddingsData = qualityEmbedsData.positive;
  const negativeEmbeddingsData = qualityEmbedsData.negative;

  // Get dimensions - positive and negative embedding arrays have the same vector length
  const nPositive = positiveEmbeddingsData.length;
  const nNegative = negativeEmbeddingsData.length;
  const embeddingSize = positiveEmbeddingsData[0].length;

  // Convert to tensors
  const positiveEmbeddings = new Tensor(
    'float32',
    Float32Array.from(positiveEmbeddingsData.flat()),
    [nPositive, embeddingSize]
  );

  const negativeEmbeddings = new Tensor(
    'float32',
    Float32Array.from(negativeEmbeddingsData.flat()),
    [nNegative, embeddingSize]
  );

  return { positiveEmbeddings, negativeEmbeddings };
}

/* ---------- Quality + metadata analysis -------------------------------- */
export async function analyzeImage(
  photo: Photo,
  embedding: number[] | undefined,
  qualityEmbeddings: { positiveEmbeddings: Tensor; negativeEmbeddings: Tensor } | null
): Promise<{ quality: number; metadata: PhotoMetadata }> {

  let quality = 0;

  if (embedding && embedding.length > 0 && qualityEmbeddings) {
    try {
      const v = new Tensor(
        'float32',
        Float32Array.from(embedding),
        [1, embedding.length]
      );

      const { positiveEmbeddings: P, negativeEmbeddings: N } = qualityEmbeddings;

      const simPos = await matmul(v, P.transpose(1, 0)); // [1, k_pos]
      const simNeg = await matmul(v, N.transpose(1, 0)); // [1, k_neg]

      const avgPos = simPos.sum().div(P.dims[0]).item() as number;
      const avgNeg = simNeg.sum().div(N.dims[0]).item() as number;

      // simple linear calibration (placeholder)
      const rawScore = avgPos - avgNeg;
      quality = Math.max(0, Math.min(100, Math.round(((rawScore * 15 + 1) / 2) * 100)));

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
  return { quality, metadata };
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
 * - Photos <= 1 minute apart get relaxed threshold (-0.05)
 * - Photos > 10 minutes apart get stricter threshold (+0.05)
 */
function getEffectiveThreshold(
  dateA: Date | undefined,
  dateB: Date | undefined,
  baseThreshold: number
): number {
  // If dates are missing, require higher similarity but don't skip
  if (!dateA || !dateB) {
    return Math.min(1, baseThreshold + 0.1);
  }

  const diffMinutes = Math.abs(dateA.getTime() - dateB.getTime()) / 60000;

  // Photos > 2 hours apart are never grouped
  if (diffMinutes > 120) return Infinity;

  // Adjust threshold based on temporal proximity
  if (diffMinutes <= 1) {
    return Math.max(0, baseThreshold - 0.05);
  } else if (diffMinutes > 10) {
    return Math.min(1, baseThreshold + 0.05);
  }

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

  // Use Union-Find for transitive clustering
  const uf = new UnionFind(n);
  const pairSimilarities = new Map<string, number>(); // Track similarities for group stats

  for (let i = 0; i < n; i++) {
    const photoA = photosWithEmbeddings[i];
    const dateA = photoA.metadata?.captureDate;

    for (let j = i + 1; j < n; j++) {
      const photoB = photosWithEmbeddings[j];
      const dateB = photoB.metadata?.captureDate;

      // Get effective threshold based on temporal distance
      const threshold = getEffectiveThreshold(dateA, dateB, similarityThreshold);

      // Skip if threshold is impossible to meet (e.g., photos > 2h apart with dates)
      if (threshold === Infinity) continue;

      const similarity = similarities[i * n + j];

      if (similarity >= threshold) {
        uf.union(i, j);
        // Store similarity for later group stats
        const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
        pairSimilarities.set(key, similarity);
      }
    }
  }

  // Build groups from Union-Find clusters
  const clusters = uf.getGroups();
  const groups: PhotoGroup[] = [];
  const uniquePhotos: Photo[] = [...photosWithoutEmbeddings];

  for (const [, indices] of clusters) {
    if (indices.length === 1) {
      // Single photo = unique
      uniquePhotos.push(photosWithEmbeddings[indices[0]]);
    } else {
      // Multiple photos = group
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

      // Sort by quality (best first)
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
    }
  }

  // Sort groups and unique photos by date
  groups.sort((a, b) => b.date.getTime() - a.date.getTime());
  uniquePhotos.sort((a, b) => {
    const dateA = a.metadata?.captureDate?.getTime() ?? 0;
    const dateB = b.metadata?.captureDate?.getTime() ?? 0;
    return dateB - dateA;
  });

  return { groups, uniquePhotos };
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
