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

/* ---------- Union-Find for clustering ---------------------------------- */
class UnionFind {
  private parent: number[];
  private rank: number[];

  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x: number): number {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // Path compression
    }
    return this.parent[x];
  }

  union(x: number, y: number): void {
    const px = this.find(x);
    const py = this.find(y);
    if (px === py) return;
    // Union by rank
    if (this.rank[px] < this.rank[py]) {
      this.parent[px] = py;
    } else if (this.rank[px] > this.rank[py]) {
      this.parent[py] = px;
    } else {
      this.parent[py] = px;
      this.rank[px]++;
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
  const pairSimilarities = new Map<string, number>(); // Track similarities for group stats

  for (let i = 0; i < n; i++) {
    const photoA = photosWithEmbeddings[i];
    for (let j = i + 1; j < n; j++) {
      const photoB = photosWithEmbeddings[j];

      // Temporal filtering with fallback for missing dates
      const dateA = photoA.metadata?.captureDate;
      const dateB = photoB.metadata?.captureDate;

      let threshold: number;
      if (!dateA || !dateB) {
        // No dates available: use higher threshold (stricter) instead of skipping
        threshold = Math.min(1, similarityThreshold + 0.15);
      } else {
        const diffMinutes = Math.abs(dateA.getTime() - dateB.getTime()) / 60000;
        if (diffMinutes > 120) continue; // > 2h => never group

        threshold = (diffMinutes <= 1)
          ? Math.max(0, similarityThreshold - 0.05)
          : (diffMinutes > 10)
            ? Math.min(1, similarityThreshold + 0.05)
            : similarityThreshold;
      }

      const similarity = similarities[i * n + j];
      if (similarity >= threshold) {
        uf.union(i, j);
        // Track this pair's similarity
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
/**
 * Add a single photo to existing groups/uniquePhotos without full recalculation.
 * This is O(n) instead of O(n²) for adding one photo at a time.
 */
export async function addPhotoIncrementally(
  newPhoto: Photo,
  existingGroups: PhotoGroup[],
  existingUniquePhotos: Photo[],
  similarityThreshold: number = 0.7
): Promise<{ groups: PhotoGroup[]; uniquePhotos: Photo[]; needsFullRegroup: boolean }> {
  // If new photo has no embedding, add to unique and return
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

  // Helper: compute cosine similarity (embeddings are already normalized)
  const computeSimilarity = (embedding: number[]): number => {
    let sum = 0;
    for (let i = 0; i < embedding.length; i++) {
      sum += newEmbedding[i] * embedding[i];
    }
    return sum;
  };

  // Helper: get effective threshold based on date difference
  const getThreshold = (otherDate?: Date): number => {
    if (!newDate || !otherDate) {
      return Math.min(1, similarityThreshold + 0.15); // Higher threshold for missing dates
    }
    const diffMinutes = Math.abs(newDate.getTime() - otherDate.getTime()) / 60000;
    if (diffMinutes > 120) return Infinity; // Never match
    if (diffMinutes <= 1) return Math.max(0, similarityThreshold - 0.05);
    if (diffMinutes > 10) return Math.min(1, similarityThreshold + 0.05);
    return similarityThreshold;
  };

  // Track which groups and unique photos match
  const matchingGroupIndices: number[] = [];
  const matchingUniqueIndices: number[] = [];

  // Check against existing groups (compare with best photo in each group)
  for (let i = 0; i < existingGroups.length; i++) {
    const group = existingGroups[i];
    // Check against the first (best quality) photo in the group
    const representative = group.photos[0];
    if (!representative.embedding) continue;

    const threshold = getThreshold(representative.metadata?.captureDate);
    if (threshold === Infinity) continue;

    const similarity = computeSimilarity(representative.embedding);
    if (similarity >= threshold) {
      matchingGroupIndices.push(i);
    }
  }

  // Check against unique photos
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

  // Decision logic
  const totalMatches = matchingGroupIndices.length + matchingUniqueIndices.length;

  if (totalMatches === 0) {
    // No matches: add as unique photo
    const uniquePhotos = [...existingUniquePhotos, newPhoto].sort((a, b) => {
      const dateA = a.metadata?.captureDate?.getTime() ?? 0;
      const dateB = b.metadata?.captureDate?.getTime() ?? 0;
      return dateB - dateA;
    });
    return { groups: existingGroups, uniquePhotos, needsFullRegroup: false };
  }

  if (matchingGroupIndices.length > 1 || (matchingGroupIndices.length >= 1 && matchingUniqueIndices.length >= 1)) {
    // Multiple matches across groups/unique photos: need full regroup to merge
    return { groups: existingGroups, uniquePhotos: existingUniquePhotos, needsFullRegroup: true };
  }

  if (matchingGroupIndices.length === 1 && matchingUniqueIndices.length === 0) {
    // Matches exactly one group: add to that group
    const groupIdx = matchingGroupIndices[0];
    const groups = [...existingGroups];
    const group = { ...groups[groupIdx] };

    // Calculate similarity to the representative
    const representative = group.photos[0];
    const similarity = computeSimilarity(representative.embedding!);

    // Add photo and re-sort by quality
    const photos = [...group.photos, newPhoto].sort((a, b) => (b.quality ?? 0) - (a.quality ?? 0));
    group.photos = photos;
    group.similarity = Math.min(group.similarity, similarity);
    // Update title and date if new photo becomes the best
    if (photos[0].id === newPhoto.id) {
      group.title = getGroupTitle(newPhoto);
      group.date = newPhoto.metadata?.captureDate ?? group.date;
      group.id = `${newPhoto.id}-group`;
    }
    groups[groupIdx] = group;

    // Re-sort groups by date
    groups.sort((a, b) => b.date.getTime() - a.date.getTime());

    return { groups, uniquePhotos: existingUniquePhotos, needsFullRegroup: false };
  }

  if (matchingUniqueIndices.length >= 1 && matchingGroupIndices.length === 0) {
    // Matches one or more unique photos: create new group
    const matchedPhotos = matchingUniqueIndices.map(i => existingUniquePhotos[i]);
    const allGroupPhotos = [newPhoto, ...matchedPhotos];

    // Calculate minimum similarity
    let minSimilarity = 1.0;
    for (const photo of matchedPhotos) {
      if (photo.embedding) {
        minSimilarity = Math.min(minSimilarity, computeSimilarity(photo.embedding));
      }
    }

    // Sort by quality
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

    // Remove matched photos from unique
    const remainingUnique = existingUniquePhotos.filter((_, i) => !matchingUniqueIndices.includes(i));

    // Add new group and sort
    const groups = [...existingGroups, newGroup].sort((a, b) => b.date.getTime() - a.date.getTime());

    return { groups, uniquePhotos: remainingUnique, needsFullRegroup: false };
  }

  // Fallback: full regroup needed
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
