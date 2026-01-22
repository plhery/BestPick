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
  const similarities = await similarityMatrix.data as Float32Array;

  const groups: PhotoGroup[] = [];
  const uniquePhotos: Photo[] = [...photosWithoutEmbeddings];
  const processed = new Set<string>();

  for (let i = 0; i < n; i++) {
    const photoA = photosWithEmbeddings[i];
    if (processed.has(photoA.id)) continue;

    const currentGroupIndices: number[] = [i];
    let minSimilarityInGroup = 1.0;
    processed.add(photoA.id);

    for (let j = i + 1; j < n; j++) {
      const photoB = photosWithEmbeddings[j];
      if (processed.has(photoB.id)) continue;

      // Temporal filtering --------------------------------------------------
      const dateA = photoA.metadata?.captureDate;
      const dateB = photoB.metadata?.captureDate;
      // Skip temporal filtering if dates are missing
      if (!dateA || !dateB) continue;
      const diffMinutes = Math.abs(dateA.getTime() - dateB.getTime()) / 60000;
      if (diffMinutes > 120) continue; // > 2 h => never group

      const threshold = (diffMinutes <= 1) ?
        Math.max(0, similarityThreshold - 0.05) :
        ((diffMinutes > 10) ? Math.min(1, similarityThreshold + 0.05) : similarityThreshold);
      const similarity = similarities[i * n + j];

      if (similarity >= threshold) {
        currentGroupIndices.push(j);
        minSimilarityInGroup = Math.min(minSimilarityInGroup, similarity);
        processed.add(photoB.id);
      }
    }

    const currentGroupPhotos = currentGroupIndices.map(index => photosWithEmbeddings[index]);

    if (currentGroupPhotos.length > 1) {
      const sortedPhotos = [...currentGroupPhotos].sort((a, b) => (b.quality ?? 0) - (a.quality ?? 0));
      const groupDate = sortedPhotos[0].metadata?.captureDate ?? new Date();
      groups.push({
        id: `${sortedPhotos[0].id}-group`,
        title: getGroupTitle(sortedPhotos[0]),
        date: groupDate,
        photos: sortedPhotos,
        similarity: minSimilarityInGroup,
        similarityThreshold,
      });
    } else {
      uniquePhotos.push(photoA);
    }
  }

  // Only need to sort uniquePhotos once at the end
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
