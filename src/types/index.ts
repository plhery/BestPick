export type PhotoCategory = 'general' | 'face' | 'group' | 'food' | 'landscape' | 'screenshot' | 'drawing' | 'pet' | 'document' | 'night';

export interface DimensionScore {
  name: string;           // e.g., "sharpness", "face_expression"
  score: number;          // 0-1 normalized score
  weight: number;         // effective weight used
}

export interface QualityBreakdown {
  detectedCategory: PhotoCategory;
  categoryConfidences: Record<PhotoCategory, number>;
  dimensions: DimensionScore[];
}

export interface Photo {
  id: string;
  file?: File;
  nonHeicFile?: File;
  url: string;
  thumbnailUrl: string;
  name: string;
  size: number;
  type: string;
  dateCreated: Date;
  quality?: number; // 0-100 quality score
  qualityBreakdown?: QualityBreakdown;
  selected: boolean;
  metadata?: PhotoMetadata;
  embedding?: number[]; // Added for CLIP features
}

export interface PhotoMetadata {
  captureDate?: Date;
  camera?: string;
  location?: string;
}

export interface PhotoGroup {
  id: string;
  title: string;
  date: Date;
  photos: Photo[];
  similarity: number;
  similarityThreshold: number;
}

export interface AppState {
  photos: Photo[];
  groups: PhotoGroup[];
  selectedPhotos: string[];
  uniquePhotos: Photo[];
  history: HistoryState[];
  currentHistoryIndex: number;
}

export interface HistoryState {
  selectedPhotos: string[];
  timestamp: Date;
}