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