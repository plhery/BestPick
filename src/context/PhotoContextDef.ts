import { createContext } from 'react';
import { AppState } from '../types';

export type ProcessingStep = 'idle' | 'loading-model' | 'converting' | 'extracting' | 'scoring' | 'grouping';

export interface ProcessingProgress {
    currentIndex: number;
    totalCount: number;
    currentStep: ProcessingStep;
    currentFileName: string;
}

export interface PhotoContextType {
    state: AppState;
    isLoading: boolean;
    isPreparingEmbeddings: boolean;
    processingProgress: ProcessingProgress | null;
    addPhotos: (files: File[]) => void;
    toggleSelectPhoto: (photoId: string) => void;
    selectAllInGroup: (groupId: string) => void;
    deselectAllInGroup: (groupId: string) => void;
    selectAll: () => void;
    deselectAll: () => void;
    undo: () => void;
    redo: () => void;
    downloadSelected: () => void;
    isSelected: (id: string) => boolean;
    similarityThreshold: number;
    setSimilarityThreshold: (threshold: number) => void;
}

export const PhotoContext = createContext<PhotoContextType | undefined>(undefined);
