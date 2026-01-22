import { createContext } from 'react';
import { AppState } from '../types';

export interface PhotoContextType {
    state: AppState;
    isLoading: boolean;
    isPreparingEmbeddings: boolean;
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
}

export const PhotoContext = createContext<PhotoContextType | undefined>(undefined);
