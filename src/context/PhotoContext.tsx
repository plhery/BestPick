import React, { useReducer, useState, useEffect, useRef } from 'react';
import { PhotoContext, ProcessingProgress, ProcessingStep } from './PhotoContextDef';
import { AppState, Photo, PhotoGroup, PhotoMetadata } from '../types';
import { analyzeImage, groupSimilarPhotos, extractFeatures, prepareQualityEmbeddings } from '../utils/imageAnalysis';
import { Tensor } from '@huggingface/transformers';
import { isHeic, heicTo } from 'heic-to';

// Define a type for the quality embeddings
type QualityEmbeddings = {
  positiveEmbeddings: Tensor;
  negativeEmbeddings: Tensor;
} | null;

type PhotoAction =
  | { type: 'ADD_PHOTO_AND_UPDATE_GROUPS'; photo: Photo; groups: PhotoGroup[]; uniquePhotos: Photo[] }
  | { type: 'TOGGLE_SELECT_PHOTO'; photoId: string }
  | { type: 'SELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'DESELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'SELECT_ALL' }
  | { type: 'DESELECT_ALL' }
  | { type: 'UNDO' }
  | { type: 'REDO' };

const initialState: AppState = {
  photos: [],
  groups: [],
  selectedPhotos: [],
  uniquePhotos: [],
  history: [],
  currentHistoryIndex: -1,
};

function reducer(state: AppState, action: PhotoAction): AppState {
  let newState: AppState;

  switch (action.type) {
    case 'ADD_PHOTO_AND_UPDATE_GROUPS': {
      const { photo, groups, uniquePhotos } = action;

      // Add the new photo and update groups/unique photos
      const updatedPhotos = [...state.photos, photo];
      const updatedStateBase: Omit<AppState, 'history' | 'currentHistoryIndex' | 'selectedPhotos'> = {
        ...state,
        photos: updatedPhotos,
        groups,
        uniquePhotos,
      };

      // Auto-select the best photo in each group and all unique photos
      const autoSelectedPhotos = [
        ...uniquePhotos.map(p => p.id),
        ...groups.map(g => g.photos[0].id) // Assumes groups photos are sorted by quality desc
      ];

      // Ensure the newly added photo is selected if it's unique or best in its group
      if (!autoSelectedPhotos.includes(photo.id)) {
        const isUnique = uniquePhotos.some(p => p.id === photo.id);
        // Select if unique, or if it ended up as the best in a new/updated group
        if (isUnique || groups.find(g => g.photos[0].id === photo.id)) {
          // This logic might need refinement based on exact desired UX for new photos
          // For now, let's stick to selecting best-in-group and unique
        }
        // If the photo isn't selected automatically by the rules above,
        // we don't add it here, respecting the auto-selection logic based on groups/unique.
      }


      // Create the final state before history update
      const intermediateState: AppState = {
        ...updatedStateBase,
        selectedPhotos: autoSelectedPhotos,
        history: state.history,
        currentHistoryIndex: state.currentHistoryIndex,
      };

      // Update the 'selected' flag on all photos based on the new selection
      intermediateState.photos = intermediateState.photos.map(p => ({
        ...p,
        selected: autoSelectedPhotos.includes(p.id)
      }));

      // --- History Update ---
      const MAX_HISTORY_SIZE = 50;
      let history = state.history;
      let currentHistoryIndex = state.currentHistoryIndex;

      // Always add a new history state when a photo is processed and groups potentially change
      const newHistoryEntry = { selectedPhotos: intermediateState.selectedPhotos, timestamp: new Date() };

      // If this is the very first photo added
      if (state.photos.length === 0) {
        history = [newHistoryEntry];
        currentHistoryIndex = 0;
      } else {
        // Append to history, discarding any future states if we were in an undone state
        history = [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          newHistoryEntry
        ];
        // Cap history size to prevent memory issues
        if (history.length > MAX_HISTORY_SIZE) {
          history = history.slice(history.length - MAX_HISTORY_SIZE);
          currentHistoryIndex = MAX_HISTORY_SIZE - 1;
        } else {
          currentHistoryIndex = history.length - 1;
        }
      }
      // --- End History Update ---

      // Final state with updated history
      const finalState: AppState = {
        ...intermediateState,
        history,
        currentHistoryIndex,
      };

      return finalState;
    }

    case 'TOGGLE_SELECT_PHOTO': {
      const selectedIndex = state.selectedPhotos.indexOf(action.photoId);
      const newSelectedPhotos = [...state.selectedPhotos];

      if (selectedIndex === -1) {
        newSelectedPhotos.push(action.photoId);
      } else {
        newSelectedPhotos.splice(selectedIndex, 1);
      }

      newState = {
        ...state,
        selectedPhotos: newSelectedPhotos,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: newSelectedPhotos, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };

      newState.photos = state.photos.map(photo =>
        photo.id === action.photoId
          ? { ...photo, selected: !photo.selected }
          : photo
      );

      return newState;
    }

    case 'SELECT_ALL_IN_GROUP': {
      const group = state.groups.find(g => g.id === action.groupId);
      if (!group) return state;

      const photoIds = group.photos.map(photo => photo.id);
      const newSelectedPhotosSet = new Set([...state.selectedPhotos, ...photoIds]);
      const newSelectedPhotos = Array.from(newSelectedPhotosSet);

      if (newSelectedPhotos.length === state.selectedPhotos.length && newSelectedPhotos.every(id => state.selectedPhotos.includes(id))) {
        return state;
      }

      newState = {
        ...state,
        selectedPhotos: newSelectedPhotos,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: newSelectedPhotos, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };

      newState.photos = state.photos.map(photo =>
        photoIds.includes(photo.id)
          ? { ...photo, selected: true }
          : photo
      );

      return newState;
    }

    case 'DESELECT_ALL_IN_GROUP': {
      const group = state.groups.find(g => g.id === action.groupId);
      if (!group) return state;

      const photoIds = group.photos.map(photo => photo.id);
      const initialSelectedCount = state.selectedPhotos.length;
      const newSelectedPhotos = state.selectedPhotos.filter(id => !photoIds.includes(id));

      if (newSelectedPhotos.length === initialSelectedCount) {
        return state;
      }

      newState = {
        ...state,
        selectedPhotos: newSelectedPhotos,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: newSelectedPhotos, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };

      newState.photos = state.photos.map(photo =>
        photoIds.includes(photo.id)
          ? { ...photo, selected: false }
          : photo
      );

      return newState;
    }

    case 'SELECT_ALL': {
      const allPhotoIds = state.photos.map(photo => photo.id);

      if (allPhotoIds.length === state.selectedPhotos.length && allPhotoIds.every(id => state.selectedPhotos.includes(id))) {
        return state;
      }

      newState = {
        ...state,
        selectedPhotos: allPhotoIds,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: allPhotoIds, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };

      newState.photos = state.photos.map(photo => ({ ...photo, selected: true }));

      return newState;
    }

    case 'DESELECT_ALL': {
      if (state.selectedPhotos.length === 0) {
        return state;
      }

      newState = {
        ...state,
        selectedPhotos: [],
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: [], timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };

      newState.photos = state.photos.map(photo => ({ ...photo, selected: false }));

      return newState;
    }

    case 'UNDO': {
      if (state.currentHistoryIndex <= 0) return state;

      const historyState = state.history[state.currentHistoryIndex - 1];

      newState = {
        ...state,
        selectedPhotos: historyState.selectedPhotos,
        currentHistoryIndex: state.currentHistoryIndex - 1,
      };

      newState.photos = state.photos.map(photo => ({
        ...photo,
        selected: historyState.selectedPhotos.includes(photo.id)
      }));

      return newState;
    }

    case 'REDO': {
      if (state.currentHistoryIndex >= state.history.length - 1) return state;

      const historyState = state.history[state.currentHistoryIndex + 1];

      newState = {
        ...state,
        selectedPhotos: historyState.selectedPhotos,
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };

      newState.photos = state.photos.map(photo => ({
        ...photo,
        selected: historyState.selectedPhotos.includes(photo.id)
      }));

      return newState;
    }

    default:
      return state;
  }
}


// Helper hook to get the latest state inside async functions
function useReducerWithLatestState<S, A>(
  reducer: React.Reducer<S, A>,
  initialState: S
): [S, React.Dispatch<A>, React.MutableRefObject<S>] {
  const [state, dispatch] = useReducer(reducer, initialState);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  return [state, dispatch, stateRef];
}

export function PhotoProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch, stateRef] = useReducerWithLatestState(reducer, initialState);
  const [isLoading, setIsLoading] = useState(false);
  const [isPreparingEmbeddings, setIsPreparingEmbeddings] = useState(true);
  const [qualityEmbeddings, setQualityEmbeddings] = useState<QualityEmbeddings>(null);
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress | null>(null);

  const updateProgress = (index: number, total: number, step: ProcessingStep, fileName: string) => {
    setProcessingProgress({
      currentIndex: index,
      totalCount: total,
      currentStep: step,
      currentFileName: fileName,
    });
  };

  // Prepare quality embeddings once on mount
  useEffect(() => {
    setIsPreparingEmbeddings(true);
    prepareQualityEmbeddings()
      .then(embeddings => {
        setQualityEmbeddings(embeddings);
      })
      .catch(error => {
        console.error("Failed to prepare quality embeddings:", error);
        // Handle error appropriately, maybe show a message to the user
      })
      .finally(() => {
        setIsPreparingEmbeddings(false);
      });
  }, []); // Empty dependency array ensures this runs only once on mount

  const addPhotos = async (files: File[]) => {
    if (!files.length) return;
    if (isPreparingEmbeddings || !qualityEmbeddings) {
      console.warn("Quality embeddings not ready yet. Please wait.");
      // Optionally: show a user-facing message
      return;
    }

    setIsLoading(true);

    try {
      const totalFiles = files.length;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        // 1. Prepare basic photo info
        updateProgress(i + 1, totalFiles, 'converting', file.name);
        const id = `${file.name}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

        const ua = navigator.userAgent;
        const isSafari = /^((?!chrome|android).)*safari/i.test(ua);

        const nonHeicFile = !isSafari && await isHeic(file) ? new File([(await heicTo({
          blob: file,
          type: "image/jpeg",
          quality: 0.8,
        })) as Blob], file.name.replace(/\.hei[c|f]$/i, '.jpg'), { type: 'image/jpeg' }) : file;
        const url = URL.createObjectURL(nonHeicFile);
        const basicPhoto = {
          id,
          file,
          nonHeicFile,
          url,
          thumbnailUrl: url,
          name: file.name,
          size: file.size,
          type: file.type,
          dateCreated: new Date(file.lastModified),
          selected: false, // Will be determined by reducer later
        };


        // 2. Extract features (embedding)
        updateProgress(i + 1, totalFiles, 'extracting', file.name);
        let embedding: number[] | undefined = undefined;
        try {
          embedding = await extractFeatures(basicPhoto as Photo);
        } catch (error) {
          console.error(`Failed to extract features for ${basicPhoto.name}:`, error);
        }


        // 3. Analyze image (quality and metadata)
        updateProgress(i + 1, totalFiles, 'scoring', file.name);
        let tempAnalysisResult: { quality: number; metadata: PhotoMetadata }; // Use PhotoMetadata which allows undefined captureDate
        try {
          tempAnalysisResult = await analyzeImage(
            basicPhoto,
            embedding,
            qualityEmbeddings // Use pre-calculated embeddings
          );
        } catch (error) {
          console.error(`Failed to analyze image ${basicPhoto.name}:`, error);
          tempAnalysisResult = { quality: 0, metadata: {} }; // Default on error
        }

        // Ensure analysisResult has the expected structure with a non-undefined captureDate
        const analysisResult = {
          quality: tempAnalysisResult.quality,
          metadata: {
            ...tempAnalysisResult.metadata,
            captureDate: tempAnalysisResult.metadata.captureDate ?? basicPhoto.dateCreated
          }
        };

        // 4. Combine into final Photo object
        const newPhoto: Photo = {
          ...basicPhoto,
          quality: analysisResult.quality,
          metadata: analysisResult.metadata, // Now analysisResult.metadata is guaranteed to match Photo['metadata'] structure
          embedding: embedding,
        };

        // 5. Update state and regroup
        updateProgress(i + 1, totalFiles, 'grouping', file.name);
        // Get the latest state photos before calculating new groups
        const currentPhotos = stateRef.current.photos;
        const nextPhotos = [...currentPhotos, newPhoto];
        const { groups, uniquePhotos } = await groupSimilarPhotos(nextPhotos);


        // 6. Dispatch action to add photo and update groups
        dispatch({
          type: 'ADD_PHOTO_AND_UPDATE_GROUPS',
          photo: newPhoto,
          groups,
          uniquePhotos
        });

        // Optional: Add a small delay here if updates are too rapid for the UI
        // await new Promise(resolve => setTimeout(resolve, 50));
      } // End for loop

    } catch (error) {
      console.error("Error processing photos:", error);
      // Handle error appropriately
    } finally {
      setIsLoading(false);
      setProcessingProgress(null);
    }
  };

  const toggleSelectPhoto = (photoId: string) => {
    dispatch({ type: 'TOGGLE_SELECT_PHOTO', photoId });
  };

  const selectAllInGroup = (groupId: string) => {
    dispatch({ type: 'SELECT_ALL_IN_GROUP', groupId });
  };

  const deselectAllInGroup = (groupId: string) => {
    dispatch({ type: 'DESELECT_ALL_IN_GROUP', groupId });
  };

  const selectAll = () => {
    dispatch({ type: 'SELECT_ALL' });
  };

  const deselectAll = () => {
    dispatch({ type: 'DESELECT_ALL' });
  };

  const undo = () => {
    dispatch({ type: 'UNDO' });
  };

  const redo = () => {
    dispatch({ type: 'REDO' });
  };

  const downloadSelected = () => {
    state.selectedPhotos.forEach(photoId => {
      const photo = state.photos.find(p => p.id === photoId);
      if (photo) {
        const link = document.createElement('a');
        link.href = photo.url;
        link.download = photo.name;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        // Note: Don't revoke URLs here as they're still needed for display
      }
    });
  };

  const isSelected = (id: string) => {
    return state.selectedPhotos.includes(id);
  }

  return (
    <PhotoContext.Provider
      value={{
        state,
        isLoading,
        isPreparingEmbeddings,
        processingProgress,
        addPhotos,
        toggleSelectPhoto,
        selectAllInGroup,
        deselectAllInGroup,
        selectAll,
        deselectAll,
        undo,
        redo,
        downloadSelected,
        isSelected
      }}
    >
      {children}
    </PhotoContext.Provider>
  );
}