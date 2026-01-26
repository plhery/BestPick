import React, { useReducer, useState, useEffect, useRef } from 'react';
import { PhotoContext, ProcessingProgress, ProcessingStep } from './PhotoContextDef';
import { AppState, Photo, PhotoGroup, PhotoMetadata } from '../types';
import { analyzeImage, groupSimilarPhotos, extractFeatures, prepareQualityEmbeddings } from '../utils/imageAnalysis';
import { isHeic, heicTo } from 'heic-to';

// Define a type for the quality embeddings - matches PreparedQualityEmbeddings from imageAnalysis
type QualityEmbeddings = Awaited<ReturnType<typeof prepareQualityEmbeddings>> | null;

type PhotoAction =
  | { type: 'ADD_PHOTO_AND_UPDATE_GROUPS'; photo: Photo; groups: PhotoGroup[]; uniquePhotos: Photo[] }
  | { type: 'ADD_PHOTOS_BATCH_AND_UPDATE_GROUPS'; photos: Photo[]; groups: PhotoGroup[]; uniquePhotos: Photo[] }
  | { type: 'TOGGLE_SELECT_PHOTO'; photoId: string }
  | { type: 'SELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'DESELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'SELECT_ALL' }
  | { type: 'DESELECT_ALL' }
  | { type: 'UNDO' }
  | { type: 'REDO' }
  | { type: 'UPDATE_GROUPS'; groups: PhotoGroup[]; uniquePhotos: Photo[] };

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

    case 'ADD_PHOTOS_BATCH_AND_UPDATE_GROUPS': {
      const { photos, groups, uniquePhotos } = action;

      // Add all new photos and update groups/unique photos
      const updatedPhotos = [...state.photos, ...photos];
      const updatedStateBase: Omit<AppState, 'history' | 'currentHistoryIndex' | 'selectedPhotos'> = {
        ...state,
        photos: updatedPhotos,
        groups,
        uniquePhotos,
      };

      // Auto-select the best photo in each group and all unique photos
      const autoSelectedPhotos = [
        ...uniquePhotos.map(p => p.id),
        ...groups.map(g => g.photos[0].id)
      ];

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
      const newHistoryEntry = { selectedPhotos: intermediateState.selectedPhotos, timestamp: new Date() };

      const history = [
        ...state.history.slice(0, state.currentHistoryIndex + 1),
        newHistoryEntry
      ];
      let currentHistoryIndex = history.length - 1;

      if (history.length > MAX_HISTORY_SIZE) {
        const trimmedHistory = history.slice(history.length - MAX_HISTORY_SIZE);
        currentHistoryIndex = MAX_HISTORY_SIZE - 1;

        return {
          ...intermediateState,
          history: trimmedHistory,
          currentHistoryIndex,
        };
      }

      return {
        ...intermediateState,
        history,
        currentHistoryIndex,
      };
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

    case 'UPDATE_GROUPS': {
      return {
        ...state,
        groups: action.groups,
        uniquePhotos: action.uniquePhotos,
      };
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
  const [similarityThreshold, setSimilarityThreshold] = useState(0.75);

  // Ref to track if re-grouping is in progress (prevents race conditions on mobile)
  const isRegroupingRef = useRef(false);

  // Detect mobile device for longer debounce
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

  // Debounced regrouping when threshold changes
  useEffect(() => {
    let isCancelled = false;

    // Longer debounce on mobile to reduce memory pressure
    const debounceTime = isMobile ? 1000 : 500;

    const timer = setTimeout(async () => {
      const currentPhotos = stateRef.current.photos;
      if (currentPhotos.length === 0) return;

      // Skip if already re-grouping to prevent multiple simultaneous operations
      // This prevents memory spikes on mobile when user drags the slider quickly
      if (isRegroupingRef.current) {
        return;
      }

      isRegroupingRef.current = true;
      setIsLoading(true);
      try {
        const { groups, uniquePhotos } = await groupSimilarPhotos(currentPhotos, similarityThreshold);

        // Only update state if this effect hasn't been cancelled
        if (!isCancelled) {
          dispatch({
            type: 'UPDATE_GROUPS',
            groups,
            uniquePhotos
          });
        }
      } catch (error) {
        console.error("Failed to regroup photos:", error);
      } finally {
        if (!isCancelled) {
          setIsLoading(false);
          isRegroupingRef.current = false;
        }
      }
    }, debounceTime);

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [similarityThreshold, dispatch, stateRef]);

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
      const newPhotos: Photo[] = [];

      // Process all photos first (convert, extract, score)
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

        // Generate efficient thumbnail (512px for SigLIP2)
        // This is the ONLY image data we will keep in memory!
        let thumbnailBlob: Blob;
        try {
          thumbnailBlob = await generateThumbnail(nonHeicFile, 512);
        } catch (e) {
          console.warn("Failed to generate thumbnail, falling back to original file (dangerous for memory)", e);
          thumbnailBlob = nonHeicFile;
        }

        const thumbnailUrl = URL.createObjectURL(thumbnailBlob);

        // Create a temporary object for processing that HAS the file references
        // We need the original file for EXIF data in analyzeImage
        const processingPhoto: Photo = {
          id,
          file,
          nonHeicFile,
          url: thumbnailUrl, // Use thumbnail URL for CLIP/display
          thumbnailUrl: thumbnailUrl,
          name: file.name,
          size: file.size,
          type: file.type,
          dateCreated: new Date(file.lastModified),
          selected: false,
        };


        // 2. Extract features (embedding) - Uses the thumbnail URL from processingPhoto
        updateProgress(i + 1, totalFiles, 'extracting', file.name);
        let embedding: number[] | undefined = undefined;
        try {
          embedding = await extractFeatures(processingPhoto);
        } catch (error) {
          console.error(`Failed to extract features for ${processingPhoto.name}:`, error);
        }


        // 3. Analyze image (quality and metadata) - Uses processingPhoto.file for EXIF
        updateProgress(i + 1, totalFiles, 'scoring', file.name);
        let tempAnalysisResult: { quality: number; metadata: PhotoMetadata };
        try {
          tempAnalysisResult = await analyzeImage(
            processingPhoto,
            embedding,
            qualityEmbeddings
          );
        } catch (error) {
          console.error(`Failed to analyze image ${processingPhoto.name}:`, error);
          tempAnalysisResult = { quality: 0, metadata: {} };
        }

        const analysisResult = {
          quality: tempAnalysisResult.quality,
          qualityBreakdown: tempAnalysisResult.qualityBreakdown,
          metadata: {
            ...tempAnalysisResult.metadata,
            captureDate: tempAnalysisResult.metadata.captureDate ?? processingPhoto.dateCreated
          }
        };

        // 4. Combine into final Photo object - CRITICAL: DO NOT INCLUDE FILE REFS
        // We specifically omit 'file' and 'nonHeicFile' to allow GC
        const newPhoto: Photo = {
          id: processingPhoto.id,
          url: processingPhoto.url,
          thumbnailUrl: processingPhoto.thumbnailUrl,
          name: processingPhoto.name,
          size: processingPhoto.size,
          type: processingPhoto.type,
          dateCreated: processingPhoto.dateCreated,
          selected: false,
          quality: analysisResult.quality,
          qualityBreakdown: analysisResult.qualityBreakdown,
          metadata: analysisResult.metadata,
          embedding: embedding,
          // Explicitly undefined to ensure no reference is held
          file: undefined,
          nonHeicFile: undefined
        };

        // Add to batch instead of dispatching immediately
        newPhotos.push(newPhoto);
      } // End for loop

      // 5. Group all photos ONCE at the end (much more efficient than grouping after each photo)
      if (newPhotos.length > 0) {
        updateProgress(totalFiles, totalFiles, 'grouping', 'all photos');

        const currentPhotos = stateRef.current.photos;
        const allPhotos = [...currentPhotos, ...newPhotos];
        const { groups, uniquePhotos } = await groupSimilarPhotos(allPhotos, similarityThreshold);

        // 6. Dispatch batch update - adds all photos and updates groups in one operation
        dispatch({
          type: 'ADD_PHOTOS_BATCH_AND_UPDATE_GROUPS',
          photos: newPhotos,
          groups,
          uniquePhotos
        });
      }

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
        isSelected,
        similarityThreshold,
        setSimilarityThreshold
      }}
    >
      {children}
    </PhotoContext.Provider>
  );
}
