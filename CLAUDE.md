# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BestPick is a React application for photo organization that uses machine learning to group similar images and score their quality. All processing happens locally in the browser using Transformers.js.

## Development Commands

- `npm run dev` - Start Vite development server (runs on http://localhost:5173)
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build
- `npm run generate-embeddings` - Regenerate quality embeddings from text prompts (requires ML model in public/models/)

## Architecture

### State Management

The app uses React Context with `useReducer` for state management ([src/context/PhotoContext.tsx](src/context/PhotoContext.tsx)):

- **PhotoContext** manages all application state including photos, groups, selections, and history
- **Reducer pattern** handles state updates with actions like `ADD_PHOTO_AND_UPDATE_GROUPS`, `TOGGLE_SELECT_PHOTO`, etc.
- **Undo/Redo** system tracks selection history (max 50 states) via `history` and `currentHistoryIndex`
- **useReducerWithLatestState** custom hook provides a `stateRef` to access current state in async callbacks, critical for the photo processing pipeline

### ML Pipeline (imageAnalysis.ts)

The core ML functionality is in [src/utils/imageAnalysis.ts](src/utils/imageAnalysis.ts):

#### Model Loading
- Uses **MobileCLIP2** vision and text models from Hugging Face (`plhery/mobileclip2-onnx`)
- Available model sizes: S0 (43MB), S2 (136MB, default), B (330MB), L-14 (1.1GB)
- Lazy loads models on first use with device detection (WebGPU → WASM → CPU fallback)
- Model size configurable via `MODEL_SIZE` constant in imageAnalysis.ts

#### Feature Extraction
- `extractFeatures(photo)` - Generates normalized 512-dimensional embeddings from images
- Handles HEIC conversion for non-Safari browsers using `heic-to` library
- Embeddings are L2-normalized for cosine similarity computation

#### Quality Scoring
- `prepareQualityEmbeddings()` - Loads pre-computed embeddings from [src/data/qualityEmbeds.json](src/data/qualityEmbeds.json)
- Supports two scoring modes: **general** (landscapes, objects) and **face** (portraits)
- `analyzeImage()` computes weighted similarity against positive/negative prompt embeddings
- Face detection heuristic uses sigmoid on face similarity margin with 0.4 threshold
- Blends general + face scores based on face confidence and `calibration.faceWeight`
- Maps raw similarity scores to 0-100 range using calibration slope/offset
- Extracts EXIF metadata (capture date) using ExifReader

#### Photo Grouping
- `groupSimilarPhotos()` - Uses **Union-Find** algorithm for connected component clustering
- Computes full similarity matrix via batch matmul for efficiency
- **Time-aware thresholds**: Photos >2 hours apart are never grouped, photos within 1 minute get lower threshold (-0.05), photos 10min-2h apart require higher similarity (+0.05)
- Photos without EXIF dates get +0.1 threshold penalty
- Groups sorted by date (descending), photos within groups sorted by quality
- Returns both `groups` (2+ similar photos) and `uniquePhotos` (singletons)

### Photo Processing Flow

When users upload photos ([src/context/PhotoContext.tsx:374-483](src/context/PhotoContext.tsx#L374-L483)):

1. **Converting** - HEIC to JPEG conversion if needed, create object URL
2. **Extracting** - Generate CLIP embedding via `extractFeatures()`
3. **Scoring** - Compute quality score and metadata via `analyzeImage()` with pre-loaded quality embeddings
4. **Grouping** - Re-run `groupSimilarPhotos()` with all photos including the new one
5. **Dispatch** - Single `ADD_PHOTO_AND_UPDATE_GROUPS` action updates state and auto-selects best photos

Progress updates via `setProcessingProgress()` are displayed in the UI overlay.

### Similarity Threshold

- Default: 0.85 (adjustable via UI slider)
- Stored in PhotoContext state as `similarityThreshold`
- Debounced re-grouping (500ms) when threshold changes
- Effective threshold modified by time distance between photos

### Quality Embeddings Generation

The [scripts/generate-embeddings.ts](scripts/generate-embeddings.ts) script:

- Loads MobileCLIP2 text model from HuggingFace (`plhery/mobileclip2-onnx`)
- Generates embeddings for weighted prompt sets (general and face, positive and negative)
- Outputs to [src/data/qualityEmbeds.json](src/data/qualityEmbeds.json) with structure:
  ```json
  {
    "general": { "positive": [], "positiveWeights": [], "negative": [], "negativeWeights": [] },
    "face": { "positive": [], "positiveWeights": [], "negative": [], "negativeWeights": [] },
    "calibration": { "slope": 12, "offset": 0.5, "faceWeight": 0.6 }
  }
  ```

## Tech Stack

- **React 19** with TypeScript
- **Vite** for build tooling
- **Tailwind CSS 4** for styling
- **Transformers.js** for ML inference
- **ExifReader** for metadata extraction
- **heic-to** for HEIC conversion

## Key Files

- [src/App.tsx](src/App.tsx) - Main app component with upload/display logic
- [src/context/PhotoContext.tsx](src/context/PhotoContext.tsx) - State management and photo processing orchestration
- [src/utils/imageAnalysis.ts](src/utils/imageAnalysis.ts) - ML pipeline (embeddings, quality, grouping)
- [src/types/index.ts](src/types/index.ts) - TypeScript interfaces
- [scripts/generate-embeddings.ts](scripts/generate-embeddings.ts) - Offline embedding generation
- [src/data/qualityEmbeds.json](src/data/qualityEmbeds.json) - Pre-computed quality prompt embeddings

## Important Notes

- All ML processing runs client-side; no backend required
- Models are loaded from HuggingFace (`plhery/mobileclip2-onnx`) on first use
- Quality embeddings are pre-generated; run `npm run generate-embeddings` after changing prompts in the script
- The app creates object URLs for images; these are not revoked during runtime to maintain display
- Auto-selection logic: Selects best photo (highest quality) from each group + all unique photos
