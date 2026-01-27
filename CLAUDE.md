# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BestPick is a React application for photo organization that uses machine learning to group similar images and score their quality. All processing happens locally in the browser using Transformers.js.

## Development Commands

- `npm run dev` - Start Vite development server (runs on http://localhost:5173)
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build
- `npm run generate-embeddings` - Regenerate quality embeddings from text prompts (downloads model from HuggingFace)

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
- Uses **SigLIP2** vision model from Hugging Face (`onnx-community/siglip2-base-patch16-512-ONNX`)
- Lazy loads models on first use with device detection (WebGPU → WASM → CPU fallback)
- Uses fp16 precision by default for efficiency

#### Feature Extraction
- `extractFeatures(photo)` - Generates normalized 512-dimensional embeddings from images
- Handles HEIC conversion for non-Safari browsers using `heic-to` library
- Embeddings are L2-normalized for cosine similarity computation

#### Quality Scoring
- `prepareQualityEmbeddings()` - Loads pre-computed embeddings from [src/data/qualityEmbeds.json](src/data/qualityEmbeds.json)
- **Category Detection**: Classifies photos into 10 categories (general, face, group, food, landscape, screenshot, drawing, pet, document, night) using softmax over anchor embeddings
- **Dimension Scoring**: 44 quality dimensions, each with positive/negative prompt pairs
- Score formula: `sigmoid(temperature × (pos_similarity - neg_similarity))` per dimension
- Category-specific dimensions only apply to matching photos (e.g., pet_attention only for pet photos)
- **Context-aware modulation**: Group photos get boosted weights for group-specific dimensions (+30%) and reduced weights for individual face dimensions (-30%)
- Final score is weighted average of applicable dimensions, mapped to 0-100
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

- Loads SigLIP2 text model from HuggingFace (`onnx-community/siglip2-base-patch16-512-ONNX`)
- Generates embeddings for 10 category anchors and 44 quality dimensions
- Each dimension has paired positive/negative prompts with weights
- Outputs to [src/data/qualityEmbeds.json](src/data/qualityEmbeds.json) with structure:
  ```json
  {
    "version": 2,
    "categories": { "face": [[...anchor embeddings...]], "pet": [...], ... },
    "dimensions": [
      { "name": "sharpness", "positive": [...], "negative": [...], "weight": 1.5, "categories": ["general", "face", ...] }
    ],
    "calibration": { "temperature": 18, "categoryTemperature": 20, "categoryThreshold": 0.20 }
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
- Models are loaded from HuggingFace (`onnx-community/siglip2-base-patch16-512-ONNX`) on first use
- Quality embeddings are pre-generated; run `npm run generate-embeddings` after changing prompts in the script
- The app creates object URLs for images; these are not revoked during runtime to maintain display
- Auto-selection logic: Selects best photo (highest quality) from each group + all unique photos
