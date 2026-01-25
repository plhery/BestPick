import React from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import {
  Download,
  Undo2,
  Redo2,
  Image as ImageIcon,
  CheckSquare,
  XSquare
} from 'lucide-react';

const Header: React.FC = () => {
  const {
    state,
    undo,
    redo,
    downloadSelected,
    selectAll,
    deselectAll,
    similarityThreshold,
    setSimilarityThreshold
  } = usePhotoContext();

  const selectedCount = state.selectedPhotos.length;
  const totalCount = state.photos.length;
  const canUndo = state.currentHistoryIndex > 0;
  const canRedo = state.currentHistoryIndex < state.history.length - 1;

  // Slider logic:
  // Left (0) = 0.99 (High threshold -> More groups)
  // Right (100) = 0.75 (Low threshold -> Less groups)
  // Formula: T = 0.99 - (S / 100) * (0.99 - 0.75)
  const MIN_THRESHOLD = 0.75;
  const MAX_THRESHOLD = 0.99;
  const RANGE = MAX_THRESHOLD - MIN_THRESHOLD;

  const sliderValue = ((MAX_THRESHOLD - similarityThreshold) / RANGE) * 100;

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    const newThreshold = MAX_THRESHOLD - (val / 100) * RANGE;
    // Clamp to ensure precision doesn't drift
    const clamped = Math.max(MIN_THRESHOLD, Math.min(MAX_THRESHOLD, newThreshold));
    setSimilarityThreshold(clamped);
  };

  return (
    <header className="sticky top-0 z-10 bg-gray-900 text-white shadow-md px-4 py-3 md:px-6 md:py-4 transition-all duration-200">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 md:gap-0">
        <div className="flex items-center justify-between w-full md:w-auto">
          <div className="flex items-center space-x-2">
            <ImageIcon size={24} className="text-blue-400" />
            <h1 className="text-xl md:text-2xl font-semibold">BestPick</h1>
          </div>

          {/* Mobile-only selection count (if needed for space saving, but let's keep it simple first) */}
        </div>

        <div className="flex flex-wrap items-center justify-between md:justify-end gap-3 md:space-x-6 w-full md:w-auto">
          {totalCount > 0 && (
            <div className="flex items-center space-x-2 bg-gray-800 rounded-lg px-3 py-2 w-full md:w-auto justify-center">
              <span className="text-[10px] md:text-xs text-gray-400 whitespace-nowrap">More groups</span>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={sliderValue}
                onChange={handleSliderChange}
                className="w-full md:w-32 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                title="Adjust grouping sensitivity"
              />
              <span className="text-[10px] md:text-xs text-gray-400 whitespace-nowrap">Less groups</span>
            </div>
          )}

          {totalCount > 0 && (
            <div className="flex items-center gap-3 ml-auto md:ml-0">
              <div className="flex items-center space-x-1 md:space-x-2">
                <button
                  onClick={selectAll}
                  className="p-2 rounded-full hover:bg-gray-700 transition-colors duration-200 flex items-center justify-center"
                  title="Select All"
                >
                  <CheckSquare size={20} className="text-blue-400" />
                </button>
                <button
                  onClick={deselectAll}
                  className="p-2 rounded-full hover:bg-gray-700 transition-colors duration-200 flex items-center justify-center"
                  title="Deselect All"
                >
                  <XSquare size={20} className="text-blue-400" />
                </button>
              </div>

              <div className="w-px h-6 bg-gray-700 mx-1 hidden md:block"></div>

              <div className="flex items-center space-x-1 md:space-x-2">
                <button
                  onClick={undo}
                  disabled={!canUndo}
                  className={`p-2 rounded-full hover:bg-gray-700 transition-colors duration-200 flex items-center justify-center ${!canUndo ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                  title="Undo"
                >
                  <Undo2 size={20} className="text-blue-400" />
                </button>
                <button
                  onClick={redo}
                  disabled={!canRedo}
                  className={`p-2 rounded-full hover:bg-gray-700 transition-colors duration-200 flex items-center justify-center ${!canRedo ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                  title="Redo"
                >
                  <Redo2 size={20} className="text-blue-400" />
                </button>
              </div>
            </div>
          )}

          {selectedCount > 0 && (
            <button
              onClick={downloadSelected}
              className="flex-1 md:flex-initial flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-md transition-colors duration-200 min-w-[140px]"
            >
              <Download size={18} />
              <span className="text-sm">Download ({selectedCount})</span>
            </button>
          )}

          {totalCount > 0 && (
            <div className="hidden md:block text-sm text-gray-300">
              {selectedCount > 0
                ? `${selectedCount} / ${totalCount}`
                : `${totalCount} photos`}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header