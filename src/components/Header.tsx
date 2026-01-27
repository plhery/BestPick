import React from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import {
  Download,
  Undo2,
  Redo2,
  Image as ImageIcon,
  CheckSquare,
  XSquare,
  MoreHorizontal
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
  // Right (100) = 0.5 (Low threshold -> Less groups)
  const MIN_THRESHOLD = 0.2;
  const MAX_THRESHOLD = 0.99;
  const RANGE = MAX_THRESHOLD - MIN_THRESHOLD;

  const sliderValue = ((MAX_THRESHOLD - similarityThreshold) / RANGE) * 100;

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    const newThreshold = MAX_THRESHOLD - (val / 100) * RANGE;
    const clamped = Math.max(MIN_THRESHOLD, Math.min(MAX_THRESHOLD, newThreshold));
    setSimilarityThreshold(clamped);
  };

  return (
    <header className="glass-header px-4 py-3 md:px-8 md:py-4 transition-all duration-300">
      <div className="max-w-[1920px] mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4 md:gap-0">

        {/* Logo Section */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <ImageIcon size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-200">
              BestPick
            </h1>
            <p className="text-[10px] text-slate-400 font-medium tracking-wide uppercase">AI Photo Organizer</p>
          </div>
        </div>

        <div className="flex flex-col md:flex-row items-center gap-4 flex-1 justify-end">

          {totalCount > 0 && (
            <>
              {/* Grouping Sensitivity Slider */}
              <div className="flex items-center space-x-3 bg-slate-800/50 rounded-full px-4 py-1.5 border border-white/5 w-full md:w-auto">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Similar</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={sliderValue}
                  onChange={handleSliderChange}
                  className="w-full md:w-24 h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Distinct</span>
              </div>

              <div className="h-6 w-px bg-white/10 hidden md:block" />

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={selectAll}
                  className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-blue-400 transition-colors"
                  title="Select All"
                >
                  <CheckSquare size={18} />
                </button>
                <button
                  onClick={deselectAll}
                  className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-red-400 transition-colors"
                  title="Deselect All"
                >
                  <XSquare size={18} />
                </button>

                <div className="h-4 w-px bg-white/10 mx-1" />

                <button
                  onClick={undo}
                  disabled={!canUndo}
                  className={`p-2 rounded-lg hover:bg-white/5 transition-colors ${!canUndo ? 'text-slate-600 cursor-not-allowed' : 'text-slate-400 hover:text-white'
                    }`}
                  title="Undo"
                >
                  <Undo2 size={18} />
                </button>
                <button
                  onClick={redo}
                  disabled={!canRedo}
                  className={`p-2 rounded-lg hover:bg-white/5 transition-colors ${!canRedo ? 'text-slate-600 cursor-not-allowed' : 'text-slate-400 hover:text-white'
                    }`}
                  title="Redo"
                >
                  <Redo2 size={18} />
                </button>
              </div>
            </>
          )}

          {/* Download Button */}
          {selectedCount > 0 && (
            <button
              onClick={downloadSelected}
              className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white px-5 py-2 rounded-full shadow-lg shadow-blue-500/25 transition-all hover:scale-105 active:scale-95 font-medium text-sm"
            >
              <Download size={16} />
              <span>Save {selectedCount} Photos</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;