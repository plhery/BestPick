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

  return (
    <header className="sticky top-0 z-10 bg-gray-900 text-white shadow-md px-6 py-4 flex items-center justify-between transition-all duration-200">
      <div className="flex items-center space-x-2">
        <ImageIcon size={24} className="text-blue-400" />
        <h1 className="text-2xl font-semibold">BestPick</h1>
      </div>

      <div className="flex items-center space-x-6">
        {totalCount > 0 && (
          <div className="flex items-center space-x-2 bg-gray-800 rounded-lg px-3 py-1.5">
            <span className="text-xs text-gray-400">Similarity</span>
            <input
              type="range"
              min="0.5"
              max="0.99"
              step="0.01"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-24 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              title={`Threshold: ${Math.round(similarityThreshold * 100)}%`}
            />
            <span className="text-xs text-blue-400 w-8 text-right">{Math.round(similarityThreshold * 100)}%</span>
          </div>
        )}
        {totalCount > 0 && (
          <>
            <div className="flex items-center space-x-2">
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

            <div className="flex items-center space-x-2">
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
          </>
        )}

        {selectedCount > 0 && (
          <button
            onClick={downloadSelected}
            className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-md transition-colors duration-200"
          >
            <Download size={18} />
            <span>Download Selected ({selectedCount})</span>
          </button>
        )}

        {totalCount > 0 && (
          <div className="text-sm text-gray-300">
            {selectedCount > 0
              ? `${selectedCount} of ${totalCount} selected`
              : `${totalCount} photos total`}
          </div>
        )}
      </div>
    </header>
  );
};

export default Header