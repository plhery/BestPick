import React, { useEffect } from 'react';
import { PhotoProvider } from './context/PhotoContext';
import { usePhotoContext } from './context/usePhotoContext';
import Header from './components/Header';
import UploadArea from './components/UploadArea';
import PhotoGroups from './components/PhotoGroups';
import UniquePhotos from './components/UniquePhotos';
import ErrorBoundary from './components/ErrorBoundary';
import { Loader2, ImageIcon, Sparkles, Layers, RefreshCw } from 'lucide-react';
import previewImageUrl from '../preview.png';
import type { ProcessingStep } from './context/PhotoContextDef';
import { preheatModel } from './utils/imageAnalysis';

const stepLabels: Record<ProcessingStep, { label: string; icon: React.ReactNode }> = {
  'idle': { label: 'Preparing...', icon: <Loader2 className="animate-spin h-5 w-5" /> },
  'converting': { label: 'Converting image...', icon: <RefreshCw className="animate-spin h-5 w-5" /> },
  'extracting': { label: 'Extracting features...', icon: <ImageIcon className="h-5 w-5" /> },
  'scoring': { label: 'Scoring quality...', icon: <Sparkles className="h-5 w-5" /> },
  'grouping': { label: 'Grouping similar photos...', icon: <Layers className="h-5 w-5" /> },
};

const LoadingOverlay: React.FC = () => {
  const { processingProgress, isPreparingEmbeddings } = usePhotoContext();

  const progress = processingProgress;
  const step = progress?.currentStep ?? 'idle';
  const stepInfo = stepLabels[step];
  const percentage = progress ? Math.round((progress.currentIndex / progress.totalCount) * 100) : 0;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="flex flex-col items-center text-white max-w-md w-full mx-4 bg-gray-800 rounded-2xl p-8 shadow-2xl">
        <Loader2 className="animate-spin h-12 w-12 mb-4 text-blue-400" />

        {isPreparingEmbeddings ? (
          <>
            <p className="text-lg font-semibold mb-2">Loading AI Model...</p>
            <p className="text-sm text-gray-400">This only happens once</p>
          </>
        ) : progress ? (
          <>
            <p className="text-lg font-semibold mb-1">Processing Photos</p>
            <p className="text-2xl font-bold text-blue-400 mb-4">
              {progress.currentIndex} / {progress.totalCount}
            </p>

            {/* Progress bar */}
            <div className="w-full bg-gray-700 rounded-full h-3 mb-4 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${percentage}%` }}
              />
            </div>

            {/* Step indicator */}
            <div className="flex items-center gap-2 text-sm text-gray-300 mb-2">
              {stepInfo.icon}
              <span>{stepInfo.label}</span>
            </div>

            {/* Current file name */}
            <p className="text-xs text-gray-500 truncate max-w-full" title={progress.currentFileName}>
              {progress.currentFileName}
            </p>
          </>
        ) : (
          <>
            <p className="text-lg font-semibold">Processing Photos...</p>
            <p className="text-sm text-gray-400">This may take a moment.</p>
          </>
        )}
      </div>
    </div>
  );
};


const MainContent: React.FC = () => {
  const { state, isLoading } = usePhotoContext();
  const hasPhotos = state.photos.length > 0;

  // Preheat the AI model on app startup
  useEffect(() => {
    preheatModel();
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-white">
      <Header />

      {isLoading && <LoadingOverlay />}

      <main className={`flex-1 container mx-auto py-8 ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}>
        {!hasPhotos ? (
          <>
            <h1 className="text-3xl font-bold mb-6 px-4">Organize Your Photos</h1>
            <UploadArea />
            <div className="mt-8 px-4 text-center">
              <img
                src={previewImageUrl}
                alt="BestPick App Preview"
                className="max-w-full md:max-w-4xl mx-auto rounded-lg shadow-lg border border-gray-700"
              />
            </div>
          </>
        ) : (
          <>
            <div className="mb-8 px-4">
              <UploadArea />
            </div>
            <PhotoGroups />
            <UniquePhotos />
          </>
        )}
      </main>

      <footer className="bg-gray-900 border-t border-gray-800 py-4 px-6 text-center text-gray-500 text-sm">
        <p>BestPick - Declutter your photo collections</p>
      </footer>
    </div>
  );
};

function App() {
  return (
    <ErrorBoundary>
      <PhotoProvider>
        <MainContent />
      </PhotoProvider>
    </ErrorBoundary>
  );
}

export default App;
