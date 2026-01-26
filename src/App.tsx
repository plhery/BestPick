import React, { useEffect, useState } from 'react';
import { PhotoProvider } from './context/PhotoContext';
import { usePhotoContext } from './context/usePhotoContext';
import Header from './components/Header';
import UploadArea from './components/UploadArea';
import PhotoGroups from './components/PhotoGroups';
import UniquePhotos from './components/UniquePhotos';
import ErrorBoundary from './components/ErrorBoundary';
import QualityDetailsModal from './components/QualityDetailsModal';
import { Loader2, ImageIcon, Sparkles, Layers, RefreshCw } from 'lucide-react';
import previewImageUrl from '../preview.png';
import type { ProcessingStep } from './context/PhotoContextDef';
import { preheatModel } from './utils/imageAnalysis';

const stepLabels: Record<ProcessingStep, { label: string; icon: React.ReactNode }> = {
  'idle': { label: 'Preparing...', icon: <Loader2 className="animate-spin h-5 w-5" /> },
  'converting': { label: 'Converting image...', icon: <RefreshCw className="animate-spin h-5 w-5" /> },
  'extracting': { label: 'Analysing visuals...', icon: <ImageIcon className="h-5 w-5" /> },
  'scoring': { label: 'Evaluating quality...', icon: <Sparkles className="h-5 w-5" /> },
  'grouping': { label: 'Finding matches...', icon: <Layers className="h-5 w-5" /> },
};

const LoadingOverlay: React.FC = () => {
  const { processingProgress, isPreparingEmbeddings } = usePhotoContext();

  const progress = processingProgress;
  const step = progress?.currentStep ?? 'idle';
  const stepInfo = stepLabels[step];
  const percentage = progress ? Math.round((progress.currentIndex / progress.totalCount) * 100) : 0;

  return (
    <div className="fixed inset-0 bg-surface-950/80 backdrop-blur-md flex items-center justify-center z-50 animate-in fade-in duration-300">
      <div className="glass-panel text-white max-w-md w-full mx-4 rounded-3xl p-10 shadow-2xl border-white/10 relative overflow-hidden">

        {/* Background Glow */}
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-32 h-32 bg-blue-500/20 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -mb-10 -ml-10 w-32 h-32 bg-indigo-500/20 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col items-center">
          <div className="mb-6 relative">
            <div className="absolute inset-0 bg-blue-500 blur-xl opacity-20 animate-pulse" />
            <Loader2 className="animate-spin h-12 w-12 text-blue-400 relative z-10" />
          </div>

          {isPreparingEmbeddings ? (
            <>
              <p className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-200 to-indigo-200">
                Warming up AI Model...
              </p>
              <p className="text-sm text-slate-400">This happens locally on your device</p>
            </>
          ) : progress ? (
            <>
              <p className="text-lg font-semibold mb-1 text-slate-200">Processing Photos</p>
              <div className="flex items-baseline gap-1 mb-6">
                <span className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
                  {percentage}%
                </span>
                <span className="text-slate-500 text-sm">complete</span>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-slate-800/50 rounded-full h-2 mb-6 overflow-hidden border border-white/5">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-blue-500 rounded-full transition-all duration-300 ease-out relative"
                  style={{ width: `${percentage}%` }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-pulse" />
                </div>
              </div>

              {/* Step indicator */}
              <div className="flex items-center gap-3 text-sm text-slate-300 bg-slate-800/50 px-4 py-2 rounded-full border border-white/5">
                <span className="text-blue-400">{stepInfo.icon}</span>
                <span>{stepInfo.label}</span>
              </div>

              <p className="mt-4 text-xs text-slate-500 truncate max-w-[200px]" title={progress.currentFileName}>
                {progress.currentFileName}
              </p>
            </>
          ) : (
            <>
              <p className="text-lg font-semibold">Preparing...</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};


const MainContent: React.FC = () => {
  const { state, isLoading } = usePhotoContext();
  const hasPhotos = state.photos.length > 0;
  const [selectedPhotoId, setSelectedPhotoId] = useState<string | null>(null);

  // Find the photo for the details modal
  const selectedPhoto = selectedPhotoId
    ? state.photos.find(p => p.id === selectedPhotoId) ?? null
    : null;

  // Preheat the AI model on app startup
  useEffect(() => {
    preheatModel();
  }, []);

  return (
    <div className="flex flex-col min-h-screen text-slate-200">
      <Header />

      {isLoading && <LoadingOverlay />}

      <main className={`flex-1 container mx-auto px-4 py-8 md:py-12 max-w-[1600px] ${isLoading ? 'opacity-30 pointer-events-none blur-sm transition-all duration-500' : 'transition-all duration-500'}`}>
        {!hasPhotos ? (
          <div className="max-w-4xl mx-auto animate-fade-in-up">
            <div className="text-center mb-10">
              <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-100 via-white to-indigo-100 tracking-tight">
                Organize Your Photos
              </h1>
              <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                Automatically group similar shots and find the best ones using AI.
                <br className="hidden md:block" /> Private, secure, and fast.
              </p>
            </div>

            <UploadArea />

            <div className="mt-16 text-center opacity-80 hover:opacity-100 transition-opacity duration-500">
              <div className="glass-panel p-2 rounded-2xl inline-block">
                <img
                  src={previewImageUrl}
                  alt="BestPick App Preview"
                  className="max-w-full md:max-w-4xl mx-auto rounded-xl shadow-2xl border border-white/5"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="animate-fade-in-up">
            <div className="mb-10">
              <UploadArea />
            </div>
            <PhotoGroups onShowPhotoDetails={setSelectedPhotoId} />
            <UniquePhotos onShowPhotoDetails={setSelectedPhotoId} />
          </div>
        )}
      </main>

      <footer className="mt-auto py-8 border-t border-white/5 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <div className="h-px w-8 bg-gradient-to-r from-transparent to-slate-700" />
          <ImageIcon size={16} className="text-slate-600" />
          <div className="h-px w-8 bg-gradient-to-l from-transparent to-slate-700" />
        </div>
        <p className="text-slate-500 text-sm">BestPick &bull; AI-Powered Photo Organization</p>
      </footer>

      {/* Quality Details Modal */}
      {selectedPhoto && (
        <QualityDetailsModal
          photo={selectedPhoto}
          onClose={() => setSelectedPhotoId(null)}
        />
      )}
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
