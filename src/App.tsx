import React from 'react';
import { PhotoProvider } from './context/PhotoContext';
import { usePhotoContext } from './context/usePhotoContext';
import Header from './components/Header';
import UploadArea from './components/UploadArea';
import PhotoGroups from './components/PhotoGroups';
import UniquePhotos from './components/UniquePhotos';
import { Loader2 } from 'lucide-react';
import previewImageUrl from '../preview.png';

const LoadingOverlay: React.FC = () => (
  <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
    <div className="flex flex-col items-center text-white">
      <Loader2 className="animate-spin h-12 w-12 mb-4" />
      <p className="text-lg font-semibold">Processing Photos...</p>
      <p className="text-sm text-gray-400">This may take a moment.</p>
    </div>
  </div>
);

const MainContent: React.FC = () => {
  const { state, isLoading } = usePhotoContext();
  const hasPhotos = state.photos.length > 0;

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
    <PhotoProvider>
      <MainContent />
    </PhotoProvider>
  );
}

export default App;