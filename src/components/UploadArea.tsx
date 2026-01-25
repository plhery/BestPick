import React, { useCallback, useState } from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import { Upload, FolderUp, FileUp, Sparkles } from 'lucide-react';

const UploadArea: React.FC = () => {
  const { addPhotos } = usePhotoContext();
  const [isDragging, setIsDragging] = useState(false);

  const handleFilesUpload = useCallback((files: FileList | null) => {
    if (!files) return;

    const imageFiles = Array.from(files).filter(file =>
      file.type.startsWith('image/')
    );

    if (imageFiles.length > 0) {
      addPhotos(imageFiles);
    }
  }, [addPhotos]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    handleFilesUpload(e.dataTransfer.files);
  }, [handleFilesUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    handleFilesUpload(e.target.files);
  }, [handleFilesUpload]);

  const handleFolderSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    handleFilesUpload(e.target.files);
  }, [handleFilesUpload]);

  return (
    <div
      className={`relative w-full p-12 lg:p-16 rounded-3xl transition-all duration-300 group overflow-hidden ${isDragging
          ? 'bg-blue-500/10 ring-4 ring-blue-500/20'
          : 'glass-panel hover:bg-surface-800/80'
        }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Animated Gradient Border */}
      <div className={`absolute inset-0 pointer-events-none transition-opacity duration-500 ${isDragging ? 'opacity-100' : 'opacity-0'}`}>
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-blue-500 to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-blue-500 to-transparent" />
        <div className="absolute inset-y-0 left-0 w-px bg-gradient-to-b from-transparent via-blue-500 to-transparent" />
        <div className="absolute inset-y-0 right-0 w-px bg-gradient-to-b from-transparent via-blue-500 to-transparent" />
      </div>

      <div className="relative z-10 flex flex-col items-center justify-center text-center">
        <div className={`mb-6 p-4 rounded-full transition-all duration-300 ${isDragging ? 'bg-blue-500/20 text-blue-300 scale-110' : 'bg-slate-800/50 text-slate-400 group-hover:bg-slate-700/50 group-hover:scale-105'
          }`}>
          {isDragging ? <Sparkles size={48} className="animate-spin-slow" /> : <Upload size={48} />}
        </div>

        <h2 className="text-2xl md:text-3xl font-bold mb-3 text-white">
          {isDragging ? 'Drop it like it\'s hot!' : 'Drag & Drop Photos Here'}
        </h2>

        <p className="text-slate-400 mb-8 max-w-lg text-lg leading-relaxed">
          Upload your collection to automatically find the best shots.
          <br />
          <span className="text-sm text-slate-500 mt-2 block">
            All processing happens locally in your browser.
          </span>
        </p>

        <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
          <label className="group/btn relative flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white py-3 px-8 rounded-xl cursor-pointer shadow-lg shadow-blue-500/20 transition-all hover:shadow-blue-500/40 hover:-translate-y-0.5 active:translate-y-0">
            <FileUp size={20} />
            <span className="font-semibold">Select Files</span>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
          </label>

          <label className="group/btn flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 py-3 px-8 rounded-xl cursor-pointer border border-white/5 transition-all hover:border-white/10 hover:-translate-y-0.5 active:translate-y-0">
            <FolderUp size={20} />
            <span className="font-semibold">Select Folder</span>
            <input
              type="file"
              accept="image/*"
              multiple
              directory=""
              webkitdirectory=""
              onChange={handleFolderSelect}
              className="hidden"
            />
          </label>
        </div>
      </div>
    </div>
  );
};

export default UploadArea;