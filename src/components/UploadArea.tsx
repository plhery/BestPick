import React, { useCallback, useState } from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import { Upload, FolderUp, FileUp } from 'lucide-react';

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
      className={`flex flex-col items-center justify-center w-full p-12 border-2 border-dashed rounded-lg transition-all duration-300 ${isDragging
        ? 'border-blue-500 bg-blue-900 bg-opacity-20'
        : 'border-gray-600 bg-gray-800 hover:bg-gray-700'
        }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <Upload
        size={64}
        className={`mb-4 transition-colors duration-300 ${isDragging ? 'text-blue-400' : 'text-gray-400'
          }`}
      />

      <h2 className="text-xl font-semibold mb-2 text-white">
        Drag & Drop Photos Here
      </h2>

      <p className="text-gray-400 mb-6 text-center max-w-md">
        Upload multiple photos to organize and declutter your collection.
        Everything happens in your browser, so your photos stay private!
      </p>

      <div className="flex gap-4">
        <label className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 transition-colors duration-200 text-white py-2 px-4 rounded-md cursor-pointer">
          <FileUp size={18} />
          <span>Select Files</span>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>

        <label className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 transition-colors duration-200 text-white py-2 px-4 rounded-md cursor-pointer">
          <FolderUp size={18} />
          <span>Select Folder</span>
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
  );
};

export default UploadArea;