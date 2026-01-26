import React from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import PhotoItem from './PhotoItem';
import { Star } from 'lucide-react';

interface UniquePhotosProps {
  onShowPhotoDetails?: (photoId: string) => void;
}

const UniquePhotos: React.FC<UniquePhotosProps> = ({ onShowPhotoDetails }) => {
  const { state, toggleSelectPhoto, isSelected } = usePhotoContext();
  const { uniquePhotos } = state;

  if (uniquePhotos.length === 0) {
    return null;
  }

  return (
    <div className="mb-12 animate-fade-in-up delay-100">
      <div className="flex items-center justify-between mb-6 px-4 md:px-0">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Star className="text-yellow-400 fill-yellow-400/20" size={20} />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-yellow-200 to-amber-400">
            Unique Photos
          </span>
          <span className="text-sm font-normal text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded-md border border-white/5">
            {uniquePhotos.length}
          </span>
        </h2>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 px-4 md:px-0">
        {uniquePhotos.map(photo => (
          <PhotoItem
            key={photo.id}
            id={photo.id}
            url={photo.url}
            thumbnailUrl={photo.thumbnailUrl}
            quality={photo.quality ?? 0}
            selected={isSelected(photo.id)}
            onSelect={() => toggleSelectPhoto(photo.id)}
            onShowDetails={onShowPhotoDetails ? () => onShowPhotoDetails(photo.id) : undefined}
          />
        ))}
      </div>
    </div>
  );
};

export default UniquePhotos;