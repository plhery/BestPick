import React from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import PhotoItem from './PhotoItem';

const UniquePhotos: React.FC = () => {
  const { state, toggleSelectPhoto, isSelected } = usePhotoContext();
  const { uniquePhotos } = state;

  if (uniquePhotos.length === 0) {
    return null;
  }

  return (
    <div className="mb-8">
      <h2 className="text-xl font-semibold mb-4 text-white px-4">Unique Photos</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 px-4">
        {uniquePhotos.map(photo => (
          <PhotoItem
            key={photo.id}
            id={photo.id}
            url={photo.url}
            thumbnailUrl={photo.thumbnailUrl}
            quality={photo.quality}
            selected={isSelected(photo.id)}
            onSelect={() => toggleSelectPhoto(photo.id)}
          />
        ))}
      </div>
    </div>
  );
};

export default UniquePhotos;