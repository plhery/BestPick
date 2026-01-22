import React from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import PhotoItem from './PhotoItem';
import { CheckCircle, XCircle } from 'lucide-react';

interface PhotoGroupProps {
  groupId: string;
  title: string;
  date: Date;
  photos: Array<{
    id: string;
    url: string;
    thumbnailUrl: string;
    quality?: number;
    selected: boolean;
  }>;
  similarity: number;
}

const PhotoGroup: React.FC<PhotoGroupProps> = ({
  groupId,
  title,
  photos,
  similarity
}) => {
  const { selectAllInGroup, deselectAllInGroup, toggleSelectPhoto, isSelected } = usePhotoContext();

  const handleSelectAll = () => {
    selectAllInGroup(groupId);
  };

  const handleDeselectAll = () => {
    deselectAllInGroup(groupId);
  };

  // Photos are already sorted by quality in the reducer/grouping logic

  return (
    <div className="mb-8 bg-gray-800 rounded-lg overflow-hidden">
      <div className="p-4 flex items-center justify-between border-b border-gray-700">
        <div>
          <h3 className="text-lg font-medium text-white">{title}</h3>
          <p className="text-gray-400 text-sm">Similarity: {(similarity * 100).toFixed(0)}%</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleSelectAll}
            className="p-2 rounded-full hover:bg-gray-700 transition-colors duration-200"
            title="Select All in Group"
          >
            <CheckCircle size={20} className="text-blue-400" />
          </button>
          <button
            onClick={handleDeselectAll}
            className="p-2 rounded-full hover:bg-gray-700 transition-colors duration-200"
            title="Deselect All in Group"
          >
            <XCircle size={20} className="text-blue-400" />
          </button>
        </div>
      </div>

      <div className="p-4 overflow-x-auto">
        <div className="flex space-x-4 pb-2">
          {photos.map((photo, index) => (
            <PhotoItem
              key={photo.id}
              id={photo.id}
              url={photo.url}
              thumbnailUrl={photo.thumbnailUrl}
              quality={photo.quality ?? 0}
              selected={isSelected(photo.id)}
              isBest={index === 0}
              onSelect={() => toggleSelectPhoto(photo.id)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

const PhotoGroups: React.FC = () => {
  const { state } = usePhotoContext();
  const { groups } = state;

  if (groups.length === 0) {
    return null;
  }

  return (
    <div className="mb-8">
      <h2 className="text-xl font-semibold mb-4 text-white px-4">Similar Photo Groups</h2>
      <div className="space-y-6 px-4">
        {groups.map(group => (
          <PhotoGroup
            key={group.id}
            groupId={group.id}
            title={group.title}
            date={group.date}
            photos={group.photos}
            similarity={group.similarity}
          />
        ))}
      </div>
    </div>
  );
};

export default PhotoGroups;