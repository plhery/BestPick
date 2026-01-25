import React from 'react';
import { usePhotoContext } from '../context/usePhotoContext';
import PhotoItem from './PhotoItem';
import { CheckCircle2, XCircle, Layers } from 'lucide-react';

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

  return (
    <div className="mb-8 glass-panel rounded-2xl overflow-hidden transition-all duration-300 hover:border-white/20">
      <div className="p-4 md:p-5 flex items-center justify-between border-b border-white/5 bg-white/5 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-500/10 hidden md:block">
            <Layers size={20} className="text-blue-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white tracking-tight">{title}</h3>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/20">
                {(similarity * 100).toFixed(0)}% Match
              </span>
              <span className="text-xs text-slate-500">{photos.length} photos</span>
            </div>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleSelectAll}
            className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-green-400 transition-colors"
            title="Keep All"
          >
            <CheckCircle2 size={20} />
          </button>
          <button
            onClick={handleDeselectAll}
            className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-red-400 transition-colors"
            title="Discard All"
          >
            <XCircle size={20} />
          </button>
        </div>
      </div>

      <div className="p-4 md:p-6 overflow-x-auto custom-scrollbar">
        <div className="flex space-x-4 pb-2 min-w-min">
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
    <div className="mb-12 animate-fade-in-up">
      <div className="flex items-center justify-between mb-6 px-4 md:px-0">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
            Similar Groups
          </span>
          <span className="text-sm font-normal text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded-md border border-white/5">
            {groups.length} found
          </span>
        </h2>
      </div>

      <div className="space-y-6 px-4 md:px-0">
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