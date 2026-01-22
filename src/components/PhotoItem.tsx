import React from 'react';
import { Award } from 'lucide-react';

interface PhotoItemProps {
  id: string;
  url: string;
  thumbnailUrl: string;
  quality: number;
  selected: boolean;
  isBest?: boolean;
  onSelect: () => void;
}

const PhotoItem: React.FC<PhotoItemProps> = ({
  thumbnailUrl,
  quality,
  selected,
  isBest = false,
  onSelect,
}) => {
  return (
    <div
      className={`relative flex-shrink-0 w-48 h-48 group rounded-lg overflow-hidden cursor-pointer transition-all duration-300 ${selected
          ? 'ring-4 ring-blue-500 scale-95 shadow-lg shadow-blue-500/40 selected-photo-pulse'
          : 'hover:scale-105 border-2 border-transparent'
        }`}
      onClick={onSelect}
    >
      <img
        src={thumbnailUrl}
        alt="Photo"
        className={`w-full h-full object-cover transition-all duration-300 ${selected ? 'brightness-75' : 'group-hover:brightness-90'
          }`}
      />

      <div
        className={`absolute inset-0 bg-gradient-to-t from-black/70 to-transparent transition-opacity duration-300 ${selected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
          }`}
      />

      {selected && (
        <>
          <div className="absolute inset-0 border-4 border-blue-500 rounded-lg z-10 opacity-70"></div>
        </>
      )}

      {isBest && (
        <div className="absolute top-2 left-2 z-10">
          <Award size={24} className="text-yellow-400" />
        </div>
      )}

      <div className="absolute bottom-0 left-0 right-0 p-2 text-white bg-black bg-opacity-50">
        <div className="flex justify-between items-center">
          <div className="text-xs">Quality</div>
          <div className="text-sm font-medium">{quality}%</div>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-1.5 mt-1">
          <div
            className={`h-1.5 rounded-full ${quality >= 90 ? 'bg-green-500' :
                quality >= 75 ? 'bg-blue-500' :
                  quality >= 60 ? 'bg-yellow-500' :
                    'bg-red-500'
              }`}
            style={{ width: `${quality}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default PhotoItem;