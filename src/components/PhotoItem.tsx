import React from 'react';
import { Award, CheckCircle2, Info } from 'lucide-react';

interface PhotoItemProps {
  id: string;
  url: string;
  thumbnailUrl: string;
  quality: number;
  selected: boolean;
  isBest?: boolean;
  onSelect: () => void;
  onShowDetails?: () => void;
}

const PhotoItem: React.FC<PhotoItemProps> = ({
  thumbnailUrl,
  quality,
  selected,
  isBest = false,
  onSelect,
  onShowDetails,
}) => {
  return (
    <div
      className={`relative shrink-0 w-48 h-48 group rounded-xl overflow-hidden cursor-pointer transition-all duration-300 ${selected
          ? 'scale-95 selected-ring shadow-lg shadow-blue-500/20'
          : 'card-hover border border-white/5'
        }`}
      onClick={onSelect}
    >
      <img
        src={thumbnailUrl}
        alt="Photo"
        className={`w-full h-full object-cover transition-all duration-500 ${selected ? 'brightness-75 scale-100' : 'brightness-90 group-hover:brightness-100 group-hover:scale-110'
          }`}
      />

      <div
        className={`absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/20 transition-opacity duration-300 ${selected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
          }`}
      />

      {selected && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <CheckCircle2 size={40} className="text-blue-500 animate-scale-in drop-shadow-[0_2px_8px_rgba(0,0,0,0.5)] bg-white/10 rounded-full backdrop-blur-sm" />
        </div>
      )}

      {isBest && (
        <div className="absolute top-2 left-2 z-10 px-2 py-1 bg-yellow-500/90 backdrop-blur-sm rounded-lg flex items-center gap-1.5 shadow-lg shadow-yellow-500/20">
          <Award size={14} className="text-white" />
          <span className="text-[10px] font-bold text-white uppercase tracking-wider">Best Pick</span>
        </div>
      )}

      {/* Info button - shows quality details */}
      {onShowDetails && (
        <button
          className="absolute top-2 right-2 z-10 p-1.5 bg-black/50 hover:bg-black/70 rounded-full opacity-0 group-hover:opacity-100 transition-all duration-200"
          onClick={(e) => {
            e.stopPropagation();
            onShowDetails();
          }}
        >
          <Info size={16} className="text-white" />
        </button>
      )}

      {/* Quality Badge Overlay */}
      <div className={`absolute bottom-2 right-2 flex items-center gap-1.5 px-2 py-1 rounded-md bg-black/60 backdrop-blur-md border border-white/10 transition-transform duration-300 ${selected || isBest ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0 group-hover:translate-y-0 group-hover:opacity-100'
        }`}>
        <div
          className={`w-2 h-2 rounded-full ${quality >= 90 ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' :
              quality >= 75 ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]' :
                quality >= 60 ? 'bg-yellow-500' :
                  'bg-red-500'
            }`}
        />
        <span className="text-xs font-medium text-white">{quality}%</span>
      </div>
    </div>
  );
};

export default PhotoItem;