import React from 'react';
import { X, TrendingUp, TrendingDown, User, Users, Utensils, Mountain, Monitor, Palette, Camera } from 'lucide-react';
import { Photo, PhotoCategory, DimensionScore } from '../types';

interface QualityDetailsModalProps {
  photo: Photo;
  onClose: () => void;
}

const categoryIcons: Record<PhotoCategory, React.ReactNode> = {
  face: <User size={18} />,
  group: <Users size={18} />,
  food: <Utensils size={18} />,
  landscape: <Mountain size={18} />,
  screenshot: <Monitor size={18} />,
  drawing: <Palette size={18} />,
  general: <Camera size={18} />,
};

const categoryIconsSmall: Record<PhotoCategory, React.ReactNode> = {
  face: <User size={12} />,
  group: <Users size={12} />,
  food: <Utensils size={12} />,
  landscape: <Mountain size={12} />,
  screenshot: <Monitor size={12} />,
  drawing: <Palette size={12} />,
  general: <Camera size={12} />,
};

const categoryLabels: Record<PhotoCategory, string> = {
  face: 'Portrait',
  group: 'Group Photo',
  food: 'Food',
  landscape: 'Landscape',
  screenshot: 'Screenshot',
  drawing: 'Art / Drawing',
  general: 'General',
};

const formatDimensionName = (name: string): string => {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const DimensionBar: React.FC<{ dimension: DimensionScore; positive: boolean }> = ({ dimension, positive }) => {
  const percentage = Math.round(dimension.score * 100);

  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-300">{formatDimensionName(dimension.name)}</span>
        <span className={positive ? 'text-emerald-400' : 'text-amber-400'}>
          {percentage}%
        </span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${positive ? 'bg-emerald-500' : 'bg-amber-500'}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

const QualityDetailsModal: React.FC<QualityDetailsModalProps> = ({ photo, onClose }) => {
  const breakdown = photo.qualityBreakdown;

  if (!breakdown) {
    return null;
  }

  // Sort dimensions by score for display
  const sortedDimensions = [...breakdown.dimensions].sort((a, b) => b.score - a.score);

  // Top strengths (highest scoring dimensions)
  const strengths = sortedDimensions.filter(d => d.score >= 0.5).slice(0, 5);

  // Areas for improvement (lowest scoring dimensions)
  const improvements = sortedDimensions.filter(d => d.score < 0.5).slice(-5).reverse();

  const qualityColor = photo.quality !== undefined
    ? photo.quality >= 90 ? 'text-emerald-400'
      : photo.quality >= 75 ? 'text-blue-400'
        : photo.quality >= 60 ? 'text-yellow-400'
          : 'text-red-400'
    : 'text-slate-400';

  return (
    <div
      className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 rounded-2xl max-w-md w-full overflow-hidden shadow-2xl border border-white/10"
        onClick={e => e.stopPropagation()}
      >
        {/* Header with image */}
        <div className="relative h-52">
          <img
            src={photo.thumbnailUrl}
            alt={photo.name}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-transparent to-transparent" />

          <button
            onClick={onClose}
            className="absolute top-3 right-3 p-2 bg-black/50 hover:bg-black/70 rounded-full transition-colors"
          >
            <X size={20} className="text-white" />
          </button>

          <div className="absolute bottom-3 left-4">
            <div className={`text-4xl font-bold ${qualityColor}`}>
              {photo.quality}%
            </div>
            <div className="text-sm text-slate-400">Quality Score</div>
          </div>
        </div>

        {/* Category detection */}
        <div className="px-4 py-3 border-b border-white/10">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-slate-800 rounded-lg text-blue-400">
              {categoryIcons[breakdown.detectedCategory]}
            </div>
            <div>
              <div className="font-medium text-white">
                {categoryLabels[breakdown.detectedCategory]}
              </div>
              <div className="text-xs text-slate-500">Detected category</div>
            </div>
          </div>

          {/* All category scores */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(breakdown.categoryConfidences)
              .sort(([, a], [, b]) => b - a)
              .map(([cat, confidence]) => {
                const category = cat as PhotoCategory;
                const isDetected = category === breakdown.detectedCategory;
                const percentage = Math.round(confidence * 100);
                return (
                  <div
                    key={cat}
                    className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs ${
                      isDetected
                        ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    <span className="opacity-70">{categoryIconsSmall[category]}</span>
                    <span>{categoryLabels[category]}</span>
                    <span className={isDetected ? 'text-blue-400 font-medium' : 'text-slate-500'}>
                      {percentage}%
                    </span>
                  </div>
                );
              })}
          </div>
        </div>

        <div className="max-h-80 overflow-y-auto">
          {/* Strengths */}
          {strengths.length > 0 && (
            <div className="p-4 border-b border-white/10">
              <h3 className="flex items-center gap-2 text-emerald-400 font-semibold mb-3">
                <TrendingUp size={16} /> Strengths
              </h3>
              {strengths.map(dim => (
                <DimensionBar key={dim.name} dimension={dim} positive />
              ))}
            </div>
          )}

          {/* Areas for improvement */}
          {improvements.length > 0 && (
            <div className="p-4">
              <h3 className="flex items-center gap-2 text-amber-400 font-semibold mb-3">
                <TrendingDown size={16} /> Could Be Better
              </h3>
              {improvements.map(dim => (
                <DimensionBar key={dim.name} dimension={dim} positive={false} />
              ))}
            </div>
          )}

          {/* If all scores are above 0.5 */}
          {improvements.length === 0 && strengths.length > 0 && (
            <div className="p-4 text-center text-slate-400 text-sm">
              Great photo! All quality dimensions scored well.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default QualityDetailsModal;
