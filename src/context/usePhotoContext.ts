import { useContext } from 'react';
import { PhotoContext } from './PhotoContextDef';

export function usePhotoContext() {
    const context = useContext(PhotoContext);
    if (context === undefined) {
        throw new Error('usePhotoContext must be used within a PhotoProvider');
    }
    return context;
}
