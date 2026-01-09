import React, { useState, useCallback } from 'react';
import { useAppContext } from '../context/AppContext';
import './ModelSelector.css';

const ModelSelector = ({ 
  selectedModel, 
  onModelSelect, 
  label = 'Select Model',
  includeBaseModels = true,
  includeSavedModels = true,
  className = ''
}) => {
  const { downloadedModels, savedModels, fetchSavedModels, fetchDownloadedModels } = useAppContext();
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Add debounce to prevent multiple rapid calls
  const handleRefresh = useCallback(async () => {
    if (isRefreshing) return;
    
    setIsRefreshing(true);
    try {
      const promises = [];
      if (includeSavedModels) {
        promises.push(fetchSavedModels());
      }
      if (includeBaseModels) {
        promises.push(fetchDownloadedModels());
      }
      await Promise.all(promises);
    } catch (error) {
      console.error('Error refreshing models:', error);
    } finally {
      setIsRefreshing(false);
    }
  }, [includeBaseModels, includeSavedModels, fetchSavedModels, fetchDownloadedModels, isRefreshing]);

  return (
    <div className={`model-selector ${className}`}>
      <div className="model-selector-header">
        <label>{label}</label>
        <button 
          className={`refresh-button ${isRefreshing ? 'refreshing' : ''}`}
          onClick={handleRefresh}
          title="Refresh model list"
          disabled={isRefreshing}
        >
          {isRefreshing ? '⟳' : '↻'}
        </button>
      </div>
      <select 
        value={selectedModel || ''} 
        onChange={(e) => onModelSelect(e.target.value)}
        className="model-select"
      >
        <option value="">-- Select a model --</option>
        
        {includeBaseModels && downloadedModels.length > 0 && (
          <>
            <optgroup label="Base Models">
              {downloadedModels.map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </optgroup>
          </>
        )}
        
        {includeSavedModels && savedModels.length > 0 && (
          <>
            <optgroup label="Fine-tuned Models">
              {savedModels.map(model => (
                <option key={model.name} value={model.path}>
                  {model.name}
                </option>
              ))}
            </optgroup>
          </>
        )}
      </select>
    </div>
  );
};

export default ModelSelector; 