import React from 'react';
import { useAppContext } from '../context/AppContext';
import './DatasetSelector.css';

const DatasetSelector = ({ 
  selectedDatasets, 
  onDatasetSelect, 
  label = 'Select Datasets',
  multiple = true,
  onlyConfigured = false,
  className = ''
}) => {
  const { downloadedDatasets, configuredDatasets } = useAppContext();
  
  const datasets = onlyConfigured ? configuredDatasets : downloadedDatasets;
  
  const handleChange = (e) => {
    if (multiple) {
      const options = e.target.options;
      const selectedValues = [];
      for (let i = 0; i < options.length; i++) {
        if (options[i].selected) {
          selectedValues.push(options[i].value);
        }
      }
      onDatasetSelect(selectedValues);
    } else {
      onDatasetSelect(e.target.value);
    }
  };

  return (
    <div className={`dataset-selector ${className}`}>
      <label>{label}</label>
      <select 
        value={selectedDatasets || (multiple ? [] : '')} 
        onChange={handleChange}
        className="dataset-select"
        multiple={multiple}
      >
        {!multiple && <option value="">-- Select a dataset --</option>}
        
        {datasets.map(dataset => (
          <option key={dataset.path} value={dataset.path}>
            {dataset.name || dataset.path.split('/').pop()}
          </option>
        ))}
      </select>
    </div>
  );
};

export default DatasetSelector; 