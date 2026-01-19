import React, { useState, useEffect } from 'react';

const ConfigForm = ({ file, structure, currentConfig, onSave, isConfigured }) => {
  const [tempConfig, setTempConfig] = useState(currentConfig || {});

  useEffect(() => {
    const nextConfig = currentConfig || {};
    const hasSingleField = Array.isArray(structure?.fields) && structure.fields.length === 1;
    if (hasSingleField) {
      const onlyField = structure.fields[0];
      setTempConfig({
        inputField: nextConfig.inputField || onlyField,
        outputField: nextConfig.outputField || onlyField
      });
    } else {
      setTempConfig(nextConfig);
    }
  }, [currentConfig, structure]);

  const handleSaveConfig = () => {
    const hasSingleField = Array.isArray(structure?.fields) && structure.fields.length === 1;
    const onlyField = hasSingleField ? structure.fields[0] : '';
    const nextConfig = {
      ...tempConfig,
      inputField: tempConfig.inputField || (hasSingleField ? onlyField : ''),
      outputField: tempConfig.outputField || (hasSingleField ? (tempConfig.inputField || onlyField) : '')
    };
    if (!nextConfig.inputField || !nextConfig.outputField) {
      alert('Please select both input and output fields');
      return;
    }
    onSave(file.dataset_id, nextConfig);
  };

  return (
    <div className="config-form">
      <div className="config-field">
        <label>Input Field:</label>
        <select
          value={tempConfig.inputField || ''}
          onChange={(e) => setTempConfig(prev => ({
            ...prev,
            inputField: e.target.value
          }))}
        >
          <option value="">Select input field</option>
          {structure.fields.map(field => (
            <option key={field} value={field}>{field}</option>
          ))}
        </select>
      </div>
      <div className="config-field">
        <label>Output Field:</label>
        <select
          value={
            tempConfig.outputField ||
            (Array.isArray(structure?.fields) && structure.fields.length === 1 ? structure.fields[0] : '')
          }
          onChange={(e) => setTempConfig(prev => ({
            ...prev,
            outputField: e.target.value
          }))}
          disabled={Array.isArray(structure?.fields) && structure.fields.length === 1}
        >
          <option value="">Select output field</option>
          {structure.fields.map(field => (
            <option key={field} value={field}>{field}</option>
          ))}
        </select>
      </div>
      <div className="config-actions">
        <button 
          className="save-config-btn"
          onClick={handleSaveConfig}
          disabled={!tempConfig.inputField || !tempConfig.outputField}
        >
          {isConfigured ? 'Update Configuration' : 'Save Configuration'}
        </button>
      </div>
    </div>
  );
};

export default ConfigForm; 