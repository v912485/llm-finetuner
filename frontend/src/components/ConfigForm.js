import React, { useState, useEffect } from 'react';

const ConfigForm = ({ file, structure, currentConfig, onSave, isConfigured }) => {
  const [tempConfig, setTempConfig] = useState(currentConfig || {});

  useEffect(() => {
    setTempConfig(currentConfig || {});
  }, [currentConfig]);

  const handleSaveConfig = () => {
    if (!tempConfig.inputField || !tempConfig.outputField) {
      alert('Please select both input and output fields');
      return;
    }
    onSave(file.path, tempConfig);
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
          value={tempConfig.outputField || ''}
          onChange={(e) => setTempConfig(prev => ({
            ...prev,
            outputField: e.target.value
          }))}
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