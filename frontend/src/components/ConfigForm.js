import React, { useState, useEffect } from 'react';

const ConfigForm = ({ file, structure, currentConfig, onSave, isConfigured }) => {
  const [tempConfig, setTempConfig] = useState(currentConfig || {});
  const isMessagesFormat = structure?.is_messages_format || false;

  useEffect(() => {
    setTempConfig(currentConfig || {});
  }, [currentConfig]);

  useEffect(() => {
    if (isMessagesFormat && !tempConfig.inputField && !tempConfig.outputField) {
      setTempConfig({ inputField: 'messages', outputField: 'messages' });
    }
  }, [isMessagesFormat, tempConfig.inputField, tempConfig.outputField]);

  const handleSaveConfig = () => {
    if (!tempConfig.inputField || !tempConfig.outputField) {
      alert('Please select both input and output fields');
      return;
    }
    onSave(file.dataset_id, tempConfig);
  };

  return (
    <div className="config-form">
      {structure?.type && (
        <div className="config-field">
          <label>Detected Format:</label>
          <div style={{ fontWeight: 'bold' }}>{String(structure.type).toUpperCase()}</div>
          {isMessagesFormat && (
            <div style={{ marginTop: '5px', fontSize: '0.9em', color: '#666' }}>
              ℹ️ Conversational format detected - will extract user/assistant messages automatically
            </div>
          )}
        </div>
      )}
      
      {isMessagesFormat ? (
        <div className="config-field">
          <label>Messages Field:</label>
          <select
            value={tempConfig.inputField || 'messages'}
            onChange={(e) => setTempConfig({
              inputField: e.target.value,
              outputField: e.target.value
            })}
          >
            <option value="messages">messages (auto-extract user/assistant)</option>
          </select>
        </div>
      ) : (
        <>
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
        </>
      )}
      
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