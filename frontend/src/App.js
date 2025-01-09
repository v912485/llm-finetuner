import React, { useState, useEffect } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Chat from './Chat';

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
          onChange={(e) => setTempConfig({
            ...tempConfig,
            inputField: e.target.value
          })}
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
          onChange={(e) => setTempConfig({
            ...tempConfig,
            outputField: e.target.value
          })}
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
          Save Configuration
        </button>
        {isConfigured && (
          <span className="config-saved-status">✓ Configuration Saved</span>
        )}
      </div>
    </div>
  );
};

function App() {
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [finetuningParams, setFinetuningParams] = useState({
    learningRate: 0.0001,
    batchSize: 8,
    epochs: 3,
    validationSplit: 0.2,
  });
  const [loading, setLoading] = useState({});
  const [error, setError] = useState({});
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [fileConfig, setFileConfig] = useState({});
  const [configuredFiles, setConfiguredFiles] = useState([]);
  const [fileStructures, setFileStructures] = useState({});
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [downloadedDatasets, setDownloadedDatasets] = useState([]);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [availableResponse, downloadedResponse] = await Promise.all([
          fetch('http://localhost:5000/api/models/available'),
          fetch('http://localhost:5000/api/models/downloaded')
        ]);

        const availableData = await availableResponse.json();
        const downloadedData = await downloadedResponse.json();

        if (downloadedData.status === 'success') {
          setDownloadedModels(downloadedData.downloaded_models);
          console.log('Downloaded models:', downloadedData.downloaded_models);
        }

        setAvailableModels(availableData);
        
        // Clear selected model if it's not in the downloaded models
        setSelectedModel(prev => {
          if (prev && !downloadedData.downloaded_models.includes(prev)) {
            return null;
          }
          return prev;
        });
        
      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };

    fetchInitialData();
  }, []);

  useEffect(() => {
    let intervalId;
    if (isTraining) {
      intervalId = setInterval(fetchTrainingStatus, 1000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isTraining]);

  useEffect(() => {
    const fetchDownloadedDatasets = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/datasets/downloaded');
        const data = await response.json();
        if (data.status === 'success') {
          setDownloadedDatasets(data.datasets);
        }
      } catch (error) {
        console.error('Error fetching datasets:', error);
      }
    };

    fetchDownloadedDatasets();
  }, []);

  useEffect(() => {
    // Load saved configurations from localStorage
    const savedConfig = localStorage.getItem('fileConfig');
    const savedConfiguredFiles = localStorage.getItem('configuredFiles');
    
    if (savedConfig) {
      setFileConfig(JSON.parse(savedConfig));
    }
    if (savedConfiguredFiles) {
      setConfiguredFiles(JSON.parse(savedConfiguredFiles));
    }
  }, []);

  // Save configurations whenever they change
  useEffect(() => {
    localStorage.setItem('fileConfig', JSON.stringify(fileConfig));
  }, [fileConfig]);

  useEffect(() => {
    localStorage.setItem('configuredFiles', JSON.stringify(configuredFiles));
  }, [configuredFiles]);

  useEffect(() => {
    const fetchConfigurations = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/datasets/configured');
        const data = await response.json();
        if (data.status === 'success') {
          const configs = {};
          const configured = [];
          data.configured_datasets.forEach(item => {
            configs[item.file_path] = item.config;
            configured.push(item.file_path);
          });
          setFileConfig(configs);
          setConfiguredFiles(configured);
        }
      } catch (error) {
        console.error('Error fetching configurations:', error);
      }
    };

    fetchConfigurations();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/available');
      const data = await response.json();
      setAvailableModels(data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchDownloadedModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/downloaded');
      const data = await response.json();
      if (data.status === 'success') {
        setDownloadedModels(data.downloaded_models);
      }
    } catch (error) {
      console.error('Error fetching downloaded models:', error);
    }
  };

  const handleModelDownload = async (modelId) => {
    setLoading(prev => ({ ...prev, [modelId]: true }));
    setError(prev => ({ ...prev, [modelId]: null }));

    try {
      const response = await fetch('http://localhost:5000/api/models/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_id: modelId }),
      });
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || 'Failed to download model');
      }

      if (data.status === 'success') {
        setSelectedModel(modelId);
        await fetchDownloadedModels();
      } else {
        throw new Error(data.message || 'Failed to download model');
      }
    } catch (error) {
      console.error('Error downloading model:', error);
      setError(prev => ({ ...prev, [modelId]: error.message }));
    } finally {
      setLoading(prev => ({ ...prev, [modelId]: false }));
    }
  };

  const handleDatasetUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      console.error('No file selected');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/dataset/prepare', {
        method: 'POST',
        body: formData,
        onUploadProgress: (progressEvent) => {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          setUploadProgress(Math.round(progress));
        },
      });
      const data = await response.json();
      if (data.status === 'success') {
        setUploadedFiles(prev => [...prev, {
          name: file.name,
          path: data.dataset_path,
          size: file.size,
          uploadedAt: new Date().toLocaleString()
        }]);
        setFileStructures(prev => ({
          ...prev,
          [data.dataset_path]: data.file_info.structure
        }));
        console.log('Dataset prepared:', data.dataset_path);
      } else {
        console.error('Error preparing dataset:', data.message);
      }
    } catch (error) {
      console.error('Error uploading dataset:', error);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleFileSelection = async (filePath) => {
    setSelectedFiles(prev => {
      if (prev.includes(filePath)) {
        return prev.filter(f => f !== filePath);
      }
      return [...prev, filePath];
    });

    // Fetch file structure if not already loaded
    if (!fileStructures[filePath]) {
      try {
        const response = await fetch('http://localhost:5000/api/dataset/structure', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ file_path: filePath }),
        });
        
        const data = await response.json();
        if (data.status === 'success') {
          setFileStructures(prev => ({
            ...prev,
            [filePath]: data.structure
          }));
        } else {
          console.error('Error fetching file structure:', data.message);
        }
      } catch (error) {
        console.error('Error fetching file structure:', error);
      }
    }
  };

  const handleConfigUpdate = async (filePath, config) => {
    try {
      const response = await fetch('http://localhost:5000/api/dataset/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_path: filePath,
          config: config
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setFileConfig(prev => ({
          ...prev,
          [filePath]: config
        }));
      } else {
        console.error('Error saving configuration:', data.message);
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
    }
  };

  const handleConfigSave = async (file) => {
    if (!fileConfig[file.path]) {
      alert('Please configure the input and output fields first');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/dataset/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_path: file.path,
          config: fileConfig[file.path]
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setConfiguredFiles(prev => 
          prev.includes(file.path) ? prev : [...prev, file.path]
        );
      } else {
        console.error('Error saving configuration:', data.message);
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
    }
  };

  const renderConfigForm = (file) => {
    const structure = fileStructures[file.path];
    
    if (!structure) {
      return <div>Loading file structure...</div>;
    }

    const currentConfig = fileConfig[file.path] || {};
    const isConfigured = configuredFiles.includes(file.path);

    const handleSaveConfigWrapper = (filePath, config) => {
      handleConfigUpdate(filePath, config);
      handleConfigSave(file);
    };

    return (
      <ConfigForm
        file={file}
        structure={structure}
        currentConfig={currentConfig}
        onSave={handleSaveConfigWrapper}
        isConfigured={isConfigured}
      />
    );
  };

  const getFileType = (filename) => {
    const ext = filename.split('.').pop().toLowerCase();
    return ext;
  };

  const getDefaultConfig = (fileType) => {
    switch (fileType) {
      case 'json':
      case 'jsonl':
        return {
          inputField: 'input',
          outputField: 'output',
          instructionField: 'instruction'
        };
      case 'csv':
        return {
          inputField: 'input',
          outputField: 'output',
          instructionField: 'instruction',
          delimiter: ','
        };
      case 'txt':
        return {
          inputMarker: '### Input:',
          outputMarker: '### Output:',
          instructionMarker: '### Instruction:'
        };
      default:
        return {};
    }
  };

  const renderFileItem = (file, index) => {
    const isConfigured = configuredFiles.includes(file.path);
    const isSelected = selectedFiles.includes(file.path);
    
    return (
      <div key={index}>
        <div className="file-item">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={() => handleFileSelection(file.path)}
          />
          <div className="file-info">
            <span className="file-name">{file.name}</span>
            <span className="file-size">
              {(file.size / 1024).toFixed(2)} KB
            </span>
            <span className="file-date">{file.uploadedAt}</span>
            <button
              className={`config-toggle ${isConfigured ? 'configured' : ''}`}
              onClick={() => {
                if (configuredFiles.includes(file.path)) {
                  setConfiguredFiles(prev => prev.filter(f => f !== file.path));
                } else {
                  handleConfigSave(file);
                }
              }}
            >
              {configuredFiles.includes(file.path) ? 'Edit Config' : 'Save Config'}
            </button>
          </div>
        </div>
        {isSelected && (
          <div className="file-config-wrapper">
            <h4>Configure Training Data Fields</h4>
            <p className="config-help">
              Specify which fields in your data file contain the input text and expected output.
            </p>
            {renderConfigForm(file)}
          </div>
        )}
      </div>
    );
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/training/status');
      const data = await response.json();
      setTrainingStatus(data);
      if (!data.is_training) {
        setIsTraining(false);
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  const startTraining = async () => {
    if (!selectedModel) {
      alert('Please select a model for training');
      return;
    }

    if (selectedFiles.length === 0) {
      alert('Please select at least one dataset by checking the checkbox next to it');
      return;
    }

    // Check if all selected files are configured
    const unconfiguredFiles = selectedFiles.filter(file => !configuredFiles.includes(file));
    if (unconfiguredFiles.length > 0) {
      alert(`Please configure and save the following datasets:\n${
        unconfiguredFiles.map(file => 
          uploadedFiles.find(f => f.path === file)?.name || file
        ).join('\n')
      }`);
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          datasets: selectedFiles,
          dataset_configs: Object.fromEntries(
            selectedFiles.map(file => [file, fileConfig[file]])
          ),
          ...finetuningParams
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setIsTraining(true);
      } else {
        alert(data.message || 'Failed to start training');
      }
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training');
    }
  };

  const handleModelSelect = (modelId) => {
    setSelectedModel(modelId);
  };

  const areSelectedFilesConfigured = () => {
    return selectedFiles.every(file => configuredFiles.includes(file));
  };

  // Add this useEffect to debug selectedModel changes
  useEffect(() => {
    console.log('Selected model changed:', selectedModel);
  }, [selectedModel]);

  return (
    <Router>
      <div className="App">
        <nav className="app-nav">
          <Link to="/">Training</Link>
          <Link to="/chat">Chat</Link>
        </nav>

        <Routes>
          <Route path="/chat" element={<Chat />} />
          <Route path="/" element={
            <>
              <h1>LLM Fine-tuning Interface</h1>
              
              <section className="model-selection">
                <h2>1. Select Model</h2>
                <p className="selection-help">
                  {selectedModel 
                    ? "Selected model will be used for fine-tuning"
                    : "Select a downloaded model to use for fine-tuning"}
                </p>
                {selectedModel && (
                  <p className="selected-model-info">
                    Selected model for training: <strong>{selectedModel}</strong>
                    <button 
                      className="deselect-button"
                      onClick={() => setSelectedModel(null)}
                      title="Clear selection"
                    >
                      ✕
                    </button>
                  </p>
                )}
                <div className="models-grid">
                  {availableModels.map((model) => {
                    const isDownloaded = downloadedModels.includes(model.id);
                    return (
                      <div 
                        key={model.id} 
                        className={`model-card ${isDownloaded ? 'downloaded' : ''} ${model.id === selectedModel ? 'selected' : ''}`}
                      >
                        <div className="model-card-content">
                          <h3>{model.name}</h3>
                          <p>Size: {model.size}</p>
                          {isDownloaded && (
                            <div className="download-status">
                              <span className="status-icon">✓</span>
                              <span>Downloaded</span>
                            </div>
                          )}
                        </div>
                        {isDownloaded ? (
                          <button
                            className={`select-model-button ${model.id === selectedModel ? 'selected' : ''}`}
                            onClick={() => handleModelSelect(model.id)}
                          >
                            {model.id === selectedModel ? 'Selected for Training ✓' : 'Select for Training'}
                          </button>
                        ) : (
                          <button
                            onClick={() => handleModelDownload(model.id)}
                            disabled={loading[model.id]}
                            className={loading[model.id] ? 'loading' : ''}
                          >
                            {loading[model.id] ? 'Downloading...' : 'Download'}
                          </button>
                        )}
                        {error[model.id] && (
                          <p className="error-message">{error[model.id]}</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>

              <section className="dataset-preparation">
                <h2>2. Prepare Dataset</h2>
                <div className="dataset-instructions">
                  <p>To prepare your dataset for training:</p>
                  <ol>
                    <li>Upload your dataset file</li>
                    <li>Select the file using the checkbox</li>
                    <li>Configure the input and output fields</li>
                    <li>Click "Save Config" to confirm the configuration</li>
                  </ol>
                </div>

                <div className="downloaded-datasets">
                  <h3>Available Datasets</h3>
                  <div className="datasets-list">
                    {downloadedDatasets.map((dataset, index) => {
                      const isSelected = selectedFiles.includes(dataset.path);
                      const isConfigured = configuredFiles.includes(dataset.path);
                      return (
                        <div key={index}>
                          <div className={`dataset-item ${isSelected ? 'selected' : ''} ${isConfigured ? 'configured' : ''}`}>
                            <div className="dataset-info">
                              <span className="dataset-name">{dataset.name}</span>
                              <span className="dataset-size">
                                {(dataset.size / 1024).toFixed(2)} KB
                              </span>
                              <span className="dataset-date">
                                {new Date(dataset.uploadedAt).toLocaleString()}
                              </span>
                              <div className="dataset-status">
                                {isConfigured ? (
                                  <span className="status-badge configured">Configured ✓</span>
                                ) : (
                                  <span className="status-badge unconfigured">Not Configured</span>
                                )}
                              </div>
                            </div>
                            <div className="dataset-actions">
                              <button
                                className="dataset-select-btn"
                                onClick={() => handleFileSelection(dataset.path)}
                              >
                                {isSelected ? 'Selected ✓' : 'Select'}
                              </button>
                              <button
                                className={`config-toggle ${isConfigured ? 'configured' : ''}`}
                                onClick={() => {
                                  if (configuredFiles.includes(dataset.path)) {
                                    setConfiguredFiles(prev => prev.filter(f => f !== dataset.path));
                                  } else {
                                    handleConfigSave({ path: dataset.path, name: dataset.name });
                                  }
                                }}
                              >
                                {isConfigured ? 'Edit Config' : 'Configure'}
                              </button>
                            </div>
                          </div>
                          {isSelected && (
                            <div className="file-config-wrapper">
                              <h4>Configure Training Data Fields</h4>
                              <p className="config-help">
                                Specify which fields in your data file contain the input text and expected output.
                              </p>
                              {renderConfigForm({ path: dataset.path, name: dataset.name })}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="upload-section">
                  <h3>Upload New Dataset</h3>
                  <div className="upload-container">
                    <input
                      type="file"
                      accept=".json,.jsonl,.csv,.txt"
                      onChange={handleDatasetUpload}
                      disabled={isUploading}
                    />
                    <button 
                      className="upload-button"
                      onClick={() => document.querySelector('input[type="file"]').click()}
                      disabled={isUploading}
                    >
                      {isUploading ? 'Uploading...' : 'Upload Dataset'}
                    </button>
                  </div>

                  {isUploading && (
                    <div className="progress-bar-container">
                      <div 
                        className="progress-bar" 
                        style={{ width: `${uploadProgress}%` }}
                      />
                      <span>{uploadProgress}%</span>
                    </div>
                  )}
                </div>

                {uploadedFiles.length > 0 && (
                  <div className="uploaded-files">
                    <h3>Uploaded Datasets</h3>
                    <div className="files-list">
                      {uploadedFiles.map((file, index) => {
                        const isSelected = selectedFiles.includes(file.path);
                        const isConfigured = configuredFiles.includes(file.path);
                        return (
                          <div key={index}>
                            <div className={`file-item ${isSelected ? 'selected' : ''} ${isConfigured ? 'configured' : ''}`}>
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => handleFileSelection(file.path)}
                              />
                              <div className="file-info">
                                <span className="file-name">{file.name}</span>
                                <span className="file-size">
                                  {(file.size / 1024).toFixed(2)} KB
                                </span>
                                <span className="file-date">{file.uploadedAt}</span>
                                <div className="file-status">
                                  {isConfigured ? (
                                    <span className="status-badge configured">Configured ✓</span>
                                  ) : (
                                    <span className="status-badge unconfigured">Not Configured</span>
                                  )}
                                </div>
                                <button
                                  className={`config-toggle ${isConfigured ? 'configured' : ''}`}
                                  onClick={() => {
                                    if (configuredFiles.includes(file.path)) {
                                      setConfiguredFiles(prev => prev.filter(f => f !== file.path));
                                    } else {
                                      handleConfigSave(file);
                                    }
                                  }}
                                >
                                  {isConfigured ? 'Edit Config' : 'Save Config'}
                                </button>
                              </div>
                            </div>
                            {isSelected && (
                              <div className="file-config-wrapper">
                                <h4>Configure Training Data Fields</h4>
                                <p className="config-help">
                                  Specify which fields in your data file contain the input text and expected output.
                                </p>
                                {renderConfigForm(file)}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </section>

              <section className="training-params">
                <h2>3. Configure Training Parameters</h2>
                <div className="params-form">
                  <div className="param-group">
                    <label>Learning Rate:</label>
                    <input
                      type="number"
                      value={finetuningParams.learningRate}
                      onChange={(e) =>
                        setFinetuningParams({
                          ...finetuningParams,
                          learningRate: parseFloat(e.target.value),
                        })
                      }
                      step="0.0001"
                    />
                  </div>
                  <div className="param-group">
                    <label>Batch Size:</label>
                    <input
                      type="number"
                      value={finetuningParams.batchSize}
                      onChange={(e) =>
                        setFinetuningParams({
                          ...finetuningParams,
                          batchSize: parseInt(e.target.value),
                        })
                      }
                    />
                  </div>
                  <div className="param-group">
                    <label>Epochs:</label>
                    <input
                      type="number"
                      value={finetuningParams.epochs}
                      onChange={(e) =>
                        setFinetuningParams({
                          ...finetuningParams,
                          epochs: parseInt(e.target.value),
                        })
                      }
                    />
                  </div>
                  <div className="param-group">
                    <label>Validation Split:</label>
                    <input
                      type="number"
                      value={finetuningParams.validationSplit}
                      onChange={(e) =>
                        setFinetuningParams({
                          ...finetuningParams,
                          validationSplit: parseFloat(e.target.value),
                        })
                      }
                      step="0.05"
                      min="0.1"
                      max="0.5"
                    />
                    <span className="param-help">
                      Portion of data used for validation (10-50%)
                    </span>
                  </div>
                </div>
              </section>

              <section className="training-control">
                <h2>4. Training Control</h2>
                <button
                  className="start-training-button"
                  onClick={startTraining}
                  disabled={isTraining || !selectedModel || selectedFiles.length === 0 || !areSelectedFilesConfigured()}
                >
                  {isTraining ? 'Training in Progress...' : 
                   !selectedModel ? 'Select a Model First' :
                   selectedFiles.length === 0 ? 'Select Dataset(s)' :
                   !areSelectedFilesConfigured() ? 'Configure Selected Dataset(s)' :
                   'Start Training'}
                </button>

                {trainingStatus && (
                  <div className="training-status">
                    <h3>Training Status</h3>
                    <div className="status-details">
                      <div className="status-item">
                        <span>Progress:</span>
                        <div className="progress-bar-container">
                          <div 
                            className="progress-bar" 
                            style={{ width: `${trainingStatus.progress}%` }}
                          />
                          <span>{trainingStatus.progress}%</span>
                        </div>
                      </div>
                      <div className="status-item">
                        <span>Epoch:</span>
                        <span>{trainingStatus.current_epoch} / {trainingStatus.total_epochs}</span>
                      </div>
                      {trainingStatus.loss && (
                        <div className="status-item">
                          <span>Loss:</span>
                          <span>{trainingStatus.loss.toFixed(4)}</span>
                        </div>
                      )}
                      {trainingStatus.error && (
                        <div className="status-error">
                          Error: {trainingStatus.error}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </section>
            </>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 