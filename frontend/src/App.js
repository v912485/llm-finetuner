import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Chat from './Chat';
import Settings from './Settings';
import TrainingGraph from './components/TrainingGraph';
import ConfigForm from './components/ConfigForm';
import apiConfig from './config';

function App() {
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [finetuningParams, setFinetuningParams] = useState({
    learningRate: 0.0001,
    batchSize: 8,
    epochs: 3,
    validationSplit: 0.2,
  });
  const [loading, setLoading] = useState({});
  const [error, setError] = useState({});
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [fileConfig, setFileConfig] = useState({});
  const [configuredFiles, setConfiguredFiles] = useState([]);
  const [fileStructures, setFileStructures] = useState({});
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [dismissedTrainingError, setDismissedTrainingError] = useState(null);
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [downloadedDatasets, setDownloadedDatasets] = useState([]);
  const [trainingMethod, setTrainingMethod] = useState('standard');
  const [config, setConfig] = useState(null);
  const [saveName, setSaveName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [datasetConfigs, setDatasetConfigs] = useState({});
  const [showAddModel, setShowAddModel] = useState(false);
  const [newModelId, setNewModelId] = useState('');
  const [newModelName, setNewModelName] = useState('');

  const getAuthHeaders = useCallback((extraHeaders = {}) => {
    const token = localStorage.getItem('adminToken');
    return {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...extraHeaders
    };
  }, []);

  const fetchFileStructure = useCallback(async (datasetId) => {
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.structure}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({ dataset_id: datasetId }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setFileStructures(prev => ({
          ...prev,
          [datasetId]: data.structure
        }));
        console.log('Loaded structure for:', datasetId, data.structure);
      } else {
        console.error('Error loading structure:', data.message);
      }
    } catch (error) {
      console.error('Error fetching file structure:', error);
    }
  }, [getAuthHeaders]);

  const fetchConfiguredDatasets = useCallback(async () => {
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.configured}`, {
        headers: getAuthHeaders()
      });
      const data = await response.json();
      if (data.status === 'success') {
        const configs = {};
        data.configured_datasets.forEach(dataset => {
          configs[dataset.dataset_id] = {
            inputField: dataset.config.input_field,
            outputField: dataset.config.output_field
          };
        });
        setDatasetConfigs(configs);
        console.log('Loaded dataset configs:', configs);
      }
    } catch (error) {
      console.error('Error fetching dataset configurations:', error);
    }
  }, [getAuthHeaders]);

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.training.status}`, {
        headers: getAuthHeaders()
      });
      const data = await response.json();
      if (data.status === 'success') {
        setTrainingStatus(data);
        setIsTraining(Boolean(data.is_training));
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  }, [getAuthHeaders]);

  useEffect(() => {
    fetchTrainingStatus();
  }, [fetchTrainingStatus]);

  useEffect(() => {
    const trainingError = trainingStatus?.error || null;
    if (!trainingError) {
      if (dismissedTrainingError !== null) setDismissedTrainingError(null);
      return;
    }
    if (dismissedTrainingError && dismissedTrainingError !== trainingError) {
      setDismissedTrainingError(null);
    }
  }, [trainingStatus, dismissedTrainingError]);

  useEffect(() => {
    fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.config}`, {
      headers: getAuthHeaders()
    })
      .then(res => res.json())
      .then(data => setConfig(data));
  }, [getAuthHeaders]);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [availableResponse, downloadedResponse] = await Promise.all([
          fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.available}`, { headers: getAuthHeaders() }),
          fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.downloaded}`, { headers: getAuthHeaders() })
        ]);

        const availableData = await availableResponse.json();
        const downloadedData = await downloadedResponse.json();

        if (!availableResponse.ok) {
          console.error('Error fetching available models:', availableData);
          setAvailableModels([]);
        } else if (Array.isArray(availableData)) {
          setAvailableModels(availableData);
        } else if (availableData && Array.isArray(availableData.models)) {
          setAvailableModels(availableData.models);
        } else {
          console.error('Unexpected available models response shape:', availableData);
          setAvailableModels([]);
        }

        if (downloadedData.status === 'success') {
          setDownloadedModels(downloadedData.downloaded_models);
          console.log('Downloaded models:', downloadedData.downloaded_models);
        }
        
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
  }, [getAuthHeaders]);

  useEffect(() => {
    let intervalId;
    if (isTraining) {
      intervalId = setInterval(fetchTrainingStatus, 1000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isTraining, fetchTrainingStatus]);

  useEffect(() => {
    const fetchDownloadedDatasets = async () => {
      try {
        const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.downloaded}`, {
          headers: getAuthHeaders()
        });
        const data = await response.json();
        if (data.status === 'success') {
          setDownloadedDatasets(data.datasets);
          // Fetch structure for each dataset
          data.datasets.forEach(dataset => {
            fetchFileStructure(dataset.dataset_id);
          });
        }
      } catch (error) {
        console.error('Error fetching datasets:', error);
      }
    };

    fetchDownloadedDatasets();
  }, [fetchFileStructure, getAuthHeaders]);

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
        const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.configured}`, {
          headers: getAuthHeaders()
        });
        const data = await response.json();
        if (data.status === 'success') {
          const configs = {};
          const configuredIds = [];
          
          data.configured_datasets.forEach(dataset => {
            configs[dataset.dataset_id] = {
              inputField: dataset.config.input_field,
              outputField: dataset.config.output_field
            };
            configuredIds.push(dataset.dataset_id);
          });
          
          setFileConfig(configs);
          setConfiguredFiles(configuredIds);
          console.log('Loaded configurations:', configs);
        }
      } catch (error) {
        console.error('Error fetching configurations:', error);
      }
    };

    fetchConfigurations();
  }, [getAuthHeaders]);

  useEffect(() => {
    fetchConfiguredDatasets();
  }, [fetchConfiguredDatasets]);

  const handleModelDownload = async (modelId) => {
    setLoading(prev => ({ ...prev, [modelId]: true }));
    setError(prev => ({ ...prev, [modelId]: null }));

    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.download}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
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
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.prepare}`);

        const authHeaders = getAuthHeaders();
        Object.entries(authHeaders).forEach(([key, value]) => {
          xhr.setRequestHeader(key, value);
        });

        xhr.upload.onprogress = (progressEvent) => {
          if (!progressEvent.lengthComputable) return;
          const percentComplete = Math.round((progressEvent.loaded / progressEvent.total) * 100);
          setUploadProgress(percentComplete);
        };

        xhr.onload = () => {
          try {
            const isSuccess = xhr.status >= 200 && xhr.status < 300;
            const parsed = xhr.responseText ? JSON.parse(xhr.responseText) : null;
            if (!isSuccess) {
              reject(new Error((parsed && parsed.message) || 'Failed to upload dataset'));
              return;
            }
            setUploadProgress(100);
            resolve(parsed);
          } catch (e) {
            reject(new Error('Failed to parse server response'));
          }
        };

        xhr.onerror = () => reject(new Error('Network error while uploading dataset'));
        xhr.send(formData);
      });
      if (data.status === 'success') {
        setFileStructures(prev => ({
          ...prev,
          [data.dataset_id]: data.file_info.structure
        }));
        console.log('Dataset prepared:', data.dataset_id);

        // Refresh downloaded datasets so the main list is up to date
        try {
          const refreshed = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.downloaded}`, {
            headers: getAuthHeaders()
          });
          const refreshedData = await refreshed.json();
          if (refreshedData.status === 'success') {
            setDownloadedDatasets(refreshedData.datasets);
          }
        } catch (e) {
          // ignore refresh errors
        }
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

  const handleFileSelect = (file) => {
    console.log('Selected file:', file);
    
    if (selectedFiles.includes(file.dataset_id)) {
      setSelectedFiles(prev => prev.filter(f => f !== file.dataset_id));
    } else {
      setSelectedFiles(prev => [...prev, file.dataset_id]);
      // Fetch structure if not already loaded
      if (!fileStructures[file.dataset_id]) {
        fetchFileStructure(file.dataset_id);
      }
    }
  };

  const handleConfigSave = async (datasetId, config) => {
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.config}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({
          dataset_id: datasetId,
          config: {
            inputField: config.inputField,
            outputField: config.outputField
          }
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setDatasetConfigs(prev => ({
          ...prev,
          [datasetId]: config
        }));
        setConfiguredFiles(prev => (prev.includes(datasetId) ? prev : [...prev, datasetId]));
        console.log('Configuration saved successfully');
      } else {
        throw new Error(data.message || 'Failed to save configuration');
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      alert(`Error saving configuration: ${error.message}`);
    }
  };

  const renderConfigForm = (file) => {
    const structure = fileStructures[file.dataset_id];
    const currentConfig = fileConfig[file.dataset_id];
    const isConfigured = configuredFiles.includes(file.dataset_id);
    
    console.log('Rendering config form:', {
      dataset_id: file.dataset_id,
      structure,
      currentConfig,
      isConfigured
    });
    
    if (!structure) {
      return <div>Loading structure...</div>;
    }
    
    return (
      <ConfigForm
        file={file}
        structure={structure}
        currentConfig={currentConfig}
        onSave={handleConfigSave}
        isConfigured={isConfigured}
      />
    );
  };

  const fetchDownloadedModels = async () => {
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.downloaded}`, {
        headers: getAuthHeaders()
      });
      const data = await response.json();
      if (data.status === 'success') {
        setDownloadedModels(data.downloaded_models);
      }
    } catch (error) {
      console.error('Error fetching downloaded models:', error);
    }
  };

  // fetchTrainingStatus is defined above via useCallback

  const startTraining = async () => {
    if (!selectedModel || selectedFiles.length === 0) {
      return;
    }

    setDismissedTrainingError(null);
    setTrainingStatus(null);
    setIsTraining(true);
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.training.start}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({
          model_id: selectedModel,
          datasets: selectedFiles,
          params: {
            learningRate: finetuningParams.learningRate,
            batchSize: finetuningParams.batchSize,
            epochs: finetuningParams.epochs,
            validationSplit: finetuningParams.validationSplit,
            training_method: trainingMethod,
            force_single_gpu: true,
            gpu_index: 0
          }
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        console.log('Training started successfully');
        fetchTrainingStatus();
      } else {
        console.error('Error starting training:', data.message);
        setIsTraining(false);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      setIsTraining(false);
    }
  };

  const handleModelSelect = (modelId) => {
    setSelectedModel(modelId);
  };

  const areSelectedFilesConfigured = () => {
    return selectedFiles.every(file => datasetConfigs[file]);
  };

  // Add this useEffect to debug selectedModel changes
  useEffect(() => {
    console.log('Selected model changed:', selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    const getSelectedModelConfig = () => {
      return availableModels.find(model => model.id === selectedModel);
    };
    
    const selectedModelConfig = getSelectedModelConfig();
    if (!selectedModelConfig?.supports_lora && trainingMethod !== 'standard') {
      setTrainingMethod('standard');
    }
  }, [selectedModel, trainingMethod, availableModels]);

  const getSelectedModelConfig = () => {
    return availableModels.find(model => model.id === selectedModel);
  };

  const handleSaveModel = async () => {
    if (!saveName.trim()) {
      alert('Please enter a name for the saved model');
      return;
    }

    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.training.save}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({
          model_id: selectedModel,
          save_name: saveName,
          description: 'Fine-tuned model'
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        alert('Model saved successfully!');
        setShowSaveDialog(false);
        setSaveName('');
        
        // Fetch saved models to update the dropdown
        try {
          const savedResponse = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.saved}`, {
            headers: getAuthHeaders()
          });
          const savedData = await savedResponse.json();
          if (savedData.status === 'success') {
            console.log('Updated saved models:', savedData.saved_models);
          }
        } catch (error) {
          console.error('Error fetching saved models after save:', error);
        }
      } else {
        alert(`Error saving model: ${data.message}`);
      }
    } catch (error) {
      console.error('Error saving model:', error);
      alert('Error saving model');
    }
  };

  // fetchFileStructure is defined above via useCallback

  // Add cancel training function
  const cancelTraining = async () => {
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.training.cancel}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        }
      });

      const data = await response.json();
      if (data.status === 'success') {
        console.log('Training cancelled');
      } else {
        console.error('Error cancelling training:', data.message);
      }
    } catch (error) {
      console.error('Error cancelling training:', error);
    }
  };

  const handleAddModel = async () => {
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.add}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({ 
          model_id: newModelId,
          display_name: newModelName 
        }),
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setAvailableModels(prev => [...prev, data.model]);
        setShowAddModel(false);
        setNewModelId('');
        setNewModelName('');
      } else {
        alert(data.message || 'Failed to add model');
      }
    } catch (error) {
      console.error('Error adding model:', error);
      const message =
        (error && typeof error.message === 'string' && error.message) ||
        'Network error';
      alert(
        `Error adding model: ${message}\n\n` +
        `If this is a network/CORS issue, ensure the backend is running and CORS allows this origin.`
      );
    }
  };

  const handleDeleteModel = async (modelId) => {
    if (!window.confirm('Are you sure you want to delete this model?')) return;
    
    try {
      const encodedModelId = encodeURIComponent(modelId);
      const response = await fetch(
        `${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.delete}/${encodedModelId}`,
        { method: 'DELETE', headers: getAuthHeaders() }
      );
      
      const data = await response.json();
      if (data.status === 'success') {
        setAvailableModels(prev => prev.filter(m => m.id !== modelId));
      } else {
        alert(data.message || 'Failed to delete model');
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      alert('Error deleting model');
    }
  };

  return (
    <Router>
      <div className="App">
        <nav className="app-nav">
          <Link to="/">Home</Link>
          <Link to="/chat">Chat</Link>
          <Link to="/settings">Settings</Link>
        </nav>

        <Routes>
          <Route path="/chat" element={<Chat />} />
          <Route path="/" element={
            <>
              <h1>LLM Fine-tuning Interface</h1>
              
              <section className="model-selection">
                <div className="section-header">
                  <h2>1. Select Model</h2>
                </div>
                <p className="selection-help">
                  {selectedModel 
                    ? "Selected model will be used for fine-tuning"
                    : "Select a downloaded model to use for fine-tuning"}
                </p>
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
                          <div className="model-meta">
                            <span className={`size-badge ${model.size_category?.toLowerCase() || 'medium'}`}>
                              {(model.size_category || 'MEDIUM').toUpperCase()}
                            </span>
                            <span>Parameters: {model.parameters || 'Unknown'}</span>
                            <span>Size: {model.storage_size || 'Unknown'}</span>
                          </div>
                        </div>
                        <div className="model-card-actions">
                          {isDownloaded ? (
                            <>
                              <button
                                className={`select-model-button ${model.id === selectedModel ? 'selected' : ''}`}
                                onClick={() => handleModelSelect(model.id)}
                              >
                                {model.id === selectedModel ? 'Selected for Training ✓' : 'Select for Training'}
                              </button>
                              {model.custom && (
                                <button 
                                  className="delete-model-button"
                                  onClick={() => handleDeleteModel(model.id)}
                                >
                                  Delete
                                </button>
                              )}
                            </>
                          ) : (
                            <>
                              <button
                                onClick={() => handleModelDownload(model.id)}
                                disabled={loading[model.id]}
                                className={loading[model.id] ? 'loading' : ''}
                              >
                                {loading[model.id] ? 'Downloading...' : 'Download'}
                              </button>
                              {model.custom && (
                                <button 
                                  className="delete-model-button"
                                  onClick={() => handleDeleteModel(model.id)}
                                >
                                  Delete
                                </button>
                              )}
                            </>
                          )}
                        </div>
                        {error[model.id] && (
                          <p className="error-message">{error[model.id]}</p>
                        )}
                      </div>
                    );
                  })}
                </div>
                <div className="add-model-container">
                  <button 
                    className="add-model-button"
                    onClick={() => setShowAddModel(true)}
                  >
                    Add New Model
                  </button>
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
                      const isSelected = selectedFiles.includes(dataset.dataset_id);
                      const isConfigured = configuredFiles.includes(dataset.dataset_id);
                      return (
                        <div key={dataset.dataset_id || index}>
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
                                onClick={() => handleFileSelect(dataset)}
                              >
                                {isSelected ? 'Selected ✓' : 'Select'}
                              </button>
                              <button
                                className={`config-toggle ${isConfigured ? 'configured' : ''}`}
                                onClick={() => {
                                  if (!isSelected) handleFileSelect(dataset);
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
                              {renderConfigForm({ dataset_id: dataset.dataset_id, name: dataset.name })}
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

              </section>

              <section className="training-params">
                <h2>3. Configure Training Parameters</h2>
                <div className="params-form">
                  <div className="param-group">
                    <label>Training Method:</label>
                    <select
                      value={trainingMethod}
                      onChange={(e) => setTrainingMethod(e.target.value)}
                    >
                      <option value="standard">Full Fine-tuning</option>
                      {selectedModel && getSelectedModelConfig()?.supports_lora && (
                        <>
                          <option value="lora">LoRA (Memory Efficient)</option>
                          <option value="qlora">QLoRA (Very Memory Efficient)</option>
                        </>
                      )}
                    </select>
                    <span className="param-help">
                      {config?.training_methods?.[trainingMethod]?.description || ''}
                    </span>
                  </div>
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
                {!isTraining ? (
                  <button
                    className="start-training-button"
                    onClick={startTraining}
                    disabled={!selectedModel || selectedFiles.length === 0 || !areSelectedFilesConfigured()}
                  >
                    {!selectedModel ? 'Select a Model First' :
                     selectedFiles.length === 0 ? 'Select Dataset(s)' :
                     !areSelectedFilesConfigured() ? 'Configure Selected Dataset(s)' :
                     'Start Training'}
                  </button>
                ) : (
                  <button
                    className="cancel-training-button"
                    onClick={cancelTraining}
                  >
                    Cancel Training
                  </button>
                )}

                {trainingStatus && (
                  <div className="training-status">
                    <h3>Training Status</h3>
                    <div className="status-details">
                      <div className="status-item device-info">
                        <span>Device:</span>
                        <span>
                          {trainingStatus.device_info?.name} ({trainingStatus.device_info?.type.toUpperCase()})
                          {trainingStatus.device_info?.memory && 
                            <span className="device-memory"> - {trainingStatus.device_info.memory}</span>
                          }
                        </span>
                      </div>
                      
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
                      {trainingStatus.error && dismissedTrainingError !== trainingStatus.error && (
                        <div className="status-error" role="alert">
                          <span>Error: {trainingStatus.error}</span>
                          <button
                            type="button"
                            className="status-error-dismiss"
                            onClick={() => setDismissedTrainingError(trainingStatus.error)}
                            aria-label="Dismiss training error"
                            title="Dismiss"
                          >
                            ×
                          </button>
                        </div>
                      )}
                    </div>

                    {trainingStatus.history && trainingStatus.history.length > 0 && (
                      <div className="training-graph-container">
                        <h4>Training Progress</h4>
                        <TrainingGraph 
                          history={trainingStatus.history}
                          currentEpoch={trainingStatus.current_epoch}
                          totalEpochs={trainingStatus.total_epochs}
                        />
                      </div>
                    )}
                  </div>
                )}

                {trainingStatus &&
                  !trainingStatus.is_training &&
                  !trainingStatus.error &&
                  (Number(trainingStatus.progress) >= 100 || Boolean(trainingStatus.end_time)) && (
                  <div className="save-model-section">
                    <button 
                      className="save-model-button"
                      onClick={() => {
                        const baseModelName = selectedModel?.split('/').pop() || '';
                        setSaveName(baseModelName ? `${baseModelName}-finetuned` : '');
                        setShowSaveDialog(true);
                      }}
                    >
                      Save Trained Model
                    </button>
                    
                    {showSaveDialog && (
                      <div className="save-dialog">
                        <input
                          type="text"
                          value={saveName}
                          onChange={(e) => setSaveName(e.target.value)}
                          placeholder="Enter name for saved model"
                        />
                        <div className="save-dialog-buttons">
                          <button onClick={handleSaveModel}>Save</button>
                          <button onClick={() => {
                            setShowSaveDialog(false);
                            setSaveName('');
                          }}>Cancel</button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </section>
            </>
          } />
          <Route path="/settings" element={<Settings />} />
        </Routes>

        {showAddModel && (
          <div className="add-model-modal">
            <div className="modal-content">
              <h3>Add New Model</h3>
              <input
                type="text"
                placeholder="Huggingface Model ID (e.g. google/gemma-2b-it)"
                value={newModelId}
                onChange={(e) => {
                  setNewModelId(e.target.value);
                  // Auto-fill display name with cleaned model ID
                  setNewModelName(e.target.value.split('/').pop().replace(/-/g, ' '));
                }}
              />
              <input
                type="text"
                placeholder="Display Name"
                value={newModelName}
                onChange={(e) => setNewModelName(e.target.value)}
              />
              <div className="modal-actions">
                <button onClick={() => setShowAddModel(false)}>Cancel</button>
                <button onClick={handleAddModel}>Add Model</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </Router>
  );
}

export default App; 