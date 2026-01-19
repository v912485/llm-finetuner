import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Chat from './Chat';
import Settings from './Settings';
import TrainingGraph from './components/TrainingGraph';
import ConfigForm from './components/ConfigForm';
import apiConfig from './config';

function App() {
  const [adminToken, setAdminToken] = useState(() => localStorage.getItem('adminToken') || '');
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [finetuningParams, setFinetuningParams] = useState({
    learningRate: 0.0001,
    batchSize: 8,
    epochs: 3,
    validationSplit: 0.2,
    checkpointEnabled: false,
    checkpointInterval: 1
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
  const [openDatasetMenuId, setOpenDatasetMenuId] = useState(null);

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
        setTrainingStatus(prevStatus => {
          if (!prevStatus || !Array.isArray(prevStatus.history) || !Array.isArray(data.history)) {
            return data;
          }
          const prevHistory = prevStatus.history;
          const nextHistory = data.history;
          if (prevHistory.length === nextHistory.length) {
            if (prevHistory.length === 0) {
              return { ...data, history: prevHistory };
            }
            const prevLast = prevHistory[prevHistory.length - 1];
            const nextLast = nextHistory[nextHistory.length - 1];
            if (prevLast?.epoch === nextLast?.epoch &&
                prevLast?.train_loss === nextLast?.train_loss &&
                prevLast?.val_loss === nextLast?.val_loss) {
              return { ...data, history: prevHistory };
            }
          }
          return data;
        });
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
    const updateAdminToken = () => {
      const nextToken = localStorage.getItem('adminToken') || '';
      setAdminToken(nextToken);
    };
    updateAdminToken();
    window.addEventListener('storage', updateAdminToken);
    window.addEventListener('admin-token-updated', updateAdminToken);
    return () => {
      window.removeEventListener('storage', updateAdminToken);
      window.removeEventListener('admin-token-updated', updateAdminToken);
    };
  }, []);

  useEffect(() => {
    const fetchInitialData = async () => {
      const hasToken = Boolean(adminToken);
      if (!hasToken) {
        return;
      }
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
  }, [adminToken, getAuthHeaders]);

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
            checkpoint_enabled: finetuningParams.checkpointEnabled,
            checkpoint_interval_epochs: finetuningParams.checkpointInterval,
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
  const getSelectedModelConfig = useCallback(() => {
    return availableModels.find(model => model.id === selectedModel);
  }, [availableModels, selectedModel]);

  useEffect(() => {
    console.log('Selected model changed:', selectedModel);
    console.log('Available models:', availableModels);
    console.log('Selected model config:', getSelectedModelConfig());
  }, [selectedModel, availableModels, getSelectedModelConfig]);

  useEffect(() => {
    const selectedModelConfig = getSelectedModelConfig();
    if (!selectedModelConfig?.supports_lora && trainingMethod !== 'standard') {
      setTrainingMethod('standard');
    }
  }, [selectedModel, trainingMethod, availableModels, getSelectedModelConfig]);

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

  const handleDeleteDownloadedModel = async (modelId) => {
    if (!window.confirm('Are you sure you want to delete the downloaded files for this model?')) return;

    try {
      const encodedModelId = encodeURIComponent(modelId);
      const response = await fetch(
        `${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.deleteDownloaded}/${encodedModelId}`,
        { method: 'DELETE', headers: getAuthHeaders() }
      );
      const data = await response.json();
      if (data.status === 'success') {
        setDownloadedModels(prev => prev.filter(id => id !== modelId));
        setSelectedModel(prev => (prev === modelId ? null : prev));
      } else {
        alert(data.message || 'Failed to delete downloaded model files');
      }
    } catch (error) {
      console.error('Error deleting downloaded model files:', error);
      alert('Error deleting downloaded model files');
    }
  };

  const handleDatasetDelete = async (datasetId) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;

    try {
      const encodedId = encodeURIComponent(datasetId);
      const response = await fetch(
        `${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.delete}/${encodedId}`,
        { method: 'DELETE', headers: getAuthHeaders() }
      );
      const data = await response.json();
      if (data.status === 'success') {
        setDownloadedDatasets(prev => prev.filter(dataset => dataset.dataset_id !== datasetId));
        setSelectedFiles(prev => prev.filter(id => id !== datasetId));
        setConfiguredFiles(prev => prev.filter(id => id !== datasetId));
        setDatasetConfigs(prev => {
          const next = { ...prev };
          delete next[datasetId];
          return next;
        });
        setFileConfig(prev => {
          const next = { ...prev };
          delete next[datasetId];
          return next;
        });
        setFileStructures(prev => {
          const next = { ...prev };
          delete next[datasetId];
          return next;
        });
      } else {
        alert(data.message || 'Failed to delete dataset');
      }
    } catch (error) {
      console.error('Error deleting dataset:', error);
      alert('Error deleting dataset');
    }
  };

  const handleDatasetRename = async (dataset) => {
    const nextName = window.prompt('Enter a new dataset name', dataset.name);
    if (!nextName || nextName.trim() === dataset.name) return;

    try {
      const encodedId = encodeURIComponent(dataset.dataset_id);
      const response = await fetch(
        `${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.rename}/${encodedId}/rename`,
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders()
          },
          body: JSON.stringify({ name: nextName.trim() })
        }
      );
      const data = await response.json();
      if (data.status === 'success') {
        setDownloadedDatasets(prev => prev.map(item => (
          item.dataset_id === dataset.dataset_id ? { ...item, ...data.dataset } : item
        )));
      } else {
        alert(data.message || 'Failed to rename dataset');
      }
    } catch (error) {
      console.error('Error renaming dataset:', error);
      alert('Error renaming dataset');
    }
  };

  const handleDatasetVersion = async (datasetId) => {
    try {
      const encodedId = encodeURIComponent(datasetId);
      const response = await fetch(
        `${apiConfig.apiBaseUrl}${apiConfig.endpoints.datasets.version}/${encodedId}/version`,
        { method: 'POST', headers: getAuthHeaders() }
      );
      const data = await response.json();
      if (data.status === 'success') {
        setDownloadedDatasets(prev => [data.dataset, ...prev]);
      } else {
        alert(data.message || 'Failed to create dataset version');
      }
    } catch (error) {
      console.error('Error creating dataset version:', error);
      alert('Error creating dataset version');
    }
  };

  const toggleDatasetMenu = (datasetId) => {
    setOpenDatasetMenuId(prev => (prev === datasetId ? null : datasetId));
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
                              <button
                                className="delete-model-button"
                                onClick={() => handleDeleteDownloadedModel(model.id)}
                              >
                                Remove Files
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
                              {Number.isFinite(dataset.version) && (
                                <span className="dataset-version">v{dataset.version}</span>
                              )}
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
                              <div className="dataset-menu">
                                <button
                                  type="button"
                                  className="dataset-menu-button"
                                  onClick={() => toggleDatasetMenu(dataset.dataset_id)}
                                  aria-haspopup="menu"
                                  aria-expanded={openDatasetMenuId === dataset.dataset_id}
                                >
                                  ...
                                </button>
                                {openDatasetMenuId === dataset.dataset_id && (
                                  <div className="dataset-menu-dropdown" role="menu">
                                    <button
                                      type="button"
                                      className="dataset-version-btn"
                                      role="menuitem"
                                      onClick={() => {
                                        toggleDatasetMenu(dataset.dataset_id);
                                        handleDatasetVersion(dataset.dataset_id);
                                      }}
                                    >
                                      New Version
                                    </button>
                                    <button
                                      type="button"
                                      className="dataset-rename-btn"
                                      role="menuitem"
                                      onClick={() => {
                                        toggleDatasetMenu(dataset.dataset_id);
                                        handleDatasetRename(dataset);
                                      }}
                                    >
                                      Rename
                                    </button>
                                    <button
                                      type="button"
                                      className="dataset-delete-btn"
                                      role="menuitem"
                                      onClick={() => {
                                        toggleDatasetMenu(dataset.dataset_id);
                                        handleDatasetDelete(dataset.dataset_id);
                                      }}
                                    >
                                      Delete
                                    </button>
                                  </div>
                                )}
                              </div>
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
                <div className="params-form-container">
                  <div className="params-column">
                    <div className="params-group-card">
                      <h3>Core Training</h3>
                      <div className="param-item">
                        <label>Training Method</label>
                        <select
                          value={trainingMethod}
                          onChange={(e) => setTrainingMethod(e.target.value)}
                        >
                          <option value="standard">Full Fine-tuning</option>
                          {(() => {
                            const modelConfig = getSelectedModelConfig();
                            return selectedModel && modelConfig?.supports_lora && (
                              <>
                                <option value="lora">LoRA (Memory Efficient)</option>
                                <option value="qlora">QLoRA (Very Memory Efficient)</option>
                              </>
                            );
                          })()}
                        </select>
                        <p className="param-description">
                          {config?.training_methods?.[trainingMethod]?.description || 'Select how the model should be trained.'}
                        </p>
                      </div>

                      <div className="params-row">
                        <div className="param-item">
                          <label>Learning Rate</label>
                          <div className="input-with-hint">
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
                        </div>
                        <div className="param-item">
                          <label>Batch Size</label>
                          <input
                            type="number"
                            value={finetuningParams.batchSize}
                            onChange={(e) =>
                              setFinetuningParams({
                                ...finetuningParams,
                                batchSize: parseInt(e.target.value, 10),
                              })
                            }
                            min="1"
                          />
                        </div>
                        <div className="param-item">
                          <label>Epochs</label>
                          <input
                            type="number"
                            value={finetuningParams.epochs}
                            onChange={(e) =>
                              setFinetuningParams({
                                ...finetuningParams,
                                epochs: parseInt(e.target.value, 10),
                              })
                            }
                            min="1"
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="params-column">
                    <div className="params-group-card">
                      <h3>Validation & Checkpointing</h3>
                      <div className="param-item">
                        <label>Validation Split</label>
                        <div className="range-input-container">
                          <input
                            type="range"
                            min="0.1"
                            max="0.5"
                            step="0.05"
                            value={finetuningParams.validationSplit}
                            onChange={(e) =>
                              setFinetuningParams({
                                ...finetuningParams,
                                validationSplit: parseFloat(e.target.value),
                              })
                            }
                          />
                          <span className="range-value">{(finetuningParams.validationSplit * 100).toFixed(0)}%</span>
                        </div>
                        <p className="param-description">
                          Portion of data held back for evaluation during training.
                        </p>
                      </div>

                      <div className="param-item checkbox-item">
                        <label className="checkbox-label">
                          <input
                            type="checkbox"
                            checked={finetuningParams.checkpointEnabled}
                            onChange={(e) =>
                              setFinetuningParams({
                                ...finetuningParams,
                                checkpointEnabled: e.target.checked
                              })
                            }
                          />
                          <span>Enable Checkpointing</span>
                        </label>
                      </div>

                      {finetuningParams.checkpointEnabled && (
                        <div className="param-item animate-fade-in">
                          <label>Checkpoint Interval (epochs)</label>
                          <input
                            type="number"
                            value={finetuningParams.checkpointInterval}
                            onChange={(e) =>
                              setFinetuningParams({
                                ...finetuningParams,
                                checkpointInterval: Math.max(1, parseInt(e.target.value, 10) || 1)
                              })
                            }
                            min="1"
                          />
                          <p className="param-description">Save model progress every N epochs.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </section>

              <section className="training-control">
                <div className="section-header">
                  <h2>4. Training Control</h2>
                  {trainingStatus?.is_training && (
                    <div className="live-indicator">
                      <span className="live-dot"></span>
                      LIVE
                    </div>
                  )}
                </div>

                <div className="control-actions">
                  {!isTraining ? (
                    <button
                      className="start-training-button"
                      onClick={startTraining}
                      disabled={!selectedModel || selectedFiles.length === 0 || !areSelectedFilesConfigured()}
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="button-icon">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                      </svg>
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
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="button-icon">
                        <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
                      </svg>
                      Cancel Training
                    </button>
                  )}
                </div>

                {trainingStatus?.device_info && (
                  <div className={`training-status-card ${trainingStatus.is_training ? 'is-active' : ''}`}>
                    <div className="status-card-header">
                      <h3>{trainingStatus.is_training ? 'Training Status' : 'System Status'}</h3>
                      <div className="status-header-badges">
                        <div className="device-badge">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
                            <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
                            <line x1="6" y1="6" x2="6.01" y2="6"></line>
                            <line x1="6" y1="18" x2="6.01" y2="18"></line>
                          </svg>
                          <span className="device-name">{trainingStatus.device_info.name}</span>
                          <span className="backend-tag">{trainingStatus.device_info.backend}</span>
                          {trainingStatus.device_info.memory_free !== undefined && trainingStatus.device_info.memory_total !== undefined && (
                            <span className="vram-mini-usage">
                              {((trainingStatus.device_info.memory_total - trainingStatus.device_info.memory_free) / (1024**3)).toFixed(1)} / {trainingStatus.device_info.memory}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="status-grid">
                      {(trainingStatus.is_training || (trainingStatus.progress > 0 && trainingStatus.progress < 100)) ? (
                        <div className="status-metric-item">
                          <span className="metric-label">Progress</span>
                          <div className="progress-display">
                            <div className="progress-bar-container">
                              <div 
                                className="progress-bar" 
                                style={{ width: `${trainingStatus.progress}%` }}
                              >
                                <div className="progress-glow"></div>
                              </div>
                            </div>
                            <span className="progress-percentage">{trainingStatus.progress}%</span>
                          </div>
                        </div>
                      ) : !trainingStatus.is_training && trainingStatus.progress === 0 ? (
                        <div className="status-ready-indicator">
                          <div className="ready-dot"></div>
                          <span>System Ready - Waiting for training to start</span>
                        </div>
                      ) : null}

                      <div className="metrics-row">
                        <div className="status-metric-mini">
                          <span className="metric-label">Epoch</span>
                          <span className="metric-value">{trainingStatus.current_epoch} / {trainingStatus.total_epochs}</span>
                        </div>
                        {trainingStatus.loss && (
                          <div className="status-metric-mini">
                            <span className="metric-label">Current Loss</span>
                            <span className="metric-value">{trainingStatus.loss.toFixed(4)}</span>
                          </div>
                        )}
                      </div>

                      {trainingStatus.error && dismissedTrainingError !== trainingStatus.error && (
                        <div className="status-error-v2" role="alert">
                          <div className="error-content">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <circle cx="12" cy="12" r="10"></circle>
                              <line x1="12" y1="8" x2="12" y2="12"></line>
                              <line x1="12" y1="16" x2="12.01" y2="16"></line>
                            </svg>
                            <span>{trainingStatus.error}</span>
                          </div>
                          <button
                            type="button"
                            className="status-error-dismiss"
                            onClick={() => setDismissedTrainingError(trainingStatus.error)}
                            aria-label="Dismiss training error"
                          >
                            ×
                          </button>
                        </div>
                      )}
                    </div>

                    {trainingStatus.history && trainingStatus.history.length > 0 && (
                      <div className="training-graph-section">
                        <div className="graph-header">
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                          </svg>
                          <h4>Learning Curves</h4>
                        </div>
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
                  <div className="save-model-container">
                    <button 
                      className="save-model-button"
                      onClick={() => {
                        const baseModelName = selectedModel?.split('/').pop() || '';
                        setSaveName(baseModelName ? `${baseModelName}-finetuned` : '');
                        setShowSaveDialog(true);
                      }}
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="button-icon">
                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
                        <polyline points="17 21 17 13 7 13 7 21"></polyline>
                        <polyline points="7 3 7 8 15 8"></polyline>
                      </svg>
                      Save Trained Model
                    </button>
                    
                    {showSaveDialog && (
                      <div className="save-dialog-overlay">
                        <div className="save-dialog-box">
                          <h4>Save Finetuned Model</h4>
                          <p>Enter a name for your saved model weights.</p>
                          <input
                            type="text"
                            value={saveName}
                            onChange={(e) => setSaveName(e.target.value)}
                            placeholder="e.g. gemma-7b-custom-v1"
                            autoFocus
                          />
                          <div className="save-dialog-actions">
                            <button className="cancel-btn" onClick={() => {
                              setShowSaveDialog(false);
                              setSaveName('');
                            }}>Cancel</button>
                            <button className="confirm-btn" onClick={handleSaveModel} disabled={!saveName.trim()}>
                              Save Weights
                            </button>
                          </div>
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