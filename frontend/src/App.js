import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [finetuningParams, setFinetuningParams] = useState({
    learningRate: 0.0001,
    batchSize: 8,
    epochs: 3,
  });
  const [loading, setLoading] = useState({});
  const [error, setError] = useState({});

  useEffect(() => {
    fetchAvailableModels();
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
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/dataset/prepare', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.status === 'success') {
        setDataset(data.dataset_path);
      }
    } catch (error) {
      console.error('Error uploading dataset:', error);
    }
  };

  return (
    <div className="App">
      <h1>LLM Fine-tuning Interface</h1>
      
      <section className="model-selection">
        <h2>1. Select Model</h2>
        <div className="models-grid">
          {availableModels.map((model) => (
            <div key={model.id} className="model-card">
              <h3>{model.name}</h3>
              <p>Size: {model.size}</p>
              <button
                onClick={() => handleModelDownload(model.id)}
                disabled={loading[model.id] || selectedModel === model.id}
                className={loading[model.id] ? 'loading' : ''}
              >
                {loading[model.id] ? 'Downloading...' : 
                 selectedModel === model.id ? 'Downloaded' : 'Download'}
              </button>
              {error[model.id] && (
                <p className="error-message">{error[model.id]}</p>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="dataset-preparation">
        <h2>2. Prepare Dataset</h2>
        <input
          type="file"
          accept=".json,.jsonl,.csv,.txt"
          onChange={handleDatasetUpload}
        />
        {dataset && <p>Dataset prepared: {dataset}</p>}
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
        </div>
      </section>
    </div>
  );
}

export default App; 