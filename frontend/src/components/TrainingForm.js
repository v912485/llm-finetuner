import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import ModelSelector from './ModelSelector';
import DatasetSelector from './DatasetSelector';
import './TrainingForm.css';

const TrainingForm = ({ onTrainingStart }) => {
  const { startTraining } = useAppContext();
  
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [trainingParams, setTrainingParams] = useState({
    learningRate: 0.0001,
    batchSize: 8,
    epochs: 3,
    validationSplit: 0.2,
  });
  const [trainingMethod, setTrainingMethod] = useState('standard');
  const [saveName, setSaveName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setTrainingParams(prev => ({
      ...prev,
      [name]: name === 'learningRate' || name === 'validationSplit' 
        ? parseFloat(value) 
        : parseInt(value, 10)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }
    
    if (selectedDatasets.length === 0) {
      setError('Please select at least one dataset');
      return;
    }
    
    if (!saveName.trim()) {
      setError('Please enter a name for the fine-tuned model');
      return;
    }
    
    setIsSubmitting(true);
    setError(null);
    
    try {
      const params = {
        model_id: selectedModel,
        datasets: selectedDatasets,
        training_params: trainingParams,
        method: trainingMethod,
        save_name: saveName
      };
      
      await startTraining(params);
      
      if (onTrainingStart) {
        onTrainingStart();
      }
    } catch (err) {
      setError(err.message || 'Failed to start training');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form className="training-form" onSubmit={handleSubmit}>
      <h2>Start Fine-tuning</h2>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="form-group">
        <ModelSelector 
          selectedModel={selectedModel}
          onModelSelect={setSelectedModel}
          label="Base Model"
          includeSavedModels={false}
        />
      </div>
      
      <div className="form-group">
        <DatasetSelector 
          selectedDatasets={selectedDatasets}
          onDatasetSelect={setSelectedDatasets}
          label="Training Datasets"
          onlyConfigured={true}
        />
      </div>
      
      <div className="form-group">
        <label>Training Method</label>
        <select
          value={trainingMethod}
          onChange={(e) => setTrainingMethod(e.target.value)}
        >
          <option value="standard">Standard Fine-tuning</option>
          <option value="lora">LoRA Fine-tuning</option>
          <option value="qlora">QLoRA Fine-tuning</option>
        </select>
      </div>
      
      <div className="form-group">
        <label>Learning Rate</label>
        <input
          type="number"
          name="learningRate"
          value={trainingParams.learningRate}
          onChange={handleParamChange}
          step="0.00001"
          min="0.00001"
          max="0.01"
        />
      </div>
      
      <div className="form-group">
        <label>Batch Size</label>
        <input
          type="number"
          name="batchSize"
          value={trainingParams.batchSize}
          onChange={handleParamChange}
          min="1"
          max="64"
        />
      </div>
      
      <div className="form-group">
        <label>Epochs</label>
        <input
          type="number"
          name="epochs"
          value={trainingParams.epochs}
          onChange={handleParamChange}
          min="1"
          max="100"
        />
      </div>
      
      <div className="form-group">
        <label>Validation Split</label>
        <input
          type="number"
          name="validationSplit"
          value={trainingParams.validationSplit}
          onChange={handleParamChange}
          step="0.01"
          min="0.1"
          max="0.5"
        />
      </div>
      
      <div className="form-group">
        <label>Save Model As</label>
        <input
          type="text"
          value={saveName}
          onChange={(e) => setSaveName(e.target.value)}
          placeholder="Enter a name for the fine-tuned model"
        />
      </div>
      
      <button 
        type="submit" 
        className="start-training-btn"
        disabled={isSubmitting}
      >
        {isSubmitting ? 'Starting...' : 'Start Training'}
      </button>
    </form>
  );
};

export default TrainingForm; 