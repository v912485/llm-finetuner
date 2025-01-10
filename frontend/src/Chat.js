import React, { useState, useEffect } from 'react';
import './Chat.css';

function Chat() {
  const [models, setModels] = useState([]);
  const [savedModels, setSavedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedSavedModel, setSelectedSavedModel] = useState(null);
  const [useFinetuned, setUseFinetuned] = useState(false);
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchDownloadedModels();
    fetchSavedModels();
  }, []);

  const fetchDownloadedModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/downloaded');
      const data = await response.json();
      if (data.status === 'success') {
        setModels(data.downloaded_models);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchSavedModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/saved');
      const data = await response.json();
      if (data.status === 'success') {
        setSavedModels(data.saved_models);
      }
    } catch (error) {
      console.error('Error fetching saved models:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if ((!selectedModel && !selectedSavedModel) || !message.trim()) return;

    setIsLoading(true);
    const newMessage = { text: message, sender: 'user' };
    setChatHistory(prev => [...prev, newMessage]);
    setMessage('');

    try {
      const response = await fetch('http://localhost:5000/api/models/inference', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedSavedModel?.original_model || selectedModel,
          input: message,
          use_finetuned: useFinetuned || !!selectedSavedModel,
          saved_model_name: selectedSavedModel?.name
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setChatHistory(prev => [...prev, {
          text: data.response,
          sender: 'assistant'
        }]);
      } else {
        console.error('Inference error:', data.message);
        setChatHistory(prev => [...prev, {
          text: `Error: ${data.message}`,
          sender: 'error'
        }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setChatHistory(prev => [...prev, {
        text: `Error: ${error.message}`,
        sender: 'error'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Chat with AI Model</h2>
        <div className="chat-model-selection">
          <div className="model-select-group">
            <label>Base Model:</label>
            <select
              value={selectedModel || ''}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setSelectedSavedModel(null);
              }}
              disabled={!!selectedSavedModel}
            >
              <option value="">Select a model</option>
              {models.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
            {selectedModel && (
              <label className="finetuned-toggle">
                <input
                  type="checkbox"
                  checked={useFinetuned}
                  onChange={(e) => setUseFinetuned(e.target.checked)}
                />
                Use fine-tuned version
              </label>
            )}
          </div>
          
          <div className="model-select-group">
            <label>Saved Models:</label>
            <select
              value={selectedSavedModel?.name || ''}
              onChange={(e) => {
                const saved = savedModels.find(m => m.name === e.target.value);
                setSelectedSavedModel(saved || null);
                setSelectedModel(null);
                setUseFinetuned(false);
              }}
              disabled={!!selectedModel}
            >
              <option value="">Select a saved model</option>
              {savedModels.map(model => (
                <option key={model.name} value={model.name}>
                  {model.name} ({model.original_model})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="chat-messages">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <div className="message-content">{msg.text}</div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={(!selectedModel && !selectedSavedModel) || isLoading}
        />
        <button 
          type="submit" 
          disabled={(!selectedModel && !selectedSavedModel) || !message.trim() || isLoading}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default Chat; 