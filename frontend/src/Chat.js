import React, { useState, useEffect } from 'react';
import './Chat.css';

function Chat() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [useFinetuned, setUseFinetuned] = useState(false);
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchDownloadedModels();
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedModel || !message.trim()) return;

    setIsLoading(true);
    const newMessage = { role: 'user', content: message };
    setChatHistory(prev => [...prev, newMessage]);
    setMessage('');

    try {
      const response = await fetch('http://localhost:5000/api/models/inference', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          message: message,
          use_finetuned: useFinetuned
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setChatHistory(prev => [...prev, {
          role: 'assistant',
          content: data.response
        }]);
      } else {
        setChatHistory(prev => [...prev, {
          role: 'error',
          content: data.message
        }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setChatHistory(prev => [...prev, {
        role: 'error',
        content: 'Failed to get response'
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
          <select
            value={selectedModel || ''}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            <option value="">Select a model</option>
            {models.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
          <label className="finetuned-toggle">
            <input
              type="checkbox"
              checked={useFinetuned}
              onChange={(e) => setUseFinetuned(e.target.checked)}
            />
            Use fine-tuned version
          </label>
        </div>
      </div>

      <div className="chat-messages">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={!selectedModel || isLoading}
        />
        <button type="submit" disabled={!selectedModel || !message.trim() || isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default Chat; 