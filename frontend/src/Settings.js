import React, { useState, useEffect } from 'react';
import './Settings.css';
import apiConfig from './config';

function Settings() {
  const [apiKey, setApiKey] = useState('');
  const [saveStatus, setSaveStatus] = useState('');

  useEffect(() => {
    // Fetch current API key if it exists
    fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.huggingfaceToken}`)
      .then(res => res.json())
      .then(data => {
        if (data.token) {
          setApiKey(data.token);
        }
      })
      .catch(error => console.error('Error fetching API key:', error));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.huggingfaceToken}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token: apiKey }),
      });
      
      const data = await response.json();
      setSaveStatus(data.status === 'success' ? 'Saved successfully!' : 'Error saving token');
      
      setTimeout(() => setSaveStatus(''), 3000);
    } catch (error) {
      setSaveStatus('Error saving token');
      console.error('Error:', error);
    }
  };

  return (
    <div className="settings-container">
      <h2>Settings</h2>
      
      <div className="settings-section">
        <h3>Hugging Face API Token</h3>
        <p className="settings-description">
          Required for accessing gated models like Gemma. Get your token from{' '}
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">
            Hugging Face Settings
          </a>
        </p>
        
        <form onSubmit={handleSubmit} className="settings-form">
          <div className="form-group">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your Hugging Face API token"
              className="api-key-input"
            />
          </div>
          
          <button type="submit" className="save-button">
            Save Token
          </button>
          
          {saveStatus && (
            <div className={`save-status ${saveStatus.includes('Error') ? 'error' : 'success'}`}>
              {saveStatus}
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

export default Settings; 