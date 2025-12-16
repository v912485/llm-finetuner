import React, { useState, useEffect } from 'react';
import './Settings.css';
import apiConfig from './config';

function Settings() {
  const [adminToken, setAdminToken] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [saveStatus, setSaveStatus] = useState('');
  const [hfConfigured, setHfConfigured] = useState(false);

  useEffect(() => {
    const storedAdminToken = localStorage.getItem('adminToken') || '';
    setAdminToken(storedAdminToken);

    // Fetch whether a Hugging Face token is configured on the server
    fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.huggingfaceToken}`, {
      headers: storedAdminToken ? { Authorization: `Bearer ${storedAdminToken}` } : {}
    })
      .then(res => res.json())
      .then(data => {
        setHfConfigured(Boolean(data.configured));
      })
      .catch(error => console.error('Error fetching API key:', error));
  }, []);

  const handleSaveAdminToken = (e) => {
    e.preventDefault();
    const trimmed = adminToken.trim();
    if (!trimmed) {
      localStorage.removeItem('adminToken');
      setSaveStatus('Cleared admin token');
    } else {
      localStorage.setItem('adminToken', trimmed);
      setSaveStatus('Saved admin token');
    }
    setTimeout(() => setSaveStatus(''), 3000);
    setTimeout(() => window.location.reload(), 250);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('adminToken');
      const headers = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      };

      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.huggingfaceToken}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ token: apiKey }),
      });
      
      const data = await response.json();
      setSaveStatus(data.status === 'success' ? 'Saved successfully!' : 'Error saving token');
      if (data.status === 'success') {
        setHfConfigured(true);
        setApiKey('');
      }
      
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
        <h3>Admin Token</h3>
        <p className="settings-description">
          Required to use this app when the backend is exposed on a LAN. This is stored in your browser only.
        </p>

        <form onSubmit={handleSaveAdminToken} className="settings-form">
          <div className="form-group">
            <input
              type="password"
              value={adminToken}
              onChange={(e) => setAdminToken(e.target.value)}
              placeholder="Enter admin token"
              className="api-key-input"
            />
          </div>
          <button type="submit" className="save-button">
            Save Admin Token
          </button>
        </form>
      </div>
      
      <div className="settings-section">
        <h3>Hugging Face API Token</h3>
        <p className="settings-description">
          Required for accessing gated models like Gemma. Get your token from{' '}
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">
            Hugging Face Settings
          </a>
        </p>

        <p className="settings-description">
          Current server status: {hfConfigured ? 'Configured' : 'Not configured'}
        </p>
        
        <form onSubmit={handleSubmit} className="settings-form">
          <div className="form-group">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter a Hugging Face token to set/replace"
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