import React, { useState, useEffect } from 'react';
import './Settings.css';
import apiConfig from './config';

function Settings() {
  const [apiKey, setApiKey] = useState('');
  const [saveStatus, setSaveStatus] = useState('');
  const [isConfigured, setIsConfigured] = useState(false);
  const [adminToken, setAdminToken] = useState('');
  const [adminConfigured, setAdminConfigured] = useState(false);
  const [adminSaveStatus, setAdminSaveStatus] = useState('');

  const getAuthHeaders = () => {
    const token = localStorage.getItem('adminToken');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  useEffect(() => {
    // Fetch current API key if it exists
    fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.huggingfaceToken}`, {
      headers: getAuthHeaders()
    })
      .then(res => res.json())
      .then(data => {
        setIsConfigured(Boolean(data.configured));
      })
      .catch(error => console.error('Error fetching API key:', error));
  }, []);

  useEffect(() => {
    fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.adminToken}`)
      .then(res => res.json())
      .then(data => {
        setAdminConfigured(Boolean(data.configured));
      })
      .catch(error => console.error('Error fetching admin token status:', error));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.huggingfaceToken}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders()
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

  const handleAdminSubmit = async (e) => {
    e.preventDefault();
    if (!adminToken.trim()) {
      setAdminSaveStatus('Admin token is required');
      return;
    }
    try {
      const authHeader = adminToken ? `Bearer ${adminToken}` : '';
      const response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.settings.adminToken}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
          ...(authHeader ? { Authorization: authHeader } : {})
        },
        body: JSON.stringify({ token: adminToken })
      });

      const data = await response.json();
      if (data.status === 'success') {
        localStorage.setItem('adminToken', adminToken);
        setAdminConfigured(true);
        setAdminSaveStatus('Admin token saved');
        window.dispatchEvent(new Event('admin-token-updated'));
      } else {
        setAdminSaveStatus(data.message || 'Error saving admin token');
      }

      setTimeout(() => setAdminSaveStatus(''), 3000);
    } catch (error) {
      setAdminSaveStatus('Error saving admin token');
      console.error('Error:', error);
    }
  };

  return (
    <div className="settings-container">
      <h2>Settings</h2>
      
      <div className="settings-section">
        <h3>Admin Token</h3>
        <p className="settings-description">
          Required for accessing protected API routes. Set this once on first use.
        </p>

        <form onSubmit={handleAdminSubmit} className="settings-form">
          <div className="form-group">
            <input
              type="password"
              value={adminToken}
              onChange={(e) => setAdminToken(e.target.value)}
              placeholder="Enter admin token"
              className="api-key-input"
            />
            <div className={`save-status ${adminConfigured ? 'success' : 'error'}`}>
              {adminConfigured ? 'Admin token configured' : 'Admin token not configured'}
            </div>
          </div>

          <button type="submit" className="save-button">
            Save Admin Token
          </button>

          {adminSaveStatus && (
            <div className={`save-status ${adminSaveStatus.includes('Error') ? 'error' : 'success'}`}>
              {adminSaveStatus}
            </div>
          )}
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
        
        <form onSubmit={handleSubmit} className="settings-form">
          <div className="form-group">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your Hugging Face API token"
              className="api-key-input"
            />
            <div className={`save-status ${isConfigured ? 'success' : 'error'}`}>
              {isConfigured ? 'Token configured on server' : 'Token not configured'}
            </div>
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