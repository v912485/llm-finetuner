import { useState } from 'react';
import apiConfig from '../config';

export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('adminToken');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const callApi = async (endpoint, method = 'GET', body = null, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const url = `${apiConfig.apiBaseUrl}${endpoint}`;
      const fetchOptions = {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
          ...(options.headers || {})
        },
        ...options
      };
      
      if (body && method !== 'GET') {
        fetchOptions.body = JSON.stringify(body);
      }
      
      const response = await fetch(url, fetchOptions);
      const contentType = response.headers.get('content-type') || '';
      const responseText = await response.text();
      let data = null;
      if (contentType.includes('application/json')) {
        try {
          data = responseText ? JSON.parse(responseText) : null;
        } catch (e) {
          data = null;
        }
      }
      
      if (!response.ok) {
        const message =
          (data && data.message) ||
          (data && data.error && data.error.message) ||
          (data && data.error && data.error.error && data.error.error.message) ||
          (responseText && responseText.trim()) ||
          `Request failed (${response.status})`;
        throw new Error(message);
      }
      
      setLoading(false);
      return data;
    } catch (err) {
      setError(err.message || 'Something went wrong');
      setLoading(false);
      throw err;
    }
  };

  return {
    loading,
    error,
    callApi,
    get: (endpoint, options) => callApi(endpoint, 'GET', null, options),
    post: (endpoint, body, options) => callApi(endpoint, 'POST', body, options),
    put: (endpoint, body, options) => callApi(endpoint, 'PUT', body, options),
    delete: (endpoint, options) => callApi(endpoint, 'DELETE', null, options)
  };
}; 