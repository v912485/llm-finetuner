import { useState } from 'react';
import apiConfig from '../config';

export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const callApi = async (endpoint, method = 'GET', body = null, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const url = `${apiConfig.apiBaseUrl}${endpoint}`;
      const fetchOptions = {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      };
      
      if (body && method !== 'GET') {
        fetchOptions.body = JSON.stringify(body);
      }
      
      const response = await fetch(url, fetchOptions);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || 'Something went wrong');
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