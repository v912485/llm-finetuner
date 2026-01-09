import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useApi } from '../hooks/useApi';
import apiConfig from '../config';

const AppContext = createContext();

export const useAppContext = () => useContext(AppContext);

export const AppProvider = ({ children }) => {
  const api = useApi();
  // This state is currently not used in this context but kept for future implementation
  // eslint-disable-next-line no-unused-vars
  const [availableModels, setAvailableModels] = useState([]);
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [savedModels, setSavedModels] = useState([]);
  const [downloadedDatasets, setDownloadedDatasets] = useState([]);
  const [configuredDatasets, setConfiguredDatasets] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  
  // Add flags to prevent duplicate API calls
  const [isFetchingSavedModels, setIsFetchingSavedModels] = useState(false);
  const [isFetchingDownloadedModels, setIsFetchingDownloadedModels] = useState(false);
  const [isInitialDataFetched, setIsInitialDataFetched] = useState(false);

  const fetchDownloadedModels = useCallback(async () => {
    // Prevent duplicate API calls
    if (isFetchingDownloadedModels) return;
    
    try {
      setIsFetchingDownloadedModels(true);
      const data = await api.get(apiConfig.endpoints.models.downloaded);
      if (data.status === 'success') {
        setDownloadedModels(data.downloaded_models || []);
      }
      return data;
    } catch (error) {
      console.error('Error fetching downloaded models:', error);
    } finally {
      setIsFetchingDownloadedModels(false);
    }
  }, [api, isFetchingDownloadedModels]);

  const fetchSavedModels = useCallback(async () => {
    // Prevent duplicate API calls
    if (isFetchingSavedModels) return;
    
    try {
      setIsFetchingSavedModels(true);
      const data = await api.get(apiConfig.endpoints.models.saved);
      if (data.status === 'success') {
        setSavedModels(data.saved_models || []);
      }
      return data;
    } catch (error) {
      console.error('Error fetching saved models:', error);
    } finally {
      setIsFetchingSavedModels(false);
    }
  }, [api, isFetchingSavedModels]);

  const fetchDownloadedDatasets = useCallback(async () => {
    try {
      const data = await api.get(apiConfig.endpoints.datasets.downloaded);
      if (data.status === 'success') {
        setDownloadedDatasets(data.downloaded_datasets || []);
      }
      return data;
    } catch (error) {
      console.error('Error fetching downloaded datasets:', error);
    }
  }, [api]);

  const fetchConfiguredDatasets = useCallback(async () => {
    try {
      const data = await api.get(apiConfig.endpoints.datasets.configured);
      if (data.status === 'success') {
        setConfiguredDatasets(data.configured_datasets || []);
      }
      return data;
    } catch (error) {
      console.error('Error fetching configured datasets:', error);
    }
  }, [api]);

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const data = await api.get(apiConfig.endpoints.training.status);
      if (data.status === 'success') {
        setTrainingStatus(data.training_status);
        setIsTraining(data.training_status?.status === 'training');
      }
      return data;
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  }, [api]);

  const startTraining = useCallback(async (params) => {
    try {
      const data = await api.post(apiConfig.endpoints.training.start, params);
      if (data.status === 'success') {
        setIsTraining(true);
        fetchTrainingStatus();
      }
      return data;
    } catch (error) {
      console.error('Error starting training:', error);
      throw error;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  const cancelTraining = useCallback(async () => {
    try {
      const data = await api.post(apiConfig.endpoints.training.cancel);
      if (data.status === 'success') {
        setIsTraining(false);
        fetchTrainingStatus();
      }
      return data;
    } catch (error) {
      console.error('Error canceling training:', error);
      throw error;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  const fetchInitialData = useCallback(async () => {
    // Prevent duplicate API calls
    if (isInitialDataFetched) return;
    
    try {
      setIsInitialDataFetched(true);
      
      // Execute API calls sequentially to avoid overwhelming the server
      await fetchDownloadedModels();
      await fetchSavedModels();
      await fetchDownloadedDatasets();
      await fetchConfiguredDatasets();
      await fetchTrainingStatus();
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  }, [fetchDownloadedModels, fetchSavedModels, fetchDownloadedDatasets, fetchConfiguredDatasets, fetchTrainingStatus, isInitialDataFetched]);

  useEffect(() => {
    fetchInitialData();
  }, [fetchInitialData]);

  const value = {
    availableModels,
    downloadedModels,
    savedModels,
    downloadedDatasets,
    configuredDatasets,
    trainingStatus,
    isTraining,
    api,
    fetchDownloadedModels,
    fetchSavedModels,
    fetchDownloadedDatasets,
    fetchConfiguredDatasets,
    fetchTrainingStatus,
    startTraining,
    cancelTraining
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}; 