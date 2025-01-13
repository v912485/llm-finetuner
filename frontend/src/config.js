const apiConfig = {
  apiBaseUrl: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  endpoints: {
    models: {
      available: '/models/available',
      downloaded: '/models/downloaded',
      download: '/models/download',
      inference: '/models/inference',
      saved: '/models/saved',
      cancelTraining: '/models/cancel-training'
    },
    datasets: {
      prepare: '/datasets/prepare',
      configured: '/datasets/configured',
      downloaded: '/datasets/downloaded',
      structure: '/datasets/structure',
      config: '/datasets/config'
    },
    training: {
      start: '/training/start',
      status: '/training/status',
      save: '/training/save'
    },
    settings: {
      huggingfaceToken: '/settings/huggingface_token',
      config: '/settings/config'
    }
  }
};

export default apiConfig;