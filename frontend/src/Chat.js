import React, { useState, useEffect, useRef } from 'react';
import './Chat.css';
import apiConfig from './config';

function Chat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [selectedBaseModel, setSelectedBaseModel] = useState('');
    const [selectedFineTunedModel, setSelectedFineTunedModel] = useState('');
    const [downloadedModels, setDownloadedModels] = useState([]);
    const [savedModels, setSavedModels] = useState([]);
    const [useOpenAIFormat, setUseOpenAIFormat] = useState(true);
    const [temperature, setTemperature] = useState(0.7);
    const [maxTokens, setMaxTokens] = useState(512);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);

    // Get the currently active model
    const activeModel = selectedFineTunedModel || selectedBaseModel;

    useEffect(() => {
        fetchAvailableModels();
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const fetchAvailableModels = async () => {
        try {
            const [downloadedResponse, savedResponse] = await Promise.all([
                fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.downloaded}`),
                fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.saved}`)
            ]);
            
            const downloadedData = await downloadedResponse.json();
            const savedData = await savedResponse.json();
            
            console.log('Downloaded models response:', downloadedData);
            console.log('Saved models response:', savedData);
            
            if (downloadedData.status === 'success') {
                setDownloadedModels(downloadedData.downloaded_models || []);
                if (downloadedData.downloaded_models?.length > 0) {
                    setSelectedBaseModel(downloadedData.downloaded_models[0]);
                }
            }
            
            if (savedData.status === 'success') {
                const savedModelsList = savedData.saved_models || [];
                console.log('Saved models list:', savedModelsList);
                setSavedModels(savedModelsList);
            }
        } catch (error) {
            console.error('Error fetching models:', error);
            setError('Failed to load available models');
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || !activeModel) return;

        const newMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, newMessage]);
        setInput('');
        setIsLoading(true);
        setError(null);

        try {
            let response;
            // Check if we're using a saved model
            const savedModel = savedModels.find(model => model.path === activeModel);
            console.log('Active model:', activeModel);
            console.log('Saved models:', savedModels);
            console.log('Found saved model:', savedModel);

            if (useOpenAIFormat) {
                const requestBody = {
                    model: savedModel ? savedModel.name : activeModel,
                    messages: [...messages, newMessage],
                    temperature: temperature,
                    max_tokens: maxTokens,
                };

                console.log('OpenAI format request body:', requestBody);

                response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.chatCompletions}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody),
                });
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error.message);
                }
                setMessages(prev => [...prev, data.choices[0].message]);
            } else {
                const requestBody = savedModel 
                    ? {
                        saved_model_name: savedModel.name,
                        input: input,
                        temperature: temperature,
                        max_length: maxTokens,
                    }
                    : {
                        model_id: activeModel,
                        input: input,
                        temperature: temperature,
                        max_length: maxTokens,
                    };
                
                console.log('Standard format request body:', requestBody);
                
                response = await fetch(`${apiConfig.apiBaseUrl}${apiConfig.endpoints.models.inference}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody),
                });
                const data = await response.json();
                if (data.status === 'error') {
                    throw new Error(data.message);
                }
                setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
            }
        } catch (error) {
            console.error('Error:', error);
            setError(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chat-container">
            <div className="chat-settings">
                <div className="model-selections">
                    <div className="model-selection">
                        <label>Base Model:</label>
                        <select 
                            value={selectedBaseModel} 
                            onChange={(e) => {
                                setSelectedBaseModel(e.target.value);
                                setSelectedFineTunedModel(''); // Clear fine-tuned selection when base model is selected
                            }}
                        >
                            <option value="">Select a model</option>
                            {downloadedModels.map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>
                    {savedModels.length > 0 && (
                        <div className="model-selection">
                            <label>Fine-tuned Model:</label>
                            <select 
                                value={selectedFineTunedModel} 
                                onChange={(e) => {
                                    setSelectedFineTunedModel(e.target.value);
                                    setSelectedBaseModel(''); // Clear base model selection when fine-tuned is selected
                                }}
                            >
                                <option value="">Select a fine-tuned model</option>
                                {savedModels.map(model => (
                                    <option key={model.path} value={model.path}>
                                        {model.name} ({model.original_model})
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}
                </div>
                <div className="api-format-toggle">
                    <label>
                        <input
                            type="checkbox"
                            checked={useOpenAIFormat}
                            onChange={(e) => setUseOpenAIFormat(e.target.checked)}
                        />
                        Use OpenAI Format
                    </label>
                </div>
                <div className="generation-params">
                    <div className="chat-param-group">
                        <label>Temperature: {temperature}</label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        />
                    </div>
                    <div className="chat-param-group">
                        <label>Max Tokens:</label>
                        <input
                            type="number"
                            min="1"
                            max="2048"
                            value={maxTokens}
                            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                        />
                    </div>
                </div>
            </div>

            <div className="messages-container">
                {messages.map((message, index) => (
                    <div 
                        key={index} 
                        className={`message ${message.role}`}
                    >
                        <div className="message-role">{message.role}:</div>
                        <div className="message-content">{message.content}</div>
                    </div>
                ))}
                {isLoading && (
                    <div className="message assistant">
                        <div className="message-role">assistant:</div>
                        <div className="message-content loading">Thinking...</div>
                    </div>
                )}
                {error && (
                    <div className="error-message">
                        Error: {error}
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="input-form">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    disabled={isLoading}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSubmit(e);
                        }
                    }}
                />
                <button type="submit" disabled={isLoading || !activeModel}>
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </form>
        </div>
    );
}

export default Chat; 