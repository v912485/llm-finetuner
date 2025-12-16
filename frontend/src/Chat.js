import React, { useState, useEffect, useRef } from 'react';
import './Chat.css';
import { useAppContext } from './context/AppContext';
import ModelSelector from './components/ModelSelector';

function Chat() {
    const { api, fetchSavedModels, isTraining } = useAppContext();
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [selectedBaseModel, setSelectedBaseModel] = useState('');
    const [selectedFineTunedModel, setSelectedFineTunedModel] = useState('');
    const [useOpenAIFormat, setUseOpenAIFormat] = useState(true);
    const [systemPrompt, setSystemPrompt] = useState('');
    const [temperature, setTemperature] = useState(0.7);
    const [maxTokens, setMaxTokens] = useState(4096);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);
    const wasTrainingRef = useRef(false);

    // Get the currently active model
    const activeModel = selectedFineTunedModel || selectedBaseModel;

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Refresh saved models when training completes
    useEffect(() => {
        if (wasTrainingRef.current && !isTraining) {
            fetchSavedModels();
        }
        wasTrainingRef.current = isTraining;
    }, [isTraining, fetchSavedModels]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
            
            if (useOpenAIFormat) {
                const systemMessage = systemPrompt.trim()
                    ? [{ role: 'system', content: systemPrompt.trim() }]
                    : [];
                const requestBody = {
                    model: activeModel,
                    messages: [...systemMessage, ...messages, newMessage],
                    temperature: temperature,
                    max_tokens: maxTokens,
                };

                response = await api.post(
                    '/models/v1/chat/completions', 
                    requestBody
                );
                
                console.log('Chat completions response:', response);
                
                if (response && response.choices && response.choices[0]?.message) {
                    const assistantMessage = {
                        role: 'assistant',
                        content: response.choices[0].message.content
                    };
                    setMessages(prev => [...prev, assistantMessage]);
                } else if (response && response.error && response.error.message) {
                    throw new Error(response.error.message);
                } else {
                    throw new Error('Failed to get response');
                }
            } else {
                const requestBody = selectedFineTunedModel
                    ? { saved_model_name: selectedFineTunedModel }
                    : { model_id: selectedBaseModel };

                requestBody.input = input;
                requestBody.temperature = temperature;
                requestBody.max_length = maxTokens;
                requestBody.do_sample = true;
                requestBody.top_p = 0.95;

                response = await api.post(
                    '/models/inference', 
                    requestBody
                );
                
                console.log('Standard inference response:', response);
                
                if (response && response.status === 'success' && response.response) {
                    const assistantMessage = {
                        role: 'assistant',
                        content: response.response
                    };
                    setMessages(prev => [...prev, assistantMessage]);
                } else {
                    throw new Error(response.message || 'Failed to get response');
                }
            }
        } catch (err) {
            console.error('Error during inference:', err);
            setError(err.message || 'An error occurred during inference');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClearChat = () => {
        setMessages([]);
    };

    return (
        <div className="chat-container">
            <div className="chat-sidebar">
                <h2>Chat Settings</h2>
                
                <div className="model-selection">
                    <ModelSelector
                        selectedModel={selectedBaseModel}
                        onModelSelect={setSelectedBaseModel}
                        label="Base Model"
                        includeSavedModels={false}
                    />
                    
                    <ModelSelector
                        selectedModel={selectedFineTunedModel}
                        onModelSelect={setSelectedFineTunedModel}
                        label="Fine-tuned Model"
                        includeBaseModels={false}
                    />
                </div>
                
                <div className="chat-settings">
                    <div className="setting-group">
                        <label>
                            <input
                                type="checkbox"
                                checked={useOpenAIFormat}
                                onChange={(e) => setUseOpenAIFormat(e.target.checked)}
                            />
                            Use OpenAI Format
                        </label>
                    </div>

                    <div className="setting-group">
                        <label htmlFor="system-prompt">System Prompt (optional)</label>
                        <textarea
                            id="system-prompt"
                            value={systemPrompt}
                            onChange={(e) => setSystemPrompt(e.target.value)}
                            placeholder="e.g. You are a classifier... Output ONLY valid JSON..."
                            rows={6}
                            disabled={!activeModel}
                        />
                    </div>
                    
                    <div className="setting-group">
                        <label>Temperature: {temperature}</label>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        />
                    </div>
                    
                    <div className="setting-group">
                        <label>Max Tokens: {maxTokens}</label>
                        <input
                            type="range"
                            min="1"
                            max="8192"
                            step="1"
                            value={maxTokens}
                            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                        />
                    </div>
                    
                    <button className="clear-chat-btn" onClick={handleClearChat}>
                        Clear Chat
                    </button>
                </div>
            </div>
            
            <div className="chat-main">
                <div className="chat-messages">
                    {messages.length === 0 ? (
                        <div className="empty-chat">
                            <p>Select a model and start chatting!</p>
                        </div>
                    ) : (
                        messages.map((msg, index) => (
                            <div key={index} className={`message ${msg.role}`}>
                                <div className="message-content">{msg.content}</div>
                            </div>
                        ))
                    )}
                    {isLoading && (
                        <div className="message assistant loading">
                            <div className="loading-indicator">
                                <div className="dot"></div>
                                <div className="dot"></div>
                                <div className="dot"></div>
                            </div>
                        </div>
                    )}
                    {error && (
                        <div className="error-message">
                            <p>{error}</p>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
                
                <form className="chat-input" onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type your message..."
                        disabled={isLoading || !activeModel}
                    />
                    <button 
                        type="submit" 
                        disabled={isLoading || !input.trim() || !activeModel}
                    >
                        Send
                    </button>
                </form>
            </div>
        </div>
    );
}

export default Chat; 