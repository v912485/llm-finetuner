import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { 
    Send, 
    Trash2, 
    Copy, 
    Check, 
    User, 
    Bot, 
    Settings as SettingsIcon,
    ChevronDown,
    ChevronUp,
    Sparkles
} from 'lucide-react';
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
    const [systemPrompt, setSystemPrompt] = useState(() => localStorage.getItem('chatSystemPrompt') || '');
    const [temperature, setTemperature] = useState(() => {
        const saved = localStorage.getItem('chatTemperature');
        return saved !== null ? parseFloat(saved) : 0.7;
    });
    const [maxTokens, setMaxTokens] = useState(() => {
        const saved = localStorage.getItem('chatMaxTokens');
        return saved !== null ? parseInt(saved, 10) : 4096;
    });
    const [showSettings, setShowSettings] = useState(true);
    const [copiedId, setCopiedId] = useState(null);
    const messagesEndRef = useRef(null);
    const textareaRef = useRef(null);
    const wasTrainingRef = useRef(false);
    const messageIdRef = useRef(0);
    const getNextMessageId = () => `${Date.now()}-${messageIdRef.current++}`;

    // Persist settings to localStorage
    useEffect(() => {
        localStorage.setItem('chatSystemPrompt', systemPrompt);
    }, [systemPrompt]);

    useEffect(() => {
        localStorage.setItem('chatTemperature', temperature.toString());
    }, [temperature]);

    useEffect(() => {
        localStorage.setItem('chatMaxTokens', maxTokens.toString());
    }, [maxTokens]);

    // Get the currently active model
    const activeModel = selectedFineTunedModel || selectedBaseModel;

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    // Refresh saved models when training completes
    useEffect(() => {
        if (wasTrainingRef.current && !isTraining) {
            fetchSavedModels();
        }
        wasTrainingRef.current = isTraining;
    }, [isTraining, fetchSavedModels]);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [input]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const handleSubmit = async (e) => {
        if (e) e.preventDefault();
        if (!input.trim() || !activeModel || isLoading) return;

        const newMessage = { role: 'user', content: input, id: getNextMessageId() };
        setMessages(prev => [...prev, newMessage]);
        setInput('');
        setIsLoading(true);
        
        try {
            const systemMessage = systemPrompt.trim()
                ? [{ role: 'system', content: systemPrompt.trim() }]
                : [];
            
            // Filter out any previous error messages from the context
            const chatHistory = messages.filter(m => m.role !== 'error');

            const requestBody = {
                model: activeModel,
                messages: [...systemMessage, ...chatHistory, newMessage],
                temperature: temperature,
                max_tokens: maxTokens,
            };

            const response = await api.post(
                '/models/v1/chat/completions', 
                requestBody
            );
            
            if (response && response.choices && response.choices[0]?.message) {
                const assistantMessage = {
                    role: 'assistant',
                    content: response.choices[0].message.content,
                    id: getNextMessageId()
                };
                setMessages(prev => [...prev, assistantMessage]);
            } else if (response && response.error && response.error.message) {
                throw new Error(response.error.message);
            } else {
                throw new Error('Failed to get response');
            }
        } catch (err) {
            console.error('Error during inference:', err);
            // Add error to messages for better visibility in chat
            setMessages(prev => [...prev, { role: 'error', content: err.message || 'An error occurred', id: getNextMessageId() }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    const handleClearChat = () => {
        if (window.confirm('Are you sure you want to clear the chat history?')) {
            setMessages([]);
        }
    };

    const copyToClipboard = (text, id) => {
        navigator.clipboard.writeText(text).then(() => {
            setCopiedId(id);
            setTimeout(() => setCopiedId(null), 2000);
        });
    };

    const renderMessage = (msg, index) => {
        if (msg.role === 'error') {
            const messageId = msg.id ?? index;
            return (
                <div key={messageId} className="message error">
                    <div className="message-icon"><Trash2 size={16} /></div>
                    <div className="message-content">{msg.content}</div>
                </div>
            );
        }

        const messageId = msg.id ?? index;
        return (
            <div key={messageId} className={`message ${msg.role}`}>
                <div className="message-header">
                    <div className="message-author">
                        {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                        <span>{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                    </div>
                    {msg.role === 'assistant' && (
                        <button 
                            className="copy-btn" 
                            onClick={() => copyToClipboard(msg.content, messageId)}
                            title="Copy to clipboard"
                        >
                            {copiedId === messageId ? <Check size={14} /> : <Copy size={14} />}
                        </button>
                    )}
                </div>
                <div className="message-content">
                    {msg.role === 'assistant' ? (
                        <ReactMarkdown
                            components={{
                                code({ node, inline, className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || '');
                                    return !inline && match ? (
                                        <SyntaxHighlighter
                                            style={vscDarkPlus}
                                            language={match[1]}
                                            PreTag="div"
                                            {...props}
                                        >
                                            {String(children).replace(/\n$/, '')}
                                        </SyntaxHighlighter>
                                    ) : (
                                        <code className={className} {...props}>
                                            {children}
                                        </code>
                                    );
                                }
                            }}
                        >
                            {msg.content}
                        </ReactMarkdown>
                    ) : (
                        <div className="user-text">{msg.content}</div>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className="chat-container">
            <aside className={`chat-sidebar ${!showSettings ? 'collapsed' : ''}`}>
                <div className="sidebar-header" onClick={() => setShowSettings(!showSettings)}>
                    <h2><SettingsIcon size={20} /> Chat Settings</h2>
                    <button className="toggle-sidebar">
                        {showSettings ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                    </button>
                </div>
                
                {showSettings && (
                    <div className="sidebar-content">
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
                                <label htmlFor="system-prompt">System Prompt</label>
                                <textarea
                                    id="system-prompt"
                                    value={systemPrompt}
                                    onChange={(e) => setSystemPrompt(e.target.value)}
                                    placeholder="e.g. You are a helpful assistant..."
                                    rows={4}
                                    disabled={!activeModel}
                                />
                            </div>
                            
                            <div className="setting-group">
                                <div className="setting-header">
                                    <label>Temperature</label>
                                    <span className="setting-value">{temperature}</span>
                                </div>
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
                                <div className="setting-header">
                                    <label>Max Tokens</label>
                                    <span className="setting-value">{maxTokens}</span>
                                </div>
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
                                <Trash2 size={16} /> Clear Chat
                            </button>
                        </div>
                    </div>
                )}
            </aside>
            
            <main className="chat-main">
                <div className="chat-messages">
                    {messages.length === 0 ? (
                        <div className="empty-chat">
                            <Sparkles size={48} className="empty-icon" />
                            <h3>Ready to Chat</h3>
                            <p>Select a model from the settings to start a conversation.</p>
                            {!activeModel && (
                                <div className="setup-hint">
                                    Please select a model in the sidebar.
                                </div>
                            )}
                        </div>
                    ) : (
                        messages.map((msg, index) => renderMessage(msg, index))
                    )}
                    {isLoading && (
                        <div className="message assistant loading">
                            <div className="message-header">
                                <div className="message-author">
                                    <Bot size={16} />
                                    <span>Assistant is thinking...</span>
                                </div>
                            </div>
                            <div className="loading-indicator">
                                <div className="dot"></div>
                                <div className="dot"></div>
                                <div className="dot"></div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
                
                <div className="chat-input-wrapper">
                    <form className="chat-input" onSubmit={handleSubmit}>
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={activeModel ? "Type your message... (Shift+Enter for new line)" : "Select a model to start..."}
                            disabled={isLoading || !activeModel}
                            rows={1}
                        />
                        <button 
                            type="submit" 
                            disabled={isLoading || !input.trim() || !activeModel}
                            className="send-btn"
                        >
                            <Send size={20} />
                        </button>
                    </form>
                    {activeModel && (
                        <div className="active-model-info">
                            Using model: <strong>{activeModel}</strong>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

export default Chat; 