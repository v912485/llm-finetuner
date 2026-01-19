import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ModelSelector from './ModelSelector';
import { useAppContext } from '../context/AppContext';

jest.mock('../context/AppContext', () => ({
  useAppContext: jest.fn()
}));

describe('ModelSelector', () => {
  beforeEach(() => {
    useAppContext.mockReturnValue({
      downloadedModels: ['base-1'],
      savedModels: [{ name: 'saved-1' }],
      fetchSavedModels: jest.fn().mockResolvedValue({}),
      fetchDownloadedModels: jest.fn().mockResolvedValue({})
    });
  });

  it('renders models and notifies on selection', async () => {
    const handleSelect = jest.fn();

    render(
      <ModelSelector
        selectedModel=""
        onModelSelect={handleSelect}
        label="Select Model"
      />
    );

    expect(screen.getByRole('option', { name: 'base-1' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'saved-1' })).toBeInTheDocument();

    await userEvent.selectOptions(screen.getByRole('combobox'), 'base-1');
    expect(handleSelect).toHaveBeenCalledWith('base-1');
  });

  it('refreshes model lists on click', async () => {
    const fetchSavedModels = jest.fn().mockResolvedValue({});
    const fetchDownloadedModels = jest.fn().mockResolvedValue({});

    useAppContext.mockReturnValue({
      downloadedModels: ['base-1'],
      savedModels: [{ name: 'saved-1' }],
      fetchSavedModels,
      fetchDownloadedModels
    });

    render(
      <ModelSelector
        selectedModel=""
        onModelSelect={jest.fn()}
        label="Select Model"
      />
    );

    await userEvent.click(screen.getByRole('button', { name: '↻' }));
    expect(fetchSavedModels).toHaveBeenCalledTimes(1);
    expect(fetchDownloadedModels).toHaveBeenCalledTimes(1);
  });
});
