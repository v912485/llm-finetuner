import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Settings from './Settings';

const mockFetchResponse = (data) => ({
  json: async () => data
});

describe('Settings', () => {
  beforeEach(() => {
    localStorage.clear();
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it('shows configured status based on server responses', async () => {
    global.fetch
      .mockResolvedValueOnce(mockFetchResponse({ configured: false }))
      .mockResolvedValueOnce(mockFetchResponse({ configured: true }))
      .mockResolvedValueOnce(
        mockFetchResponse({
          status: 'success',
          llama_cpp_dir: '',
          gguf_converter_path: '',
          gguf_outtype: 'f16'
        })
      );

    render(<Settings />);

    await waitFor(() => {
      screen.getByText('Token not configured');
      screen.getByText('Admin token configured');
    });
    expect(screen.getByText('Token not configured')).toBeInTheDocument();
    expect(screen.getByText('Admin token configured')).toBeInTheDocument();
  });

  it('validates admin token input', async () => {
    global.fetch
      .mockResolvedValueOnce(mockFetchResponse({ configured: false }))
      .mockResolvedValueOnce(mockFetchResponse({ configured: false }))
      .mockResolvedValueOnce(
        mockFetchResponse({
          status: 'success',
          llama_cpp_dir: '',
          gguf_converter_path: '',
          gguf_outtype: 'f16'
        })
      );

    render(<Settings />);

    await userEvent.click(screen.getByRole('button', { name: 'Save Admin Token' }));

    expect(screen.getByText('Admin token is required')).toBeInTheDocument();
  });

  it('saves admin token when provided', async () => {
    global.fetch
      .mockResolvedValueOnce(mockFetchResponse({ configured: false }))
      .mockResolvedValueOnce(mockFetchResponse({ configured: false }))
      .mockResolvedValueOnce(
        mockFetchResponse({
          status: 'success',
          llama_cpp_dir: '',
          gguf_converter_path: '',
          gguf_outtype: 'f16'
        })
      )
      .mockResolvedValueOnce(mockFetchResponse({ status: 'success' }));

    render(<Settings />);

    await userEvent.type(
      screen.getByPlaceholderText('Enter admin token'),
      'admin-token-123'
    );
    await userEvent.click(screen.getByRole('button', { name: 'Save Admin Token' }));

    await waitFor(() => {
      expect(screen.getByText('Admin token saved')).toBeInTheDocument();
    });
    expect(localStorage.getItem('adminToken')).toBe('admin-token-123');
  });
});
