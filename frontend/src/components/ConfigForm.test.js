import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConfigForm from './ConfigForm';

const buildProps = (overrides = {}) => ({
  file: { dataset_id: 'dataset-1', name: 'Dataset 1' },
  structure: { fields: ['input', 'output'] },
  currentConfig: null,
  onSave: jest.fn(),
  isConfigured: false,
  ...overrides
});

describe('ConfigForm', () => {
  it('saves selected input/output fields', async () => {
    const onSave = jest.fn();
    render(<ConfigForm {...buildProps({ onSave })} />);

    const [inputSelect, outputSelect] = screen.getAllByRole('combobox');
    await userEvent.selectOptions(inputSelect, 'input');
    await userEvent.selectOptions(outputSelect, 'output');
    await userEvent.click(screen.getByRole('button', { name: 'Save Configuration' }));

    expect(onSave).toHaveBeenCalledWith('dataset-1', {
      inputField: 'input',
      outputField: 'output'
    });
  });

  it('auto-selects single field and disables output select', async () => {
    const onSave = jest.fn();
    render(
      <ConfigForm
        {...buildProps({
          onSave,
          structure: { fields: ['onlyField'] }
        })}
      />
    );

    const [inputSelect, outputSelect] = screen.getAllByRole('combobox');
    await waitFor(() => {
      expect(inputSelect).toHaveValue('onlyField');
    });

    expect(outputSelect).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Save Configuration' }));

    expect(onSave).toHaveBeenCalledWith('dataset-1', {
      inputField: 'onlyField',
      outputField: 'onlyField'
    });
  });
});
