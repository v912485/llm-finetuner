import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

const TrainingGraph = ({ history }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (!history || history.length === 0) return;

    const ctx = chartRef.current.getContext('2d');

    // Destroy previous chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const metricSeries = [
      { key: 'accuracy', label: 'Accuracy', color: 'rgb(54, 162, 235)' },
      { key: 'bleu', label: 'BLEU', color: 'rgb(153, 102, 255)' },
      { key: 'rouge1', label: 'ROUGE-1', color: 'rgb(255, 159, 64)' },
      { key: 'rouge2', label: 'ROUGE-2', color: 'rgb(201, 203, 207)' },
      { key: 'rougeL', label: 'ROUGE-L', color: 'rgb(75, 192, 192)' },
    ];

    const metricDatasets = metricSeries
      .filter(series => history.some(h => typeof h[series.key] === 'number'))
      .map(series => ({
        label: series.label,
        data: history.map(h => (typeof h[series.key] === 'number' ? h[series.key] : null)),
        borderColor: series.color,
        borderDash: [6, 4],
        tension: 0.1,
        yAxisID: 'y2',
      }));

    // Create new chart
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: history.map(h => h.epoch),
        datasets: [
          {
            label: 'Training Loss',
            data: history.map(h => h.train_loss),
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
          },
          {
            label: 'Validation Loss',
            data: history.map(h => h.val_loss),
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1
          },
          ...metricDatasets
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Training Progress'
          },
          tooltip: {
            mode: 'index',
            intersect: false,
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Epoch'
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Loss'
            }
          },
          y2: {
            display: metricDatasets.length > 0,
            position: 'right',
            min: 0,
            max: 1,
            grid: {
              drawOnChartArea: false
            },
            title: {
              display: true,
              text: 'Metrics'
            }
          }
        }
      }
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [history]);

  return (
    <div className="training-graph">
      <canvas ref={chartRef}></canvas>
    </div>
  );
};

export default TrainingGraph; 