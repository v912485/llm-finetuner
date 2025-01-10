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
          }
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