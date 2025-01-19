document.addEventListener('DOMContentLoaded', () => {
  const wellDropdown = document.getElementById('wellDropdown');
  const filterButton = document.getElementById('filterButton');
  const chartElement = document.getElementById('chart');
  const loadingElement = document.getElementById('loading');
  const noDataElement = document.getElementById('noData');



  // Initialize chart with updated configuration
  let chart = new ApexCharts(chartElement, {
    chart: {
      type: 'line',
      height: 400,
      zoom: {
        // enabled: true, // Enable zooming for better exploration
      },
    },
    series: [], // Series data will be dynamically updated
    xaxis: {
      type: 'datetime',
      labels: {
        format: 'MMM yyyy', // Format date labels for better readability
      },
      title: {
        text: 'Date',
      },
    },
    yaxis: [
      {
        title: {
          text: 'Oil',
        },
        labels: {
          formatter: function (value) {
            return value.toFixed(2); // Format nilai hingga 2 desimal
          },
        },
        // min: Math.min(...data.historical.map((d) => d.Production)) * 0.9, // Sesuaikan batas bawah
        // max: Math.max(...data.historical.map((d) => d.Production)) * 1.1, // Sesuaikan batas atas
      },
      {
        opposite: true,
        title: {
          text: 'Fluid',
        },
        labels: {
          formatter: function (value) {
            return value.toFixed(2); // Format nilai hingga 2 desimal
          },
        },
      },
    ],
    legend: {
      show: true,
      // position: 'top',
      horizontalAlign: 'center', // Membuat legenda dalam satu baris
      labels: {
        useSeriesColors: true,
      },
      markers: {
        width: 12,
        height: 12,
        radius: 12,
      },
    },
    toolbar: {
      tools: {
        selection: true, // Aktifkan brush selection
        zoom: true
      }
    },
    events: {
      selection: function (chartContext, { xaxis }) {
        // Tangkap range waktu yang dipilih (start dan end date)
        const startDate = new Date(xaxis.min).toISOString().split('T')[0];
        const endDate = new Date(xaxis.max).toISOString().split('T')[0];
        // Filter data berdasarkan range waktu yang dipilih
        const selectedData = chartContext.opts.series[0].data.filter(point => {
          const date = new Date(point.x);
          return date >= new Date(startDate) && date <= new Date(endDate);
        });

        // console.log("Selected Data:", selectedData); // Debugging untuk melihat data yang dipilih

        // Simpan data yang dipilih dalam variable global (opsional)
        window.selectedData = selectedData;

        // Tampilkan alert atau lanjutkan proses
        if (selectedData.length === 0) {
          alert("No data selected in the given range.");
        } else {
          alert(`Selected ${selectedData.length} data points.`);
        }

        // Trigger backend call with selected data
        fetchPredictionWithSelectedData(selectedData, startDate, endDate);
      }
    },
    markers: {
      size: [4, 5], // Customize marker size for better visualization
      strokeWidth: 2, // Add a stroke width for better visibility
    },
    tooltip: {
      shared: true, // Tooltip shows all data points for the hovered date
      intersect: false, // Ensure tooltip works well for close points
      custom: function ({series, seriesIndex, dataPointIndex, w}) {
        const pointData = w.config.series[seriesIndex].data[dataPointIndex];
        const seriesName = w.config.series[seriesIndex].name;
        // Format date as "2 May 2024"
        const formattedDate = new Date(pointData.x).toLocaleDateString("en-US", {
          day: "numeric",
          month: "long",
          year: "numeric"
        });
        let displayedValue = pointData?.y;
        if (seriesName == "Job Code") {
          displayedValue = pointData?.name
        }
        return `<div class="custom-tooltip">
        <span class="date">${formattedDate}</span>
        <span>Series : ${seriesName}</span>
        <span>Data   : ${displayedValue}</span>
      </div>
    `;
      }
    },
    title: {
      text: 'Historical Production Data',
      align: 'center',
    },
    annotations: {
      xaxis: [], // Will be dynamically updated for Start, Mid, and End points
    },
  });

  chart.render();

  // Show loading spinner
  const showLoading = () => {
    loadingElement.style.display = 'block';
    noDataElement.style.display = 'none';
    chartElement.style.display = 'none';
  };

  // Hide loading spinner
  const hideLoading = () => {
    loadingElement.style.display = 'none';
    chartElement.style.display = 'block';
  };

  // Show "No Data Found" message
  const showNoData = () => {
    noDataElement.style.display = 'block';
    chartElement.style.display = 'none';
  };

  // Hide "No Data Found" message
  const hideNoData = () => {
    noDataElement.style.display = 'none';
    chartElement.style.display = 'block';
  };

  // Fetch well data from Flask backend
  fetch('http://127.0.0.1:5000/get_wells')
    .then(response => {
      if (!response.ok) {
        throw new Error("Failed to fetch well data");
      }
      return response.json();
    })
    .then(data => {
      // Clear existing options
      wellDropdown.innerHTML = '<option value="">Select...</option>';

      // Populate dropdown with wells
      if (data.wells) {
        data.wells.forEach(well => {
          const option = document.createElement("option");
          option.value = well;
          option.textContent = well;
          wellDropdown.appendChild(option);
        });
      }
    })
    .catch(error => {
      console.error("Error loading wells:", error);
      wellDropdown.innerHTML = '<option value="">Failed to load</option>';
    });

  // // Fetch and display historical data
  // const fetchHistory = (well, startDate, endDate) => {
  //   showLoading(); // Show loading spinner
  //   fetch('http://127.0.0.1:5000/get_history', {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //     body: JSON.stringify({ well, start_date: startDate, end_date: endDate }),
  //   })
  //     .then((response) => response.json())
  //     .then((data) => {
  //       hideLoading(); // Hide loading spinner
  //
  //       if (data.length === 0) {
  //         showNoData(); // Show "No Data Found" if no data
  //       } else {
  //         hideNoData(); // Hide "No Data Found" message
  //
  //         // Series untuk Production (line) pada y-axis pertama (index 0)
  //         const productionSeries = {
  //           name: 'Oil Production',
  //           type: 'line',
  //           data: data.map((entry) => ({
  //             x: entry.Date,
  //             y: entry.Production,
  //           })),
  //           yAxisIndex: 0
  //         };
  //
  //         // Series untuk Fluid (scatter) pada y-axis kedua (index 1)
  //         const fluidSeries = {
  //           name: 'Fluid',
  //           type: 'scatter',
  //           data: data.map((entry) => ({
  //             x: entry.Date,
  //             y: entry.Fluid,
  //           })),
  //           yAxisIndex: 1
  //         };
  //
  //         chart.updateSeries([productionSeries, fluidSeries]);
  //       }
  //     })
  //     .catch((error) => {
  //       hideLoading(); // Hide loading spinner even on error
  //       console.error('Error fetching history:', error);
  //     });
  // };

  const fetchHistory = (well, startDate, endDate) => {
    showLoading(); // Show loading spinner
    fetch('http://127.0.0.1:5000/get_history', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ well, start_date: startDate, end_date: endDate }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Server Error: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        console.log("Data received from backend:", data); // Debugging data received
        hideLoading(); // Hide loading spinner

        if (!data || data.length === 0) {
          console.warn("No data available for the given filters.");
          showNoData(); // Show "No Data Found" if no data
          return;
        }

        hideNoData(); // Hide "No Data Found" message

        // Transform data for chart
        const productionSeries = {
          name: 'Oil',
          type: 'line',
          data: data.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Production,
          })),
          yAxisIndex: 0,
        };

        const fluidSeries = {
          name: 'Fluid',
          type: 'scatter', // Use scatter for fluid
          data: data.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Fluid,
          })),
          yAxisIndex: 1,
        };

        const jobCodeSeries = {
          name: "Job Code",
          type: "scatter",
          group: "default",
          color: '#ffb01a',
          data: data.filter((it) => !!it?.JobCode).map((entry) => ({
            x: entry.Date,
            y: 2500,
            name: entry.JobCode
          }))
        }

        console.log("Production series:", productionSeries); // Debugging series data
        console.log("Fluid series:", fluidSeries); // Debugging series data

        // Update chart series
        chart.updateSeries([productionSeries, fluidSeries, jobCodeSeries])
          .then(() => {
            console.log("Chart updated successfully.", chart.opts.series);
          })
          .catch((error) => {
            console.error("Error updating chart series:", error);
          });

        console.log("Chart Options after update:", chart.opts);
      })
      .catch((error) => {
        hideLoading(); // Hide loading spinner even on error
        console.error('Error fetching history:', error);
        showNoData(); // Optionally show "No Data Found" on error
      });
  };



  // Fetch default data (last 12 months) on page load
  fetchHistory();

  // Filter data based on user input
  filterButton.addEventListener('click', () => {
    const selectedWell = wellDropdown.value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    fetchHistory(selectedWell, startDate, endDate);
  });
/*
  const analyzeDCAButton = document.getElementById('analyzeDCA');

  analyzeDCAButton.addEventListener('click', () => {
    const selectedWell = wellDropdown.value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    // Show loading spinner
    showLoading();

    // Fetch DCA analysis
    fetch('http://127.0.0.1:5000/calculate_dca3', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ well: selectedWell, start_date: startDate, end_date: endDate }),
    })
      .then((response) => response.json())
      .then((data) => {
        hideLoading();

        if (data.error) {
          console.error(data.error);
          alert('Error calculating DCA');
          return;
        }

        // Update start, mid, and end point information
        document.getElementById('startPoint').innerText = data.start_date;
        document.getElementById('midPoint').innerText = data.mid_date;
        document.getElementById('endPoint').innerText = data.end_date;

        // Update chart with DCA analysis
        chart.updateSeries([
          {
            name: 'Historical Data',
            data: data.historical.map((entry) => ({
              x: entry.Date,
              y: entry.Production,
            })),
          },
          {
            name: 'Exponential Decline',
            data: data.exponential.map((entry) => ({
              x: entry.Date,
              y: entry.Production,
            })),
          },
          {
            name: 'Harmonic Decline',
            data: data.harmonic.map((entry) => ({
              x: entry.Date,
              y: entry.Production,
            })),
          },
          {
            name: 'Hyperbolic Decline',
            data: data.hyperbolic.map((entry) => ({
              x: entry.Date,
              y: entry.Production,
            })),
          },
        ]);

        // Highlight start, mid, and end points
        const annotations = [
          {
            x: new Date(data.start_date).getTime(),
            borderColor: '#00E396',
            label: {
              text: 'Start Point',
              style: { color: '#fff', background: '#00E396' },
            },
          },
          {
            x: new Date(data.mid_date).getTime(),
            borderColor: '#FEB019',
            label: {
              text: 'Mid Point',
              style: { color: '#fff', background: '#FEB019' },
            },
          },
          {
            x: new Date(data.end_date).getTime(),
            borderColor: '#FF4560',
            label: {
              text: 'End Point',
              style: { color: '#fff', background: '#FF4560' },
            },
          },
        ];
        chart.updateOptions({
          annotations: { xaxis: annotations },
        });
      })
      .catch((error) => {
        hideLoading();
        console.error('Error fetching DCA:', error);
        alert('An unexpected error occurred. Please try again.');
      });
  });
*/

  document.getElementById('automateDCA').addEventListener('click', function () {
    const selectedWell = document.getElementById('wellDropdown').value;

    if (!selectedWell) {
      alert("Please select a well.");
      return;
    }

    const loadingMessage = document.getElementById('loadingMessage');
    loadingMessage.style.display = 'block';

    fetch('http://127.0.0.1:5000/automatic_dca', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ well: selectedWell })
    })
      .then(response => response.json())
      .then(data => {
        loadingMessage.style.display = 'none';

        if (data.error) {
          alert(`Error: ${data.error}`);
          return;
        }

        const actualData = data.ActualData.map(point => ({
          x: new Date(point.date),
          y: point.value
        }));

        const fluidData = data.ActualData.map(point => ({
          x: new Date(point.date),
          y: point.fluid
        }));

        const startDate = new Date(data.StartDate);
        const endDate = new Date(data.EndDate);

        const exponentialModel = (t, qi, b) => qi * Math.exp(-b * t);
        const harmonicModel = (t, qi, b) => qi / (1 + b * t);
        const hyperbolicModel = (t, qi, b, n) => qi * Math.pow(1 + b * t, -1 / n);

        const generatePrediction = (modelFunction, params, startDate, endDate) => {
          const predictions = [];
          let t = 0;
          let currentDate = new Date(startDate);

          while (currentDate <= endDate) {
            const predictionValue = modelFunction(t, ...params);
            predictions.push({
              x: new Date(currentDate),
              y: predictionValue
            });
            t += 1; // Increment time by 1 day
            currentDate.setDate(currentDate.getDate() + 1); // Move to the next day
          }

          return predictions;
        };

        const exponentialData = generatePrediction(exponentialModel, data.Exponential, startDate, endDate);
        const harmonicData = generatePrediction(harmonicModel, data.Harmonic, startDate, endDate);
        const hyperbolicData = generatePrediction(hyperbolicModel, data.Hyperbolic, startDate, endDate);

        chart.updateSeries([
          { name: 'Actual Data', data: actualData },
          { name: 'Fluid Data', data: fluidData },
          { name: 'Exponential Decline', type: 'line', data: exponentialData },
          { name: 'Harmonic Decline', type: 'line', data: harmonicData },
          { name: 'Hyperbolic Decline', type: 'line', data: hyperbolicData }
        ]);

        // Hide certain series by default
        chart.hideSeries('Fluid Data'); // Hide Fluid Data
        chart.hideSeries('Harmonic Decline'); // Hide Harmonic Decline
        chart.hideSeries('Hyperbolic Decline'); // Hide Hyperbolic Decline

      })
      .catch(error => {
        loadingMessage.style.display = 'none';
        console.error('Error:', error);
        alert('An unexpected error occurred. Please try again.');
      });
  });


  const fetchPredictionWithSelectedData = (selectedData, startDate, endDate) => {
    showLoading();
    const selectedWell = document.getElementById('wellDropdown').value;

    if (!selectedWell) {
      alert("Please select a well.");
      return;
    }
    fetch('http://127.0.0.1:5000/predict_production', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        well: selectedWell, // Kirimkan nama well
        start_date: startDate, // Kirimkan start date
        end_date: endDate, // Kirimkan end date
        selected_data: selectedData // Kirimkan data yang dipilih
      })
    })
      .then(response => response.json())
      .then(data => {
        hideLoading();

        if (data.error) {
          alert(`Error: ${data.error}`);
          return;
        }

        // Update chart dengan prediksi baru
        const exponentialData = data.Predictions.Exponential.map(point => ({
          x: new Date(point.date),
          y: point.value
        }));
        const harmonicData = data.Predictions.Harmonic.map(point => ({
          x: new Date(point.date),
          y: point.value
        }));
        const hyperbolicData = data.Predictions.Hyperbolic.map(point => ({
          x: new Date(point.date),
          y: point.value
        }));

        chart.updateSeries([
          { name: 'Actual Data', type: 'scatter', data: selectedData },
          { name: 'Exponential Decline', type: 'line', data: exponentialData },
          { name: 'Harmonic Decline', type: 'line', data: harmonicData },
          { name: 'Hyperbolic Decline', type: 'line', data: hyperbolicData }
        ]);
      })
      .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('An unexpected error occurred. Please try again.');
      });
  };


// Event listener untuk tombol Predict DCA
  const predictDCAButton = document.getElementById('predictDCA');

  predictDCAButton.addEventListener('click', () => {
    const selectedWell = document.getElementById('wellDropdown').value;
    const elr = document.getElementById('elr').value || 5; // Default ELR = 5 jika kosong
    let selectedData = window.selectedData || []; // Data dari brush selection

    if (!selectedWell) {
      alert("Please select a well to predict.");
      return;
    }

    if (selectedData.length === 0) {
      // Jika tidak ada data yang dipilih, ambil dua data terakhir dari grafik
      const allData = chart.opts.series[0]?.data || [];
      console.log("Chart Options:", chart.opts);
      console.log("Series in Chart:", chart.opts.series);
      console.log('all data:', allData);
      if (allData.length >= 2) {
        selectedData = allData.slice(-5); // Ambil dua data terakhir
      } else {
        alert("Not enough data available for prediction.");
        return;
      }
    }

    console.log("Selected Data for Prediction:", selectedData);

    // Panggil fungsi fetchPredictionWithSelectedData
    fetchPredictionWithSelectedData(selectedData, selectedWell, elr);
  });



});
