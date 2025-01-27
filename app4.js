document.addEventListener('DOMContentLoaded', () => {
  const wellDropdown = document.getElementById('wellDropdown');
  const filterButton = document.getElementById('filterButton');
  const chartElement = document.getElementById('chart');
  const loadingElement = document.getElementById('loading');
  const noDataElement = document.getElementById('noData');
  let productionData = [];
  let automaticDataSeries = [];
  let selectedPredictData = [];
  let currentState = '';
  let selectedPredictObject = undefined;
  let currentDataSeries = [];

  // Initialize chart with updated configuration
  let chart = new ApexCharts(chartElement, {
    chart: {
      type: 'line',
      height: 400,
      zoom: {
        // enabled: true, // Enable zooming for better exploration
      },
      events: {
        markerClick: function (event, chartContext, opts) {
          console.log("Marker Click", opts)
          const selectedIndex = opts.dataPointIndex;
          // production series only
          if(opts.seriesIndex == 0){
            const item = productionData[selectedIndex];
            console.log("Selected Data:", item);
            updateSelectedPredictData({
              ...item,
              dataPointIndex: selectedIndex
            })
          }
          
        },
      }
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
        seriesName:"Oil",
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
        seriesName:"Fluid",
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
      // {
      //   seriesName:"Exponential Decline",
      //   title: {
      //     text: 'Exponential Decline',
      //   },
      //   labels: {
      //     formatter: function (value) {
      //       return value.toFixed(2); // Format nilai hingga 2 desimal
      //     },
      //   },
      //   min: 0,
      //   max: 150,
      // },
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
    colors: [
      '#1f77b4', // Blue
      '#ff7f0e', // Orange
      '#2ca02c', // Green
      '#d62728', // Red
      '#9467bd', // Purple
      '#8c564b', // Brown
      '#e377c2', // Pink
      '#7f7f7f', // Gray
      '#bcbd22', // Olive
      '#17becf', // Cyan
      '#aec7e8', // Light Blue
      '#ffbb78', // Light Orange
      '#98df8a', // Light Green
      '#ff9896', // Light Red
      '#c5b0d5', // Light Purple
    ],
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
      size: [5,5, 5,5,0,0,0,0,0,0], // Customize marker size for better visualization
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

  function updateSelectedPredictData(data) {
    console.log("Data selected:", data);
    if(currentState !== 'prediction'){
      const currentDataPointIndexes = selectedPredictData.map((it) => it.dataPointIndex);
    if(currentDataPointIndexes.includes(data.dataPointIndex)){
      // remove data
      selectedPredictData = selectedPredictData.filter((it) => it.dataPointIndex !== data.dataPointIndex);
    }else {
      // add data
      selectedPredictData.push(data);
      // sort selectedPredictData by dataPointIndex
      selectedPredictData.sort((a, b) => a.dataPointIndex - b.dataPointIndex);
    }
    updateSelectedPredictView();
    }else {
      selectedPredictObject = data
      updateSelectedPredictObjectView();
    }
    
  }

  function updateChartMarkerConfig(series){
    const markerSizes = series.map((it) => it.showMarker ? 5 : 0);
    chart.updateOptions({
      markers: {
        size: markerSizes
      }
    })
  }

  function appendUniqueSeries(oldSeries, newSeries){
    let series = [];
    console.log("Old Series:", oldSeries);
    console.log("New Series:", newSeries);
    const newSeriesNames = newSeries.map((it) => it.name);
    oldSeries.forEach((it) => {
      if(!newSeriesNames.includes(it.name)){
        series.push(it);
      }
    })
    newSeries.forEach((it) => {
      series.push(it);
    })

    return series;
  }

  function updateSelectedPredictView(){
    const dates = selectedPredictData.map((it) => it.Date);
    const dateString = dates.join(", ");

    const filteredSeries = currentDataSeries.filter((it) => it.name !== "Selected Automatic Data");
    const newSeries = [
      ...filteredSeries,
      {
        name: "Selected Automatic Data",
        type: "line",
        title: {
          text: "Selected Automatic Data"
        },
        showMarker: true,
        data: selectedPredictData.map((it) => ({
          x: new Date(it.Date),
          y: it.Production,
        })),
      }
    ]
    currentDataSeries = newSeries;
    chart.updateSeries(newSeries)
    updateChartMarkerConfig(newSeries)
    document.getElementById("selectedPredictData").innerText = dateString;
  }

  function updateSelectedPredictObjectView(){
    const dateString = selectedPredictObject?.Date;
    document.getElementById("selectedPredictObject").innerText =dateString;

    const filteredSeries = currentDataSeries.filter((it) => it.name !== "Selected Prediction Data");
    const newSeries = [
      ...filteredSeries,
      {
        name: "Selected Prediction Data",
        type: "line",
        isPrediction: true,
        title: {
          text: "Selected Prediction Data"
        },
        showMarker: true,
        data: [
          {
            x: new Date(selectedPredictObject.Date),
            y: selectedPredictObject.Production
          }
        ],
      }
    ]
    currentDataSeries = newSeries;
    chart.updateSeries(newSeries)
    updateChartMarkerConfig(newSeries)
  }

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
          displayedData = [];
          showNoData(); // Show "No Data Found" if no data
          return;
        }

        hideNoData(); // Hide "No Data Found" message
        
        productionData = data;
        // Transform data for chart
        const productionSeries = {
          name: 'Oil',
          type: 'line',
          data: data.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Production,
          })),
          showMarker: true,
        };

        const fluidSeries = {
          name: 'Fluid',
          type: 'scatter', // Use scatter for fluid
          data: data.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Fluid,
          })),
          showMarker: true,
        };

        const jobCodeSeries = {
          name: "Job Code",
          type: "scatter",
          group: "default",
          data: data.filter((it) => !!it?.JobCode).map((entry) => ({
            x: entry.Date,
            y: 2500,
            name: entry.JobCode
          })),
          showMarker: true,
        }

        console.log("Production series:", productionSeries); // Debugging series data
        console.log("Fluid series:", fluidSeries); // Debugging series data
        currentDataSeries = [
          productionSeries,
          fluidSeries,
          jobCodeSeries
        ]
        // Update chart series
        chart.updateSeries(currentDataSeries)
          .then(() => {
            console.log("Chart updated successfully.", chart.opts.series);
          })
          .catch((error) => {
            console.error("Error updating chart series:", error);
          });
          updateChartMarkerConfig(currentDataSeries)

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
    currentState = '';
    selectedPredictObject = undefined;
    if (!selectedWell) {
      alert("Please select a well.");
      return;
    }

    const loadingMessage = document.getElementById('loadingMessage');
    loadingMessage.style.display = 'block';

    const selected_data = selectedPredictData?.map((it, index) => {
      return {
        Date: it.Date,
        Production: it.Production,
        Fluid: it.Fluid
      }
    }) ?? undefined;

    console.log("Selected Well for Automation:", selectedWell);
    console.log("Selected Data for Automation:", JSON.stringify(selected_data, null, 2));
    

    fetch('http://127.0.0.1:5000/automatic_dca', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ well: selectedWell, selected_data })
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

        const productionSeries = {
          name: 'Oil',
          type: 'line',
          data: productionData.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Production,
          })),
          yAxisIndex: 0,
          showMarker: true,
        };

        const fluidSeries = {
          name: 'Fluid',
          type: 'scatter', // Use scatter for fluid
          hidden: true,
          data: productionData.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Fluid,
          })),
          yAxisIndex: 1,
          showMarker: true,
        };

        const jobCodeSeries = {
          name: "Job Code",
          type: "scatter",
          group: "default",
          hidden: true,
          data: productionData.filter((it) => !!it?.JobCode).map((entry) => ({
            x: entry.Date,
            y: 2500,
            name: entry.JobCode
          })),
          showMarker: true,
        }

        // update current data series, set job code and fluid series hidden
        currentDataSeries = currentDataSeries.map((it) => {
          if(it.name === "Job Code" || it.name === "Fluid"){
            return {
              ...it,
              hidden: true
            }
          }
          return it;
        })

        // remove prediction series
        currentDataSeries = currentDataSeries.filter((it) => !it.isPrediction);
        const newSeries = [
          { name: 'Exponential Decline', type: 'line', data: exponentialData, hidden: false},
          { name: 'Harmonic Decline', type: 'line', data: harmonicData, hidden:true},
          { name: 'Hyperbolic Decline', type: 'line', data: hyperbolicData, hidden:true }
        ]
        const finalSeries = appendUniqueSeries(currentDataSeries, newSeries);
        currentDataSeries = finalSeries;

        chart.updateSeries(finalSeries);
        updateChartMarkerConfig(finalSeries)

        // Hide certain series by default
        // chart.hideSeries('Fluid Data'); // Hide Fluid Data
        // chart.hideSeries('Harmonic Decline'); // Hide Harmonic Decline
        // chart.hideSeries('Hyperbolic Decline'); // Hide Hyperbolic Decline

        
      })
      .catch(error => {
        loadingMessage.style.display = 'none';
        console.error('Error:', error);
        alert('An unexpected error occurred. Please try again.');
      });
  });


  const fetchPredictionWithSelectedData = (selectedData, elr) => {
    showLoading();
    const selectedWell = document.getElementById('wellDropdown').value;

    if (!selectedWell) {
      alert("Please select a well.");
      hideLoading();
      return;
    }

    // if (!selectedData || selectedData.length === 0) {
    //   alert("No selected data available for prediction.");
    //   hideLoading();
    //   return;
    // }

    // Kirim data ke backend
    console.log("well", selectedWell)
    console.log("elr", elr)
    const latestItem = {
      Date: productionData[productionData.length - 1].Date,
      Production: productionData[productionData.length - 1].Production,
    }

    console.log("selectedData", latestItem)
    fetch('http://127.0.0.1:5000/predict_production', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        well: selectedWell,
        economic_limit: elr, // Kirim ELR
        selected_data: selectedPredictObject ?? latestItem // Kirim satu data saja jika dipilih
      })
    })
      .then(response => response.json())
      .then(data => {
        hideLoading();
        console.log(data)
        currentState = 'prediction';

        if (data.error) {
          alert(`Error: ${data.error}`);
          return;
        }

        // Update chart dengan prediksi baru
        const exponentialData = data.ExponentialPrediction.map(point => ({
          x: new Date(point.date),
          y: point.value
        }));
        const harmonicData = data.HarmonicPrediction.map(point => ({
          x: new Date(point.date),
          y: point.value
        }));
        const hyperbolicData = data.HyperbolicPrediction.map(point => ({
          x: new Date(point.date),
          y: point.value
        }));

        
        const newSeries = [
          { name: 'Exponential Decline (Prediction)', type: 'line', data: exponentialData, hidden: false,isPrediction: true },
          { name: 'Harmonic Decline (Prediction)', type: 'line', data: harmonicData , hidden: true, isPrediction: true},
          { name: 'Hyperbolic Decline (Prediction)', type: 'line', data: hyperbolicData, hidden: true, isPrediction: true }
        ]
        const predictionSeries = [
          ...currentDataSeries
        ]

        const finalSeries = appendUniqueSeries(predictionSeries, newSeries);
        currentDataSeries = finalSeries;
        chart.updateSeries(finalSeries);
        updateChartMarkerConfig(finalSeries)
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

    // if (selectedData.length === 0) {
    //   // Jika tidak ada data yang dipilih, ambil dua data terakhir dari grafik
    //   const allData = chart.opts.series[0]?.data || [];
    //   if (allData.length >= 2) {
    //     selectedData = allData.slice(-1); // Ambil satu data terakhir
    //   } else {
    //     alert("Not enough data available for prediction.");
    //     return;
    //   }
    // }

    // latest selectedPredictData
    selectedData=selectedPredictData[selectedPredictData.length - 1];
    
    // selectedData = selectedData?.length ? selectedData[selectedData?.length - 1] : undefined;

    console.log("Selected Data for Prediction:", selectedData);

    // Panggil fungsi fetchPredictionWithSelectedData
    fetchPredictionWithSelectedData(selectedData, elr);
  });



});
