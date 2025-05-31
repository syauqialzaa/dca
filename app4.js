document.addEventListener('DOMContentLoaded', () => {
  const wellDropdown = document.getElementById('wellDropdown');
  const filterButton = document.getElementById('filterButton');
  const chartElement = document.getElementById('chart');
  const loadingElement = document.getElementById('loading');
  const noDataElement = document.getElementById('noData');
  let productionData = [];
  // let automaticDataSeries = [];
  let selectedPredictData = [];
  let currentState = '';
  let selectedPredictObject = undefined;
  let currentDataSeries = [];

  // ========== NGROK CONFIGURATION ==========
  const NGROK_BASE_URL = 'https://5c959a7dff3c.ngrok.app';

  // Alternative: You can also set this dynamically
  // const NGROK_BASE_URL = prompt("Enter your ngrok URL (e.g., https://abc123.ngrok-free.app):");

  const NGROK_HEADERS = {
    'ngrok-skip-browser-warning': 'true',
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  };

  // API Configuration
  const API_CONFIG = {
    baseURL: NGROK_BASE_URL,
    headers: NGROK_HEADERS,
    timeout: 30000 // 30 seconds timeout
  };

  // Enhanced fetch function with ngrok support
  const fetchWithNgrok = async (endpoint, options = {}) => {
    const url = `${API_CONFIG.baseURL}${endpoint}`;

    const fetchOptions = {
      ...options,
      headers: {
        ...API_CONFIG.headers,
        ...options.headers
      },
      timeout: API_CONFIG.timeout
    };

    console.log(`Making request to: ${url}`);
    console.log('Request options:', fetchOptions);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);

      const response = await fetch(url, {
        ...fetchOptions,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Response received:', data);
      return data;
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);

      if (error.name === 'AbortError') {
        throw new Error('Request timeout - ngrok tunnel might be slow');
      }

      if (error.message.includes('Failed to fetch')) {
        throw new Error('Network error - check if ngrok tunnel is active and URL is correct');
      }

      throw error;
    }
  };

  // URL Parameters Management
  const URLParams = {
    get: () => {
      const params = new URLSearchParams(window.location.search);
      return {
        well: params.get('well') || '',
        start_date: params.get('start_date') || '',
        end_date: params.get('end_date') || '',
        elr: params.get('elr') || '',
        view: params.get('view') || 'history',
        selected_data: params.get('selected_data') ? JSON.parse(decodeURIComponent(params.get('selected_data'))) : null
      };
    },

    set: (newParams) => {
      const url = new URL(window.location);
      const params = url.searchParams;

      Object.keys(newParams).forEach(key => {
        if (newParams[key] !== null && newParams[key] !== undefined && newParams[key] !== '') {
          if (key === 'selected_data' && typeof newParams[key] === 'object') {
            params.set(key, encodeURIComponent(JSON.stringify(newParams[key])));
          } else {
            params.set(key, newParams[key]);
          }
        } else {
          params.delete(key);
        }
      });

      window.history.pushState({}, '', url);
    },

    clear: () => {
      window.history.pushState({}, '', window.location.pathname);
    }
  };

  // Load initial state from URL
  const loadFromURL = () => {
    const params = URLParams.get();

    if (params.well) {
      wellDropdown.value = params.well;
    }
    if (params.start_date) {
      const startDateElement = document.getElementById('startDate');
      if (startDateElement) startDateElement.value = params.start_date;
    }
    if (params.end_date) {
      const endDateElement = document.getElementById('endDate');
      if (endDateElement) endDateElement.value = params.end_date;
    }
    if (params.elr) {
      const elrElement = document.getElementById('elr');
      if (elrElement) elrElement.value = params.elr;
    }

    if (params.well || params.start_date || params.end_date) {
      fetchHistory(params.well, params.start_date, params.end_date);
    }

    setTimeout(() => {
      switch (params.view) {
        case 'dca':
          if (params.well) {
            executeAutomaticDCA(params.well, params.selected_data);
          }
          break;
        case 'prediction':
          if (params.well) {
            executePrediction(params.well, params.elr || 5, params.selected_data);
          }
          break;
        case 'ml':
          if (params.well) {
            executeMLPrediction(params.well, params.elr || 10);
          }
          break;
      }
    }, 1000);
  };

  const oilAxis = {
    seriesName: "Oil",
    title: {
      text: 'Oil (BOPD)',
    },
    labels: {
      formatter: function (value) {
        return value.toFixed(2);
      },
    },
  }

  const fluidAxis = {
    seriesName: "Fluid",
    opposite: true,
    title: {
      text: 'Fluid (BOPD)',
    },
    labels: {
      formatter: function (value) {
        return value.toFixed(2);
      },
    },
  }

  const baseYAxis = [
    oilAxis, fluidAxis, fluidAxis
  ]

  // Initialize chart with updated configuration
  let chart = new ApexCharts(chartElement, {
    chart: {
      type: 'line',
      height: 700,
      zoom: {},
      events: {
        markerClick: function (event, chartContext, opts) {
          console.log("Marker Click", opts)
          const selectedIndex = opts.dataPointIndex;
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
    series: [],
    xaxis: {
      type: 'datetime',
      labels: {
        format: 'MMM yyyy',
      },
      title: {
        text: 'Date',
      },
    },
    yaxis: baseYAxis,
    legend: {
      show: true,
      position: 'top',
      floating: false,
      horizontalAlign: 'center',
      labels: {
        useSeriesColors: false,
      },
      itemMargin: {
        horizontal: 10,
        vertical: 5
      },
      markers: {
        width: 12,
        height: 12,
        radius: 12,
      },
    },
    colors: [
      '#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd',
      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
      '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ],
    toolbar: {
      tools: {
        selection: true,
        zoom: true
      }
    },
    events: {
      selection: function (chartContext, { xaxis }) {
        const startDate = new Date(xaxis.min).toISOString().split('T')[0];
        const endDate = new Date(xaxis.max).toISOString().split('T')[0];
        const selectedData = chartContext.opts.series[0].data.filter(point => {
          const date = new Date(point.x);
          return date >= new Date(startDate) && date <= new Date(endDate);
        });

        window.selectedData = selectedData;

        URLParams.set({
          selected_data: selectedData.map(point => ({
            Date: new Date(point.x).toISOString().split('T')[0],
            Production: point.y
          }))
        });

        if (selectedData.length === 0) {
          alert("No data selected in the given range.");
        } else {
          alert(`Selected ${selectedData.length} data points.`);
        }

        fetchPredictionWithSelectedData(selectedData, startDate, endDate);
      }
    },
    stroke: {
      width: 1.5
    },
    markers: {
      size: [5,5, 5,5,0,0,0,0,0,0],
      strokeWidth: 2,
    },
    tooltip: {
      shared: true,
      intersect: false,
      custom: function ({series, seriesIndex, dataPointIndex, w}) {
        const pointData = w.config.series[seriesIndex].data[dataPointIndex];
        const seriesName = w.config.series[seriesIndex].name;
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
      xaxis: [],
    },
  });

  chart.render();

  function updateSelectedPredictData(data) {
    console.log("Data selected:", data);
    if(currentState !== 'prediction'){
      const currentDataPointIndexes = selectedPredictData.map((it) => it.dataPointIndex);
      if(currentDataPointIndexes.includes(data.dataPointIndex)){
        selectedPredictData = selectedPredictData.filter((it) => it.dataPointIndex !== data.dataPointIndex);
      }else {
        selectedPredictData.push(data);
        selectedPredictData.sort((a, b) => a.dataPointIndex - b.dataPointIndex);
      }
      updateSelectedPredictView();

      URLParams.set({
        selected_data: selectedPredictData.map(item => ({
          Date: item.Date,
          Production: item.Production,
          Fluid: item.Fluid
        }))
      });
    }else {
      selectedPredictObject = data
      updateSelectedPredictObjectView();

      URLParams.set({
        selected_data: {
          Date: data.Date,
          Production: data.Production
        }
      });
    }
  }

  function getFinalAxisSeries(series){
    const baseAxisLength = baseYAxis.length
    const newSeriesLength = series.length
    const newAxis = [...baseYAxis]
    const diffLength = newSeriesLength - baseAxisLength
    for (let i = 0; i < diffLength; i++) {
      newAxis.push(oilAxis)
    }
    return newAxis;
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
        type: "scatter",
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
    chart.updateOptions({
      series: newSeries,
      yaxis: getFinalAxisSeries(newSeries)
    })
    updateChartMarkerConfig(newSeries)
    const selectedPredictDataElement = document.getElementById("selectedPredictData");
    if (selectedPredictDataElement) {
      selectedPredictDataElement.innerText = dateString;
    }
  }

  function updateSelectedPredictObjectView(){
    const dateString = selectedPredictObject?.Date;
    const selectedPredictObjectElement = document.getElementById("selectedPredictObject");
    if (selectedPredictObjectElement) {
      selectedPredictObjectElement.innerText = dateString;
    }

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
    chart.updateOptions({
      series: newSeries,
      yaxis: getFinalAxisSeries(newSeries)
    })
    updateChartMarkerConfig(newSeries)
  }

  const showLoading = () => {
    loadingElement.style.display = 'block';
    noDataElement.style.display = 'none';
    chartElement.style.display = 'none';
  };

  const hideLoading = () => {
    loadingElement.style.display = 'none';
    chartElement.style.display = 'block';
  };

  const showNoData = () => {
    noDataElement.style.display = 'block';
    chartElement.style.display = 'none';
  };

  const hideNoData = () => {
    noDataElement.style.display = 'none';
    chartElement.style.display = 'block';
  };

  // Fetch well data using ngrok
  fetchWithNgrok('/get_wells')
    .then(data => {
      wellDropdown.innerHTML = '<option value="">Select...</option>';
      if (data.wells) {
        data.wells.forEach(well => {
          const option = document.createElement("option");
          option.value = well;
          option.textContent = well;
          wellDropdown.appendChild(option);
        });
      }
      loadFromURL();
    })
    .catch(error => {
      console.error("Error loading wells:", error);
      wellDropdown.innerHTML = '<option value="">Failed to load</option>';
      alert(`Failed to load wells: ${error.message}\n\nPlease check:\n1. ngrok tunnel is running\n2. ngrok URL is correct in the code\n3. Flask server is running`);
    });

  const fetchHistory = async (well, startDate, endDate) => {
    showLoading();

    URLParams.set({
      well: well || '',
      start_date: startDate || '',
      end_date: endDate || '',
      view: 'history'
    });

    try {
      const data = await fetchWithNgrok('/get_history', {
        method: 'POST',
        body: JSON.stringify({
          well,
          start_date: startDate,
          end_date: endDate
        })
      });

      console.log("Data received from backend:", data);
      hideLoading();

      if (!data || data.length === 0) {
        console.warn("No data available for the given filters.");
        productionData = [];
        showNoData();
        return;
      }

      hideNoData();
      productionData = data;

      const productionSeries = {
        name: 'Oil',
        type: 'line',
        data: data.map((entry) => ({
          x: new Date(entry.Date),
          y: entry.Production,
        })),
        yAxisIndex: 0,
        showMarker: true,
      };

      const fluidSeries = {
        name: 'Fluid',
        type: 'line',
        data: data.map((entry) => ({
          x: new Date(entry.Date),
          y: entry.Fluid,
        })),
        yAxisIndex: 1,
        showMarker: true,
      };

      const maxFluid = Math.max(...data.map(entry => entry.Fluid)) + 100;
      const jobCodeSeries = {
        name: "Job Code",
        type: "scatter",
        group: "default",
        data: data.filter((it) => !!it?.JobCode).map((entry) => ({
          x: entry.Date,
          y: maxFluid,
          name: entry.JobCode
        })),
        yAxisIndex: 0,
        showMarker: true,
      }

      currentDataSeries = [
        productionSeries,
        fluidSeries,
        jobCodeSeries,
      ]

      chart.updateOptions({
        series: currentDataSeries,
        yaxis: baseYAxis
      })

      console.log("Chart Options after update:", chart.opts);
    } catch (error) {
      hideLoading();
      console.error('Error fetching history:', error);
      showNoData();
      alert(`Failed to fetch history: ${error.message}`);
    }
  };

  const executeAutomaticDCA = async (selectedWell, selected_data) => {
    if (!selectedWell) {
      alert("Please select a well.");
      return;
    }

    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) loadingMessage.style.display = 'block';

    const requestData = {
      well: selectedWell,
      selected_data: selected_data || selectedPredictData?.map((it) => ({
        Date: it.Date,
        Production: it.Production,
        Fluid: it.Fluid
      }))
    };

    try {
      const data = await fetchWithNgrok('/automatic_dca', {
        method: 'POST',
        body: JSON.stringify(requestData)
      });

      if (loadingMessage) loadingMessage.style.display = 'none';

      if (data.error) {
        alert(`Error: ${data.error}`);
        return;
      }

      URLParams.set({
        view: 'dca',
        well: selectedWell,
        selected_data: selected_data
      });

      const expDeclineElement = document.getElementById('exp-decline');
      const harmDeclineElement = document.getElementById('harm-decline');
      const hyperDeclineElement = document.getElementById('hyper-decline');

      if (expDeclineElement) expDeclineElement.value = `${data.DeclineRate.Exponential}`;
      if (harmDeclineElement) harmDeclineElement.value = `${data.DeclineRate.Harmonic}`;
      if (hyperDeclineElement) hyperDeclineElement.value = `${data.DeclineRate.Hyperbolic}`;

      const actualData = data.ActualData.map(point => ({
        x: new Date(point.date),
        y: point.value
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
            y: parseFloat(predictionValue.toFixed(2))
          });
          t += 1;
          currentDate.setDate(currentDate.getDate() + 1);
        }

        return predictions;
      };

      const exponentialData = generatePrediction(exponentialModel, data.Exponential, startDate, endDate);
      const harmonicData = generatePrediction(harmonicModel, data.Harmonic, startDate, endDate);
      const hyperbolicData = generatePrediction(hyperbolicModel, data.Hyperbolic, startDate, endDate);

      const actualDataSeries = {
        name: 'Actual Data',
        type: 'line',
        data: actualData,
        yAxisIndex: 0,
        showMarker: true,
        stroke: {
          color: '#000000',
          width: 4
        }
      };

      currentDataSeries = currentDataSeries.map((it) => {
        if(it.name === "Job Code"){
          return { ...it, hidden: true }
        }
        return it;
      })

      currentDataSeries = currentDataSeries.filter((it) => !it.isPrediction);
      const newSeries = [
        actualDataSeries,
        { name: 'Exponential Decline', type: 'line', data: exponentialData, yaxis: 0, hidden: false},
        { name: 'Harmonic Decline', type: 'line', data: harmonicData, yaxis: 0, hidden: true},
        { name: 'Hyperbolic Decline', type: 'line', data: hyperbolicData, yaxis: 0, hidden: true }
      ];

      const finalSeries = appendUniqueSeries(currentDataSeries, newSeries);
      currentDataSeries = finalSeries;

      chart.updateOptions({
        series: finalSeries,
        yaxis: getFinalAxisSeries(finalSeries)
      });
      updateChartMarkerConfig(finalSeries);

    } catch (error) {
      if (loadingMessage) loadingMessage.style.display = 'none';
      console.error('Error:', error);
      alert(`An error occurred during DCA analysis: ${error.message}`);
    }
  };

  const executePrediction = async (selectedWell, elr, selected_data) => {
    showLoading();

    if (!selectedWell) {
      alert("Please select a well.");
      hideLoading();
      return;
    }

    const latestItem = selected_data || {
      Date: productionData[productionData.length - 1]?.Date,
      Production: productionData[productionData.length - 1]?.Production,
    };

    URLParams.set({
      view: 'prediction',
      well: selectedWell,
      elr: elr,
      selected_data: latestItem
    });

    try {
      const data = await fetchWithNgrok('/predict_production', {
        method: 'POST',
        body: JSON.stringify({
          well: selectedWell,
          economic_limit: elr,
          selected_data: selectedPredictObject || latestItem
        })
      });

      hideLoading();
      currentState = 'prediction';

      if (data.error) {
        alert(`Error: ${data.error}`);
        return;
      }

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
        { name: 'Exponential Decline (Prediction)', type: 'line', data: exponentialData, yAxisIndex: 0, hidden: false, isPrediction: true },
        { name: 'Harmonic Decline (Prediction)', type: 'line', data: harmonicData, yAxisIndex: 0, hidden: true, isPrediction: true},
        { name: 'Hyperbolic Decline (Prediction)', type: 'line', data: hyperbolicData, yAxisIndex: 0, hidden: true, isPrediction: true }
      ];

      const finalSeries = appendUniqueSeries(currentDataSeries, newSeries);
      currentDataSeries = finalSeries;
      chart.updateOptions({
        series: finalSeries,
        yaxis: getFinalAxisSeries(finalSeries)
      });
      updateChartMarkerConfig(finalSeries);
    } catch (error) {
      hideLoading();
      console.error('Error:', error);
      alert(`An error occurred during prediction: ${error.message}`);
    }
  };

  const executeMLPrediction = async (selectedWell, elr) => {
    if (!selectedWell) {
      alert("Please select a well to predict.");
      return;
    }

    showLoading();

    URLParams.set({
      view: 'ml',
      well: selectedWell,
      elr: elr
    });

    try {
      const data = await fetchWithNgrok('/predict_ml', {
        method: 'POST',
        body: JSON.stringify({
          well: selectedWell,
          elr: parseFloat(elr)
        })
      });

      hideLoading();

      if (data.error) {
        alert(`Error: ${data.error}`);
        return;
      }

      const actualSeries = {
        name: 'Actual Production',
        type: 'line',
        data: data.dates_actual.map((date, index) => ({
          x: new Date(date),
          y: data.actual[index]
        })),
        showMarker: true
      };

      const predictedSeries = {
        name: 'Predicted Production (Historical)',
        type: 'line',
        data: data.dates_actual.map((date, index) => ({
          x: new Date(date),
          y: data.predicted[index]
        })),
        showMarker: false,
        hidden: true
      };

      const extendedSeries = {
        name: 'ML Prediction (Future)',
        type: 'line',
        data: data.dates_extended.map((date, index) => ({
          x: new Date(date),
          y: data.extended_prediction[index]
        })),
        showMarker: false
      };

      const elrAnnotation = {
        yaxis: [{
          y: data.elr_threshold,
          borderColor: '#ff0000',
          label: {
            text: `ELR: ${data.elr_threshold}`,
            style: {
              color: '#fff',
              background: '#ff0000'
            }
          }
        }]
      };

      const filteredSeries = currentDataSeries.filter(series => !series.name.includes('ML Prediction'));
      const newSeries = [...filteredSeries, actualSeries, predictedSeries, extendedSeries];
      currentDataSeries = newSeries;

      chart.updateOptions({
        series: newSeries,
        yaxis: getFinalAxisSeries(newSeries),
        annotations: elrAnnotation
      });
      updateChartMarkerConfig(newSeries);
    } catch (error) {
      hideLoading();
      console.error('Error:', error);
      alert(`An error occurred during ML prediction: ${error.message}`);
    }
  };

  const fetchPredictionWithSelectedData = (selectedData, elr) => {
    const selectedWell = document.getElementById('wellDropdown').value;
    const latestItem = {
      Date: productionData[productionData.length - 1]?.Date,
      Production: productionData[productionData.length - 1]?.Production,
    }

    executePrediction(selectedWell, elr || 5, selectedPredictObject || latestItem);
  };

  // Load default data only if no URL params
  const urlParams = URLParams.get();
  if (!urlParams.well && !urlParams.start_date && !urlParams.end_date) {
    fetchHistory();
  }

  // Event listeners
  filterButton.addEventListener('click', () => {
    const selectedWell = wellDropdown.value;
    const startDateElement = document.getElementById('startDate');
    const endDateElement = document.getElementById('endDate');
    const startDate = startDateElement ? startDateElement.value : '';
    const endDate = endDateElement ? endDateElement.value : '';
    fetchHistory(selectedWell, startDate, endDate);
  });

  document.getElementById('automateDCA').addEventListener('click', function () {
    const selectedWell = document.getElementById('wellDropdown').value;
    currentState = '';
    selectedPredictObject = undefined;
    executeAutomaticDCA(selectedWell);
  });

  const predictDCAButton = document.getElementById('predictDCA');
  if (predictDCAButton) {
    predictDCAButton.addEventListener('click', () => {
      const selectedWell = document.getElementById('wellDropdown').value;
      const elrElement = document.getElementById('elr');
      const elr = elrElement ? elrElement.value || 5 : 5;
      let selectedData = selectedPredictData[selectedPredictData.length - 1];
      executePrediction(selectedWell, elr, selectedData);
    });
  }

  const mlDCAButton = document.getElementById('mlDCA');
  if (mlDCAButton) {
    mlDCAButton.addEventListener('click', () => {
      const selectedWell = document.getElementById('wellDropdown').value;
      const elrElement = document.getElementById('elr');
      const elr = elrElement ? elrElement.value || 10 : 10;
      executeMLPrediction(selectedWell, elr);
    });
  }

  const resetChartButton = document.getElementById('resetChart');
  if (resetChartButton) {
    resetChartButton.addEventListener('click', () => {
      URLParams.clear();
      window.location.reload();
    });
  }

  // Listen for browser back/forward button
  window.addEventListener('popstate', () => {
    loadFromURL();
  });

  // Update URL when form fields change
  wellDropdown.addEventListener('change', () => {
    const params = URLParams.get();
    URLParams.set({ ...params, well: wellDropdown.value });
  });

  const startDateElement = document.getElementById('startDate');
  if (startDateElement) {
    startDateElement.addEventListener('change', (e) => {
      const params = URLParams.get();
      URLParams.set({ ...params, start_date: e.target.value });
    });
  }

  const endDateElement = document.getElementById('endDate');
  if (endDateElement) {
    endDateElement.addEventListener('change', (e) => {
      const params = URLParams.get();
      URLParams.set({ ...params, end_date: e.target.value });
    });
  }

  const elrElement = document.getElementById('elr');
  if (elrElement) {
    elrElement.addEventListener('change', (e) => {
      const params = URLParams.get();
      URLParams.set({ ...params, elr: e.target.value });
    });
  }

});
