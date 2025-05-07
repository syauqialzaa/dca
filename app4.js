document.addEventListener('DOMContentLoaded', () => {
  // const API_BASE = "https://seagull-literate-rat.ngrok-free.app"
  const API_BASE = "http://127.0.0.1:8000"

  const startDateInput = document.getElementById('startDate');
  const endDateInput   = document.getElementById('endDate');

  const params = new URLSearchParams(window.location.search);

  // get_history parse params
  const initialWell  = params.get('well')       || '';
  const initialStart = params.get('start_date') || '';
  const initialEnd   = params.get('end_date')   || '';

  // automatic_dca parse params
  const selectedDataParam = params.get('selected_data');
  const dateRangeParam = params.get('date_range');
  const productionRangeParam = params.get('production_range');

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

  const oilAxis ={
      seriesName:"Oil",
      title: {
        text: 'Oil (BOPD)',
      },
      labels: {
        formatter: function (value) {
          return value.toFixed(2); // Format nilai hingga 2 desimal
        },
      },

    }

    const fluidAxis = {
    seriesName:"Fluid",
    opposite:true,
    title: {
      text: 'Fluid (BOPD)',
    },
    labels: {
      formatter: function (value) {
        return value.toFixed(2); // Format nilai hingga 2 desimal
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
    yaxis: baseYAxis,
    legend: {
      show: true,

      position: 'top',
      floating: false,
      horizontalAlign: 'center', // Membuat legenda dalam satu baris
      labels: {
        useSeriesColors: false,
      },
      itemMargin: {
        horizontal: 10, // Menambahkan jarak antar legend
        vertical: 5
      },
      markers: {
        width: 12,
        height: 12,
        radius: 12,
      },
    },
    colors: [
      '#2ca02c', // Oil Green
      '#1f77b4', // Blue
      '#ff7f0e', // Orange
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
    stroke: {
      width: 1.5 // Membuat garis lebih tipis
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
    // chart.updateSeries(newSeries)
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
    chart.updateOptions({
      series: newSeries,
      yaxis: getFinalAxisSeries(newSeries)
    })
    // chart.updateSeries(newSeries)
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

  let initialSelectedData = [];
  if (selectedDataParam) {
    try {
      initialSelectedData = JSON.parse(selectedDataParam);
    } catch (err) {
      console.warn('bad selected_data JSON:', err);
    }
  }

  const dateRange = dateRangeParam
    ? dateRangeParam.split(',')
    : [];
  const productionRange = productionRangeParam
    ? productionRangeParam.split(',').map(Number)
    : [];

  // Fetch well data from Flask backend
  fetch(`${API_BASE}/get_wells`)
    .then(res => res.json())
    .then(data => {
      wellDropdown.innerHTML = '<option value="">All Wells</option>';
      data.wells.forEach(w => {
        const o = document.createElement('option');
        o.value = w;
        o.textContent = w;
        if (w === initialWell) o.selected = true;
        wellDropdown.appendChild(o);
      });

      if (initialStart) startDateInput.value = initialStart;
      if (initialEnd)   endDateInput.value   = initialEnd;

      fetchHistory(initialWell, initialStart, initialEnd);
      selectedPredictData = initialSelectedData;

      const hasSelectedDataParam    = params.has('selected_data');
      const hasDateRangeParam       = params.has('date_range');
      const hasProductionRangeParam = params.has('production_range');

      if (hasSelectedDataParam || hasDateRangeParam || hasProductionRangeParam) {
        document.getElementById('automateDCA').click();
      }
    })
    .catch(err => {
      console.error('Error loading wells:', err);
      wellDropdown.innerHTML = '<option value="">Failed to load</option>';
    });

  const fetchHistory = (well = '', startDate = '', endDate = '') => {
    showLoading();
    const params = new URLSearchParams();
    if (well)       params.append('well', well);
    if (startDate)  params.append('start_date', startDate);
    if (endDate)    params.append('end_date', endDate);

    const url = `${API_BASE}/get_history?${params.toString()}`;

    fetch(url, { method: 'GET' })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Server Error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        hideLoading();
        if (!data || data.length === 0) {
          showNoData();
          return;
        }
        hideNoData();

        productionData = data;
        // Transform data for chart
        const productionSeries = {
          name: 'Oil',
          type: 'line',
          data: data.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Production,
          })),
          yAxisIndex: 0, // Gunakan sumbu Y pertama untuk Oil
          showMarker: true,
        };

        const fluidSeries = {
          name: 'Fluid',
          type: 'line', // Use scatter for fluid
          data: data.map((entry) => ({
            x: new Date(entry.Date), // Convert date string to Date object
            y: entry.Fluid,
          })),
          yAxisIndex: 1, // Gunakan sumbu Y kedua untuk Fluid
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
          yAxisIndex: 0, // Gunakan sumbu Y pertama (sama dengan Oil)
          showMarker: true,
        }

        currentDataSeries = [
          productionSeries,
          fluidSeries,
          jobCodeSeries,
        ]

        chart.updateOptions({
          series: currentDataSeries,
          yaxis:baseYAxis
        })
        console.log("Chart Options after update:", chart.opts);
      })
      .catch(error => {
        hideLoading();
        console.error('Error fetching history:', error);
        showNoData();
      });
  };

  // Fetch default data (last 12 months) on page load
  fetchHistory();

  filterButton.addEventListener('click', () => {
    fetchHistory(
      wellDropdown.value,
      startDateInput.value,
      endDateInput.value
    );
  });

  document.getElementById('automateDCA').addEventListener('click', function () {
    const well = document.getElementById('wellDropdown').value;
    if (!well) {
      return alert('Please select a well.');
    }

    const selected_data = selectedPredictData.map(it => ({
      Date:       it.Date,
      Production: it.Production,
      Fluid:      it.Fluid
    }));

    const params = new URLSearchParams();
    params.append('well', well);

    if (selected_data.length) {
      params.append('selected_data', JSON.stringify(selected_data));
    }
    if (dateRange.length === 2) {
      params.append('date_range', dateRange.join(','));
    }
    if (productionRange.length === 2) {
      params.append('production_range', productionRange.join(','));
    }

    document.getElementById('loadingMessage').style.display = 'block';

    fetch(`${API_BASE}/automatic_dca?${params.toString()}`, {
      method: 'GET'
    })
      .then(r => {
        document.getElementById('loadingMessage').style.display = 'none';
        if (!r.ok) throw new Error(`Server ${r.status}`);
        return r.json();
      })
      .then(data => {
        if (data.error) {
          return alert(`Error: ${data.error}`);
        }

        // **✅ Update nilai Decline Rate di Frontend**
        document.getElementById('exp-decline').value = `${data.DeclineRate.Exponential}`;
        document.getElementById('harm-decline').value = `${data.DeclineRate.Harmonic}`;
        document.getElementById('hyper-decline').value = `${data.DeclineRate.Hyperbolic}`;

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
              y: parseFloat(predictionValue).toFixed(2)
            });
            t += 1; // Increment time by 1 day
            currentDate.setDate(currentDate.getDate() + 1); // Move to the next day
          }

          return predictions;
        };

        const exponentialData = generatePrediction(exponentialModel, data.Exponential, startDate, endDate);
        const harmonicData = generatePrediction(harmonicModel, data.Harmonic, startDate, endDate);
        const hyperbolicData = generatePrediction(hyperbolicModel, data.Hyperbolic, startDate, endDate);

        console.log("range exp : ", exponentialData);
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
          type: 'line', // Use scatter for fluid
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
          yAxisIndex: 0,
          showMarker: true,
        }

        // update current data series, set job code and fluid series hidden
        currentDataSeries = currentDataSeries.map((it) => {
          if(it.name === "Job Code"){
            return {
              ...it,
              hidden: true
            }
          }
          return it;
        })
        // console.log("current data serr: ", currentDataSeries);

        // remove prediction series
        currentDataSeries = currentDataSeries.filter((it) => !it.isPrediction);
        const newSeries = [
          { name: 'Exponential Decline', type: 'line', data: exponentialData,  yaxis: 0, hidden: false},
          { name: 'Harmonic Decline', type: 'line', data: harmonicData, yaxis: 0, hidden:true},
          { name: 'Hyperbolic Decline', type: 'line', data: hyperbolicData, yaxis: 0, hidden:true }
        ]
        // console.log("current data serr 2: ", currentDataSeries);
        const finalSeries = appendUniqueSeries(currentDataSeries, newSeries);
        // console.log("final : ", finalSeries)
        currentDataSeries = finalSeries;

        chart.updateOptions({
          series: finalSeries,
          yaxis: getFinalAxisSeries(finalSeries)
        })
        // chart.updateSeries(finalSeries);
        updateChartMarkerConfig(finalSeries)

      })
      .catch(err => {
        document.getElementById('loadingMessage').style.display = 'none';
        console.error(err);
        alert('An unexpected error occurred.');
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

    console.log("well", selectedWell)
    console.log("elr", elr)
    const latestItem = {
      Date: productionData[productionData.length - 1].Date,
      Production: productionData[productionData.length - 1].Production,
    }

    console.log("selectedData", latestItem)
    fetch(`${API_BASE}/predict_production`, {
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
          { name: 'Exponential Decline (Prediction)', type: 'line', data: exponentialData, yAxisIndex: 0, hidden: false,isPrediction: true },
          { name: 'Harmonic Decline (Prediction)', type: 'line', data: harmonicData, yAxisIndex: 0, hidden: true, isPrediction: true},
          { name: 'Hyperbolic Decline (Prediction)', type: 'line', data: hyperbolicData, yAxisIndex: 0, hidden: true, isPrediction: true }
        ]
        const predictionSeries = [
          ...currentDataSeries
        ]

        const finalSeries = appendUniqueSeries(predictionSeries, newSeries);
        currentDataSeries = finalSeries;
        chart.updateOptions({
          series: finalSeries,
          yaxis: getFinalAxisSeries(finalSeries)
        })
        // chart.updateSeries(finalSeries);
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
    // latest selectedPredictData
    selectedData=selectedPredictData[selectedPredictData.length - 1];

    // selectedData = selectedData?.length ? selectedData[selectedData?.length - 1] : undefined;

    console.log("Selected Data for Prediction:", selectedData);

    // Panggil fungsi fetchPredictionWithSelectedData
    fetchPredictionWithSelectedData(selectedData, elr);
  });

  //Model ML DCA
  document.getElementById('mlDCA').addEventListener('click', () => {
    const selectedWell = document.getElementById('wellDropdown').value;
    const elr = document.getElementById('elr').value || 10; // Default ELR = 10 jika kosong

    if (!selectedWell) {
      alert("Please select a well to predict.");
      return;
    }

    showLoading(); // Tampilkan loading spinner

    fetch(`${API_BASE}/predict_ml`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        well: selectedWell,
        elr: parseFloat(elr) // Pastikan ELR dikirim sebagai angka
      })
    })
      .then(response => response.json())
      .then(data => {
        hideLoading(); // Sembunyikan loading spinner

        if (data.error) {
          alert(`Error: ${data.error}`);
          return;
        }

        // Data historis
        const actualSeries = {
          name: 'Actual Production',
          type: 'line',
          data: data.dates_actual.map((date, index) => ({
            x: new Date(date),
            y: data.actual[index]
          })),
          showMarker: true
        };

        // Prediksi historis (validasi model)
        const predictedSeries = {
          name: 'Predicted Production (Historical)',
          type: 'line',
          data: data.dates_actual.map((date, index) => ({
            x: new Date(date),
            y: data.predicted[index]
          })),
          showMarker: false,
          hidden: true // Opsional: sembunyikan secara default
        };

        // Prediksi masa depan hingga ELR
        const extendedSeries = {
          name: 'ML Prediction (Future)',
          type: 'line',
          data: data.dates_extended.map((date, index) => ({
            x: new Date(date),
            y: data.extended_prediction[index]
          })),
          showMarker: false
        };

        // Garis ELR sebagai anotasi
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

        // Filter series yang bukan prediksi ML sebelumnya
        const filteredSeries = currentDataSeries.filter(series => !series.name.includes('ML Prediction'));
        const newSeries = [...filteredSeries, actualSeries, predictedSeries, extendedSeries];
        currentDataSeries = newSeries;

        // Perbarui grafik
        chart.updateOptions({
          series: newSeries,
          yaxis: getFinalAxisSeries(newSeries),
          annotations: elrAnnotation
        });
        updateChartMarkerConfig(newSeries);
      })
      .catch(error => {
        hideLoading();
        console.error('Error:', error);
        alert('An unexpected error occurred. Please try again.');
      });
  });

  document.getElementById('resetChart').addEventListener('click', () => {
    // Reload halaman untuk mengembalikan semua state ke kondisi awal
    window.location.reload();
  });


});
