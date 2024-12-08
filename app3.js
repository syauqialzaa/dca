document.addEventListener('DOMContentLoaded', () => {
  const wellDropdown = document.getElementById('wellDropdown');
  const filterButton = document.getElementById('filterButton');
  const chartElement = document.getElementById('chart');
  const loadingElement = document.getElementById('loading');
  const noDataElement = document.getElementById('noData');
  let displayedData = [];

  let selectedStartPoint;
  let selectedEndPoint;
  const chartSelectedColors = [
    "#008ffb",
    "#775dd0",
    "#ffb01a",
    "#ff4560",
    "#00e396",
  ];

  const chartYAxis = [
    {
      title: {
        text: 'Oil Production',
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
  ]

  const chartYAxisSelected = [
    {
      title: {
        text: 'Oil Production',
      },
      labels: {
        formatter: function (value) {
          return value.toFixed(2); // Format nilai hingga 2 desimal
        },
      },
    },
  ]

  // Initialize chart with updated configuration
  let chart = new ApexCharts(chartElement, {
    chart: {
      type: 'line',
      height: 400,
      zoom: {
        enabled: true, // Enable zooming for better exploration
      },
      events: {
        // dataPointSelection: function (event, chartContext, opts) {
        //   const selectedIndex = opts.dataPointIndex;
        //
        // },
        markerClick: function (event, chartContext, opts) {
          console.log("Marker Click", opts)
          const selectedIndex = opts.dataPointIndex;
          const item = displayedData[selectedIndex];

          updatePointsValue({
            ...item,
            dataPointIndex: selectedIndex
          })

        }
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
    yaxis: chartYAxis,
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

  function updatePointsValue(point) {
    if (!selectedStartPoint) {
      selectedStartPoint = point;
    } else {
      if (!selectedEndPoint) {
        selectedEndPoint = point
      } else {
        selectedStartPoint = point
        selectedEndPoint = undefined
      }
    }

    // Adapt values to ensure start is the lowest and end is the highest
    let points = [selectedStartPoint, selectedEndPoint].filter(p => p !== undefined);
    points.sort((a, b) => a.dataPointIndex - b.dataPointIndex); // Sort points in ascending order

    [selectedStartPoint, selectedEndPoint] = points;
    updatePointsView()
    loadPointsData()
  }

  function updatePointsView() {
    document.getElementById("startSelectedChartPoint").innerText = selectedStartPoint?.Date ?? "-";
    document.getElementById("endSelectedChartPoint").innerText = selectedEndPoint?.Date ?? "-";
  }

  function resetSelectedPointData() {
    selectedStartPoint = undefined
    selectedEndPoint = undefined
    updatePointsView()
    chart.updateOptions({
      annotations: {xaxis: []},
    });

  }

  function loadPointsData() {
    if (selectedStartPoint && selectedEndPoint) {
      console.log("wellDropdown", wellDropdown.value)
      if (!wellDropdown.value) {
        alert("Select Well First")
        return
      }
      // Show loading spinner
      // showLoading();

      // Fetch DCA analysis
      fetch('http://127.0.0.1:5000/calculate_dca3', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          well: wellDropdown.value,
          start_date: selectedStartPoint.Date,
          end_date: selectedEndPoint.Date
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          // hideLoading();

          if (data.error) {
            console.error(data.error);
            alert('Error calculating DCA');
            return;
          }

          // Update start, mid, and end point information
          document.getElementById('startPoint').innerText = data.start_date;
          document.getElementById('midPoint').innerText = data.mid_date;
          document.getElementById('endPoint').innerText = data.end_date;

          // trim displayed data based on selected start and end point
          const selectedData = [...displayedData].filter(
            (d) => d.Date >= data.start_date && d.Date <= data.end_date
          );


          // Update chart with DCA analysis
          chart.updateSeries([
            {
              name: 'Historical Data',
              type: 'line',
              data: displayedData.map((entry) => ({
                x: entry.Date,
                y: entry.Production,
              })),
            },
            {
              name: 'Selected Data',
              type: 'line',
              data: selectedData.map((entry) => ({
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
                style: {color: '#fff', background: '#00E396'},
              },
            },
            {
              x: new Date(data.mid_date).getTime(),
              borderColor: '#FEB019',
              label: {
                text: 'Mid Point',
                style: {color: '#fff', background: '#FEB019'},
              },
            },
            {
              x: new Date(data.end_date).getTime(),
              borderColor: '#FF4560',
              label: {
                text: 'End Point',
                style: {color: '#fff', background: '#FF4560'},
              },
            },
          ];
          chart.updateOptions({
            annotations: {xaxis: annotations},
            colors: chartSelectedColors,
            yAxis: chartYAxisSelected
          });
        })
        .catch((error) => {
          hideLoading();
          console.error('Error fetching DCA:', error);
          alert('An unexpected error occurred. Please try again.');
        });
    }
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

  // Fetch and display historical data
  const fetchHistory = (well, startDate, endDate) => {
    showLoading(); // Show loading spinner
    fetch('http://127.0.0.1:5000/get_history', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({well, start_date: startDate, end_date: endDate}),
    })
      .then((response) => response.json())
      .then((data) => {
        hideLoading(); // Hide loading spinner

        if (data.length === 0) {
          displayedData = [];
          showNoData(); // Show "No Data Found" if no data
        } else {
          hideNoData(); // Hide "No Data Found" message
          displayedData = data;
          // Series untuk Production (line) pada y-axis pertama (index 0)
          const productionSeries = {
            name: 'Oil Production',
            group: "default",
            type: 'line',
            data: data.map((entry) => ({
              x: entry.Date,
              y: entry.Production,
            })),
            yAxisIndex: 0
          };

          // Series untuk Fluid (scatter) pada y-axis kedua (index 1)
          const fluidSeries = {
            name: 'Fluid',
            type: 'scatter',
            group: "Fluid",
            color: '#13de9f',
            data: data.map((entry) => ({
              x: entry.Date,
              y: entry.Fluid,
            })),
            yAxisIndex: 1
          };

          const jobCodeSeries = {
            name: "Job Code",
            type: "scatter",
            group: "default",
            color: '#ffb01a',
            data: data.filter((it) => !!it?.JobCode).map((entry) => ({
              x: entry.Date,
              y: 0,
              name: entry.JobCode
            }))
          }

          resetSelectedPointData()
          chart.updateSeries([productionSeries, fluidSeries, jobCodeSeries]);
        }
      })
      .catch((error) => {
        hideLoading(); // Hide loading spinner even on error
        console.error('Error fetching history:', error);
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
      body: JSON.stringify({well: selectedWell, start_date: startDate, end_date: endDate}),
    })
      .then((response) => response.json())
      .then((data) => {
        hideLoading();

        if (data.error) {
          console.error(data.error);
          alert('Error calculating DCA');
          displayedData = [];
          return;
        }

        // Update start, mid, and end point information
        document.getElementById('startPoint').innerText = data.start_date;
        document.getElementById('midPoint').innerText = data.mid_date;
        document.getElementById('endPoint').innerText = data.end_date;

        displayedData = data.historical;
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
              style: {color: '#fff', background: '#00E396'},
            },
          },
          {
            x: new Date(data.mid_date).getTime(),
            borderColor: '#FEB019',
            label: {
              text: 'Mid Point',
              style: {color: '#fff', background: '#FEB019'},
            },
          },
          {
            x: new Date(data.end_date).getTime(),
            borderColor: '#FF4560',
            label: {
              text: 'End Point',
              style: {color: '#fff', background: '#FF4560'},
            },
          },
        ];
        chart.updateOptions({
          annotations: {xaxis: annotations},
          // colors: chartSelectedColors,
          // yAxis: chartYAxisSelected
        });
      })
      .catch((error) => {
        hideLoading();
        console.error('Error fetching DCA:', error);
        alert('An unexpected error occurred. Please try again.');
      });
  });
});
