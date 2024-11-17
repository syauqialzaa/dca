let chart;
let selectedData = [];
let startDateParam;
let endDateParam;

let startSelection;
let endSelection;
let formattedData = [];

function getFormattedData(startDate, endDate, historicalData) {
  let filteredData
  if (!startDate || !endDate) {
    const today = new Date();
    const oneYearAgo = new Date(today);
    oneYearAgo.setFullYear(today.getFullYear() - 1);

    filteredData = historicalData.filter(item => {
      const date = new Date(item.x);
      return date >= oneYearAgo && date <= today;
    });
  } else {
    const startDateFilter = new Date(startDate);
    const endDateFilter = new Date(endDate);

    // Filter data based on query parameters
    filteredData = historicalData.filter(item => {
      const date = new Date(item.x);
      return date >= startDateFilter && date <= endDateFilter;
    });

  }

  return filteredData.map(item => ({
    x: new Date(item.x).getTime(),
    y: item.y || 0
  }));
}

// Initialize Chart
async function loadChart() {
  try {
    const response = await fetch('http://127.0.0.1:5000/get_data');
    const historicalData = await response.json();
    // Get query parameters for start and end dates
    let urlParams = new URLSearchParams(window.location.search);
    startDateParam = urlParams.get('startDate');
    endDateParam = urlParams.get('endDate');
    if (startDateParam && endDateParam) {
      const startDateInput = document.getElementById('start-date');
      const endDateInput = document.getElementById('end-date');

      startDateInput.value = startDateParam;
      endDateInput.value = endDateParam;
    }

    formattedData = getFormattedData(startDateParam, endDateParam, historicalData);

    // Initialize the chart
    chart = new ApexCharts(document.querySelector("#chart"), {
      chart: {
        type: 'line',
        height: 350,
        zoom: {
          enabled: false // Disable zoom to focus on selection
        },
        toolbar: {
          tools: {
            selection: true, // Ensure selection tool is enabled
            zoom: false, // Disable zoom tool
            pan: true, // Allow panning
            reset: true // Allow resetting chart
          }
        },
        selection: {
          enabled: false, // Ensure selection is globally enabled
          type: 'x', // Restrict selection to x-axis
        },
        events: {
          markerClick: function (event, chartContext, opts) {
            // console.log("Marker Click", opts)
            const selectedIndex = opts.dataPointIndex;
            formattedData = getFormattedData(startDateParam, endDateParam, historicalData);
            const item = formattedData[selectedIndex];
            // set end of selection
            if (startSelection && endSelection) {
              startSelection = item
              endSelection = undefined
            } else if (startSelection) {
              if (
                startSelection['x'] < item['x']
              ) {
                endSelection = item
              } else {
                startSelection = item
              }
            } else {
              startSelection = item
            }
            loadSelectionChart()
          },
          selection: async (event, {xaxis}, config) => {
            // urlParams = new URLSearchParams(window.location.search);
            // startDateParam = urlParams.get('startDate');
            // endDateParam = urlParams.get('endDate');
            // resetSelection()

            formattedData = getFormattedData(startDateParam, endDateParam, historicalData);

            if (xaxis) {
              const {min, max} = xaxis; // Destructure min and max from xaxis

              // get num of data points in selected range
              const numPoints = Math.ceil((max - min) / (1000 * 60 * 60 * 24));

              // Filter the selected data based on the x-axis range
              selectedData = formattedData.filter(point => {
                return point.x >= min && point.x <= max;
              });

              // Show loading indicator
              const loadingElement = document.getElementById('loading');
              loadingElement.style.display = 'block';


              try {
                // Perform DCA calculation
                const response = await fetch('http://127.0.0.1:5000/calculate_dca', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({selected_area: selectedData})
                });

                const dcaResults = await response.json();
                console.log(dcaResults.exponential)

                // Update chart with the selected data and DCA results
                chart.updateSeries([
                  {name: "Selected Data", data: formattedData},
                  {name: "Exponential Decline", data: dcaResults.exponential},
                  {name: "Harmonic Decline", data: dcaResults.harmonic},
                  {name: "Hyperbolic Decline", data: dcaResults.hyperbolic},
                ]);
              } catch (error) {
                console.error("Error during DCA calculation:", error);
                alert("An error occurred while performing DCA analysis.");
              } finally {
                // Hide loading indicator
                loadingElement.style.display = 'none';
              }
            } else {
              console.warn("xaxis range not provided.");
            }
          }
        }
      },
      stroke: {
        curve: 'smooth', // Membuat garis halus
        width: 2, // Ketebalan garis
        dashArray: [0, 8, 5] // Pola garis (solid, dashed, dotted)
      },
      markers: {
        size: 6, // Ukuran marker
        colors: undefined, // Gunakan warna seri
        strokeColors: '#fff', // Warna border marker
        strokeWidth: 2, // Ketebalan border marker
        shape: 'circle', // Bentuk marker
        hover: {
          size: 8 // Ukuran marker saat hover
        }
      },
      series: [
        {
          name: "Historical Data",
          data: formattedData
        }
      ],
      xaxis: {
        type: 'datetime',
        labels: {
          format: 'MMM dd, yyyy'
        }
      },
      yaxis: {
        labels: {
          formatter: function (value) {
            return value.toFixed(2); // Batasi angka desimal menjadi 2 digit
          }
        }
      }
    });


    // Render the chart
    chart.render();
  } catch (error) {
    console.error("Error loading data or rendering chart:", error);
  }
}

async function loadSelectionChart() {
  // change the color here
  // change the marker color from start selection to endselection

  if (startSelection && endSelection) {
    const min = startSelection['x']
    const max = endSelection['x']
    selectedData = formattedData.filter(point => {
      return point.x >= min && point.x <= max;
    });
    // Define custom colors for markers
    const markerColors = formattedData.map(point => {
      if (point.x >= min && point.x <= max) {
        return '#FF0000'; // Highlight color for selected range
      }
      return '#0000FF'; // Default color for other points
    });
    // Show loading indicator
    const loadingElement = document.getElementById('loading');
    loadingElement.style.display = 'block';

    try {
      // Perform DCA calculation
      const response = await fetch('http://127.0.0.1:5000/calculate_dca', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({selected_area: selectedData})
      });

      const dcaResults = await response.json();
      console.log(dcaResults.exponential)

      // // Update chart with the selected data and DCA results
      // chart.updateOptions({
      //   markers: {
      //     size: 6,
      //     colors: markerColors, // Apply dynamic colors
      //     strokeColors: '#fff',
      //     strokeWidth: 2,
      //     shape: 'circle',
      //     hover: {
      //       size: 8
      //     }
      //   },
      //   series: [
      //     { name: "Selected Data", data: formattedData },
      //     { name: "Exponential Decline", data: dcaResults.exponential },
      //     { name: "Harmonic Decline", data: dcaResults.harmonic },
      //     { name: "Hyperbolic Decline", data: dcaResults.hyperbolic },
      //   ]
      // });

      chart.updateSeries([
        {name: "Selected Data", data: formattedData},
        {name: "Exponential Decline", data: dcaResults.exponential},
        {name: "Harmonic Decline", data: dcaResults.harmonic},
        {name: "Hyperbolic Decline", data: dcaResults.hyperbolic},
        {name: "Selected Range", data: selectedData},
      ]);
    } catch (error) {
      console.error("Error during DCA calculation:", error);
      alert("An error occurred while performing DCA analysis.");
    } finally {
      // Hide loading indicator
      loadingElement.style.display = 'none';
    }
  }
}


function resetSelection() {
  chart.updateOptions({
    chart: {
      selection: {
        xaxis: {
          min: undefined,
          max: undefined
        }
      }
    }
  })
}

// Handle Date Filter
document.getElementById('filter-btn').addEventListener('click', async () => {
  const startDate = document.getElementById('start-date').value;
  const endDate = document.getElementById('end-date').value;

  if (!startDate || !endDate) {
    alert("Please select both start and end dates.");
    return;
  }
  resetSelection()
  const url = new URL(window.location.href);
  url.searchParams.set('startDate', startDate);
  url.searchParams.set('endDate', endDate);
  startDateParam = startDate;
  endDateParam = endDate;
  window.history.pushState({}, '', url);

  try {
    const response = await fetch('http://127.0.0.1:5000/get_data');
    const historicalData = await response.json();

    const filteredData = historicalData.filter(item => {
      const date = new Date(item.x);
      return date >= new Date(startDate) && date <= new Date(endDate);
    });

    formattedData = filteredData.map(item => ({
      x: new Date(item.x).getTime(),
      y: item.y || 0
    }));

    chart.updateSeries([
      {
        name: "Filtered Data",
        data: formattedData
      }
    ]);
  } catch (error) {
    console.error("Error filtering data:", error);
  }
});

document.getElementById('clear-filter-btn').addEventListener('click', async () => {
  resetSelection()

  const url = new URL(window.location.href);
  url.searchParams.delete('startDate');
  url.searchParams.delete('endDate');
  window.history.pushState({}, '', url);
  const startDateInput = document.getElementById('start-date');
  const endDateInput = document.getElementById('end-date');

  startDateInput.value = '';
  endDateInput.value = '';
  resetSelection()

  try {
    const response = await fetch('http://127.0.0.1:5000/get_data');
    const historicalData = await response.json();
    // Default filter: Last year
    const today = new Date();
    const oneYearAgo = new Date(today);
    oneYearAgo.setFullYear(today.getFullYear() - 1);

    const filteredData = historicalData.filter(item => {
      const date = new Date(item.x);
      return date >= oneYearAgo && date <= today;
    });

    chart.updateSeries([
      {
        name: "Historical Data",
        data: filteredData
      }
    ]);
  } catch (error) {
    console.error("Error clearing filter:", error);
  }
})


// Handle DCA Analysis
document.getElementById('analyze-btn').addEventListener('click', async () => {
  if (selectedData.length === 0) {
    alert("Please select an area on the chart for analysis.");
    return;
  }

  const response = await fetch('http://127.0.0.1:5000/calculate_dca', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({selected_area: selectedData})
  });

  const dcaResults = await response.json();

  chart.updateSeries([
    {name: "Historical Data", data: selectedData},
    {name: "Exponential Decline", data: dcaResults.exponential},
    {name: "Harmonic Decline", data: dcaResults.harmonic},
    {name: "Hyperbolic Decline", data: dcaResults.hyperbolic},
  ]);
});

// Load Chart on Page Load
loadChart();
