let chart;
let selectedData = [];

// Initialize Chart
async function loadChart() {
  try {
    const response = await fetch('http://127.0.0.1:5000/get_data');
    const historicalData = await response.json();

    // Get the current date and one year ago
    const today = new Date();
    const oneYearAgo = new Date(today);
    oneYearAgo.setFullYear(today.getFullYear() - 1);

    // Filter data for the last year
    const defaultData = historicalData.filter(item => {
      const date = new Date(item.x);
      return date >= oneYearAgo && date <= today;
    });

    const formattedData = defaultData.map(item => ({
      x: new Date(item.x).getTime(),
      y: item.y || 0
    }));

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
          enabled: true, // Ensure selection is globally enabled
          type: 'x', // Restrict selection to x-axis
        },
        events: {
          selection: async (event, chartContext, config) => {
            if (config.xaxis) {
              selectedData = formattedData.filter(
                (point) =>
                  point.x >= config.xaxis.min &&
                  point.x <= config.xaxis.max
              );

              console.log("Selected Data:", selectedData);

              // Show loading indicator
              const loadingElement = document.getElementById('loading');
              loadingElement.style.display = 'block';

              // Perform DCA Calculation
              try {
                const response = await fetch('http://127.0.0.1:5000/calculate_dca', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({ selected_area: selectedData })
                });

                const dcaResults = await response.json();

                // Update chart with DCA results
                chart.updateSeries([
                  { name: "Historical Data", data: selectedData },
                  { name: "Exponential Decline", data: dcaResults.exponential },
                  { name: "Harmonic Decline", data: dcaResults.harmonic },
                  { name: "Hyperbolic Decline", data: dcaResults.hyperbolic },
                ]);
              } catch (error) {
                console.error("Error performing DCA calculation:", error);
                alert("An error occurred while processing DCA.");
              } finally {
                // Hide loading indicator
                loadingElement.style.display = 'none';
              }
            } else {
              console.warn("No xaxis selection range provided.");
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

// Handle Date Filter
document.getElementById('filter-btn').addEventListener('click', async () => {
  const startDate = document.getElementById('start-date').value;
  const endDate = document.getElementById('end-date').value;

  if (!startDate || !endDate) {
    alert("Please select both start and end dates.");
    return;
  }

  try {
    const response = await fetch('http://127.0.0.1:5000/get_data');
    const historicalData = await response.json();

    const filteredData = historicalData.filter(item => {
      const date = new Date(item.x);
      return date >= new Date(startDate) && date <= new Date(endDate);
    });

    const formattedData = filteredData.map(item => ({
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
    body: JSON.stringify({ selected_area: selectedData })
  });

  const dcaResults = await response.json();

  chart.updateSeries([
    { name: "Historical Data", data: selectedData },
    { name: "Exponential Decline", data: dcaResults.exponential },
    { name: "Harmonic Decline", data: dcaResults.harmonic },
    { name: "Hyperbolic Decline", data: dcaResults.hyperbolic },
  ]);
});

// Load Chart on Page Load
loadChart();
